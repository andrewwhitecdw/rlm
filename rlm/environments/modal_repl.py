import json
import threading
import time

import modal
import requests

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.types import REPLResult, RLMChatCompletion
from rlm.environments.base_env import IsolatedEnv
from rlm.environments.constants import APT_PACKAGES, PIP_PACKAGES

# =============================================================================
# Default Modal Image
# =============================================================================


def get_default_image() -> modal.Image:
    """
    Build a default Modal image with common libraries for data science,
    math, and general Python work.
    """
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(*APT_PACKAGES)
        .pip_install(*PIP_PACKAGES)
    )


# =============================================================================
# Execution Script (runs inside the sandbox for each code block)
# =============================================================================


def _build_exec_script(code: str, broker_port: int = 8080, depth: int = 1) -> str:
    from rlm.environments._exec_templates import build_broker_exec_script

    return build_broker_exec_script(code, broker_port, depth)


class ModalREPL(IsolatedEnv):
    """
    Modal REPL environment that runs Python code in a Modal Sandbox.

    Uses Modal tunnels for LLM communication:
    - Sandbox runs a broker server exposed via encrypted_ports
    - ModalREPL polls the broker for pending LLM requests
    - ModalREPL forwards requests to the LM handler and posts responses back
    """

    BROKER_PORT = 8080

    def __init__(
        self,
        app_name: str = "rlm-sandbox",
        image: modal.Image | None = None,
        timeout: int = 600,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        persistent: bool = False,
        depth: int = 1,
        **kwargs,
    ):
        if persistent:
            raise NotImplementedError(
                "Persistent REPLs are currently not supported for environment: ModalREPL"
            )
        super().__init__(persistent=persistent, depth=depth, **kwargs)

        self.app_name = app_name
        self.timeout = timeout
        self.lm_handler_address = lm_handler_address

        self.image = image or get_default_image()

        self.app = None
        self.sandbox = None
        self.broker_process = None
        self.broker_url: str | None = None
        self.poller_thread: threading.Thread | None = None
        self.poller_stop = threading.Event()
        self.pending_llm_calls: list[RLMChatCompletion] = []
        self._calls_lock = threading.Lock()

        self.setup()

        if context_payload is not None:
            self.load_context(context_payload)

        if setup_code:
            self.execute_code(setup_code)

    def setup(self):
        """Create the Modal app, sandbox, broker, and start polling."""
        self.app = modal.App.lookup(self.app_name, create_if_missing=True)

        # Create sandbox with encrypted port for broker
        self.sandbox = modal.Sandbox.create(
            app=self.app,
            image=self.image,
            timeout=self.timeout,
            encrypted_ports=[self.BROKER_PORT],
        )

        # Start the broker server in the sandbox
        from rlm.environments._exec_templates import build_broker_script

        self.broker_process = self.sandbox.exec(
            "python",
            "-c",
            build_broker_script(self.BROKER_PORT),
        )

        # Wait for broker to be ready
        self._wait_for_broker()

        # Get the tunnel URL
        tunnels = self.sandbox.tunnels()
        if self.BROKER_PORT in tunnels:
            self.broker_url = tunnels[self.BROKER_PORT].url

        # Start polling thread if we have an LM handler
        if self.lm_handler_address and self.broker_url:
            self.poller_stop.clear()
            self.poller_thread = threading.Thread(target=self._poll_broker, daemon=True)
            self.poller_thread.start()

    def _wait_for_broker(self, max_attempts: int = 30):
        """Wait for the broker to be ready by polling its health endpoint."""
        health_check = (
            f"import requests; "
            f"r = requests.get('http://127.0.0.1:{self.BROKER_PORT}/health', timeout=2); "
            f"print(r.text)"
        )
        for _ in range(max_attempts):
            time.sleep(0.5)
            try:
                proc = self.sandbox.exec("python", "-c", health_check)
                stdout = proc.stdout.read()
                if "ok" in stdout.lower():
                    return
            except Exception:
                pass
        raise RuntimeError("Broker failed to start within the expected time")

    def _poll_broker(self):
        """Poll the broker for pending LLM requests and handle them."""
        while not self.poller_stop.is_set():
            try:
                # Get pending requests
                resp = requests.get(
                    f"{self.broker_url}/pending",
                    timeout=5,
                )
                pending = resp.json().get("pending", [])

                for item in pending:
                    request_id = item["id"]
                    req_data = item["request"]

                    # Handle the request
                    response = self._handle_llm_request(req_data)

                    # Send response back
                    requests.post(
                        f"{self.broker_url}/respond",
                        json={"id": request_id, "response": response},
                        timeout=10,
                    )

            except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError):
                pass

            time.sleep(0.1)

    def _handle_llm_request(self, req_data: dict) -> dict:
        """Handle an LLM request from the sandbox."""
        req_type = req_data.get("type")
        model = req_data.get("model")

        if req_type == "single":
            prompt = req_data.get("prompt")
            request = LMRequest(prompt=prompt, model=model, depth=self.depth)
            response = send_lm_request(self.lm_handler_address, request)

            if not response.success:
                return {"error": response.error}

            # Track the call
            with self._calls_lock:
                self.pending_llm_calls.append(response.chat_completion)

            return {"response": response.chat_completion.response}

        elif req_type == "batched":
            prompts = req_data.get("prompts", [])
            responses = send_lm_request_batched(
                self.lm_handler_address, prompts, model=model, depth=self.depth
            )

            results = []
            for resp in responses:
                if not resp.success:
                    results.append(f"Error: {resp.error}")
                else:
                    with self._calls_lock:
                        self.pending_llm_calls.append(resp.chat_completion)
                    results.append(resp.chat_completion.response)

            return {"responses": results}

        return {"error": "Unknown request type"}

    def load_context(self, context_payload: dict | list | str):
        """Load context into the sandbox environment."""
        if isinstance(context_payload, str):
            escaped = context_payload.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
            context_code = f'context = """{escaped}"""'
        else:
            context_json = json.dumps(context_payload)
            escaped_json = context_json.replace("\\", "\\\\").replace("'", "\\'")
            context_code = f"import json; context = json.loads('{escaped_json}')"

        self.execute_code(context_code)

    def execute_code(self, code: str) -> REPLResult:
        """Execute code in the Modal sandbox and return result."""
        start_time = time.perf_counter()

        # Clear pending LLM calls
        with self._calls_lock:
            self.pending_llm_calls.clear()

        # Build and execute the script
        script = _build_exec_script(code, self.BROKER_PORT, self.depth)
        process = self.sandbox.exec("python", "-c", script)

        # Read output
        stdout = process.stdout.read()
        stderr = process.stderr.read()

        # Collect LLM calls made during this execution
        with self._calls_lock:
            pending_calls = self.pending_llm_calls.copy()
            self.pending_llm_calls.clear()

        execution_time = time.perf_counter() - start_time

        # Parse the JSON result
        try:
            lines = stdout.strip().split("\n")
            result_json = lines[-1] if lines else "{}"
            result = json.loads(result_json)

            return REPLResult(
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", "") + stderr,
                locals=result.get("locals", {}),
                execution_time=execution_time,
                rlm_calls=pending_calls,
            )
        except json.JSONDecodeError:
            return REPLResult(
                stdout=stdout,
                stderr=stderr or "Failed to parse execution result",
                locals={},
                execution_time=execution_time,
                rlm_calls=pending_calls,
            )

    def cleanup(self):
        """Terminate the sandbox and stop polling."""
        # Stop the poller thread
        if self.poller_thread is not None:
            self.poller_stop.set()
            self.poller_thread.join(timeout=2)
            self.poller_thread = None

        if self.sandbox is not None:
            try:
                self.sandbox.terminate()
            except Exception:
                pass
            self.sandbox = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


