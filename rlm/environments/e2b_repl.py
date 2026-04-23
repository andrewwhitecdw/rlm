"""
E2B REPL environment that runs Python code in E2B sandboxes.

Uses the E2B Code Interpreter SDK (https://e2b.dev/docs) for sandbox management.
Follows the same HTTP broker pattern as ModalREPL for LLM communication.
"""

import json
import threading
import time
from typing import Any

import requests
from e2b_code_interpreter import Sandbox

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.types import REPLResult, RLMChatCompletion
from rlm.environments.base_env import IsolatedEnv

# =============================================================================
# Execution Script (runs inside the sandbox for each code block)
# =============================================================================


def _build_exec_script(code: str, broker_port: int = 8888, depth: int = 1) -> str:
    from rlm.environments._exec_templates import build_broker_exec_script

    return build_broker_exec_script(code, broker_port, depth)


class E2BREPL(IsolatedEnv):
    """
    E2B REPL environment that runs Python code in E2B sandboxes.

    Uses E2B's public URL feature for LLM communication:
    - Sandbox runs a broker server exposed via get_host()
    - E2BREPL polls the broker for pending LLM requests
    - E2BREPL forwards requests to the LM handler and posts responses back
    """

    BROKER_PORT = 8889  # Use 8889 since E2B Code Interpreter uses 8888 for Jupyter

    def __init__(
        self,
        timeout: int = 300,  # 5 minutes default (E2B uses seconds)
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        persistent: bool = False,
        **kwargs: Any,
    ):
        if persistent:
            raise NotImplementedError(
                "Persistent REPLs are currently not supported for environment: E2BREPL"
            )
        super().__init__(persistent=persistent, **kwargs)

        self.timeout = timeout
        self.lm_handler_address = lm_handler_address

        # Sandbox state
        self.sandbox: Sandbox | None = None
        self.sandbox_id: str | None = None
        self.broker_url: str | None = None
        self.broker_process = None

        # Polling thread for LLM requests
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
        """Create the E2B sandbox, broker, and start polling."""
        # Create the sandbox using the recommended Sandbox.create() API
        self.sandbox = Sandbox.create(timeout=self.timeout)
        self.sandbox_id = self.sandbox.sandbox_id

        # Install dependencies for the broker
        self.sandbox.commands.run("pip install flask requests dill")

        # Write the broker script to the sandbox
        from rlm.environments._exec_templates import build_broker_script

        self.sandbox.files.write("/tmp/broker.py", build_broker_script(self.BROKER_PORT))

        # Start the broker as a background process
        self.broker_process = self.sandbox.commands.run(
            "python /tmp/broker.py",
            background=True,
        )

        # Wait for broker to be ready
        self._wait_for_broker()

        # Get the public URL for the broker port
        host = self.sandbox.get_host(self.BROKER_PORT)
        self.broker_url = f"https://{host}"

        # Start polling thread if we have an LM handler
        if self.lm_handler_address and self.broker_url:
            self.poller_stop.clear()
            self.poller_thread = threading.Thread(target=self._poll_broker, daemon=True)
            self.poller_thread.start()

    def _wait_for_broker(self, max_attempts: int = 30):
        """Wait for the broker to be ready by checking health endpoint."""
        health_check_cmd = (
            f'python -c "import requests; '
            f"r = requests.get('http://127.0.0.1:{self.BROKER_PORT}/health', timeout=2); "
            f'print(r.text)"'
        )

        for _ in range(max_attempts):
            time.sleep(1)
            try:
                result = self.sandbox.commands.run(health_check_cmd)
                if result.stdout and "ok" in result.stdout.lower():
                    return
            except Exception:
                pass

        raise RuntimeError("Broker failed to start within timeout")

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
            responses = send_lm_request_batched(self.lm_handler_address, prompts, model=model, depth=self.depth)

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
        """Execute code in the E2B sandbox and return result."""
        start_time = time.perf_counter()

        # Clear pending LLM calls
        with self._calls_lock:
            self.pending_llm_calls.clear()

        # Build and write the script to sandbox
        script = _build_exec_script(code, self.BROKER_PORT, self.depth)
        self.sandbox.files.write("/tmp/run_script.py", script)

        # Run the script
        result = self.sandbox.commands.run("python /tmp/run_script.py", timeout=600)
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # Collect LLM calls made during this execution
        with self._calls_lock:
            pending_calls = self.pending_llm_calls.copy()
            self.pending_llm_calls.clear()

        execution_time = time.perf_counter() - start_time

        # Parse the JSON result
        try:
            lines = stdout.strip().split("\n")
            result_json = lines[-1] if lines else "{}"
            parsed = json.loads(result_json)

            return REPLResult(
                stdout=parsed.get("stdout", ""),
                stderr=parsed.get("stderr", "") + stderr,
                locals=parsed.get("locals", {}),
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

        # Kill the sandbox
        if self.sandbox is not None:
            try:
                self.sandbox.kill()
            except Exception:
                pass
            self.sandbox = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


