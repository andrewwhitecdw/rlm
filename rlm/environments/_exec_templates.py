"""
Shared execution script templates for isolated REPL environments.

These functions build Python source code strings that are executed inside
sandboxed environments (Modal, Prime, E2B, Daytona). The generated scripts
communicate with a local HTTP broker for LLM queries and persist state via dill.
"""

import base64
import textwrap


def build_broker_script(broker_port: int) -> str:
    """
    Build the Flask broker script that runs inside the sandbox.

    The broker provides /health, /enqueue, /pending, and /respond endpoints
    for host-sandbox communication.
    """
    return textwrap.dedent(
        '''
        import json
        import threading
        import uuid
        from flask import Flask, request, jsonify

        app = Flask(__name__)

        # Request queue: {{request_id: {{"request": {{...}}, "response": None, "event": Event}}}}
        pending_requests = {{}}
        lock = threading.Lock()

        @app.route("/health")
        def health():
            return jsonify({{"status": "ok"}})

        @app.route("/enqueue", methods=["POST"])
        def enqueue():
            """Called by sandbox code to submit an LLM request and wait for response."""
            data = request.json
            request_id = str(uuid.uuid4())
            event = threading.Event()

            with lock:
                pending_requests[request_id] = {{
                    "request": data,
                    "response": None,
                    "event": event,
                }}

            # Wait for response (with timeout)
            event.wait(timeout=300)

            with lock:
                entry = pending_requests.pop(request_id, None)

            if entry and entry["response"] is not None:
                return jsonify(entry["response"])
            else:
                return jsonify({{"error": "Request timed out"}}), 504

        @app.route("/pending")
        def get_pending():
            """Called by host poller to get pending requests."""
            with lock:
                pending = [
                    {{"id": rid, "request": entry["request"]}}
                    for rid, entry in pending_requests.items()
                    if entry["response"] is None
                ]
            return jsonify({{"pending": pending}})

        @app.route("/respond", methods=["POST"])
        def respond():
            """Called by host poller to submit a response."""
            data = request.json
            request_id = data.get("id")
            response = data.get("response")

            with lock:
                if request_id in pending_requests:
                    pending_requests[request_id]["response"] = response
                    pending_requests[request_id]["event"].set()
                    return jsonify({{"status": "ok"}})

            return jsonify({{"error": "Request not found"}}), 404

        if __name__ == "__main__":
            app.run(host="0.0.0.0", port={broker_port}, threaded=True)
        '''
    ).format(broker_port=broker_port)


def build_broker_exec_script(
    code: str,
    broker_port: int,
    depth: int,
    custom_tools_code: str = "",
) -> str:
    """
    Build a script that executes code with state persistence inside a sandbox.

    LLM queries go through a local HTTP broker. State is persisted across
    executions via dill serialization to /tmp/rlm_state.dill.

    Args:
        code: The Python code to execute.
        broker_port: Port for the broker server.
        depth: Depth level for LLM requests.
        custom_tools_code: Optional Python code to inject for custom tools.
    """
    code_b64 = base64.b64encode(code.encode()).decode()

    return textwrap.dedent(
        f'''
import sys
import io
import json
import base64
import traceback
import os
import requests

try:
    import dill
except ImportError:
    import pickle as dill

# =============================================================================
# LLM Query Functions (via local broker)
# =============================================================================

BROKER_URL = "http://127.0.0.1:{broker_port}"

def llm_query(prompt, model=None):
    """Query the LM via the broker."""
    try:
        response = requests.post(
            f"{{BROKER_URL}}/enqueue",
            json={{"type": "single", "prompt": prompt, "model": model, "depth": {depth}}},
            timeout=300,
        )
        data = response.json()
        if data.get("error"):
            return f"Error: {{data['error']}}"
        return data.get("response", "Error: No response")
    except Exception as e:
        return f"Error: LM query failed - {{e}}"


def llm_query_batched(prompts, model=None):
    """Query the LM with multiple prompts."""
    try:
        response = requests.post(
            f"{{BROKER_URL}}/enqueue",
            json={{"type": "batched", "prompts": prompts, "model": model, "depth": {depth}}},
            timeout=300,
        )
        data = response.json()
        if data.get("error"):
            return [f"Error: {{data['error']}}"] * len(prompts)
        return data.get("responses", ["Error: No response"] * len(prompts))
    except Exception as e:
        return [f"Error: LM query failed - {{e}}"] * len(prompts)


def rlm_query(prompt, model=None):
    """Recursive RLM query (falls back to plain llm_query in isolated environments)."""
    return llm_query(prompt, model=model)


def rlm_query_batched(prompts, model=None):
    """Recursive RLM batched query (falls back to plain llm_query_batched in isolated environments)."""
    return llm_query_batched(prompts, model=model)


# =============================================================================
# State Management
# =============================================================================

STATE_FILE = "/tmp/rlm_state.dill"

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "rb") as f:
                return dill.load(f)
        except:
            pass
    return {{}}

def save_state(state):
    clean_state = {{}}
    for k, v in state.items():
        if k.startswith("_"):
            continue
        try:
            dill.dumps(v)
            clean_state[k] = v
        except:
            pass
    with open(STATE_FILE, "wb") as f:
        dill.dump(clean_state, f)

def serialize_locals(state):
    result = {{}}
    for k, v in state.items():
        if k.startswith("_"):
            continue
        try:
            result[k] = repr(v)
        except:
            result[k] = f"<{{type(v).__name__}}>"
    return result

# =============================================================================
# Execution
# =============================================================================

_locals = load_state()

def FINAL_VAR(variable_name):
    variable_name = variable_name.strip().strip("\\"\\'")
    if variable_name in _locals:
        return str(_locals[variable_name])
    available = [k for k in _locals.keys() if not k.startswith("_")]
    if available:
        return f"Error: Variable '{{variable_name}}' not found. Available variables: {{available}}. You must create and assign a variable BEFORE calling FINAL_VAR on it."
    return f"Error: Variable '{{variable_name}}' not found. No variables have been created yet. You must create and assign a variable in a REPL block BEFORE calling FINAL_VAR on it."

def SHOW_VARS():
    available = {{k: type(v).__name__ for k, v in _locals.items() if not k.startswith("_")}}
    if not available:
        return "No variables created yet. Use ```repl``` blocks to create variables."
    return f"Available variables: {{available}}"

_globals = {{
    "__builtins__": __builtins__,
    "__name__": "__main__",
    "llm_query": llm_query,
    "llm_query_batched": llm_query_batched,
    "rlm_query": rlm_query,
    "rlm_query_batched": rlm_query_batched,
    "FINAL_VAR": FINAL_VAR,
    "SHOW_VARS": SHOW_VARS,
}}

# =============================================================================
# Custom Tools Injection
# =============================================================================
{custom_tools_code}

code = base64.b64decode("{code_b64}").decode()

stdout_buf = io.StringIO()
stderr_buf = io.StringIO()
old_stdout, old_stderr = sys.stdout, sys.stderr

try:
    sys.stdout = stdout_buf
    sys.stderr = stderr_buf
    combined = {{**_globals, **_locals}}
    exec(code, combined, combined)
    for key, value in combined.items():
        if key not in _globals and not key.startswith("_"):
            _locals[key] = value
except Exception as e:
    traceback.print_exc(file=stderr_buf)
finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr

# Restore scaffold aliases if overwritten by executed code
if "context_0" in _locals:
    _locals["context"] = _locals["context_0"]
if "history_0" in _locals:
    _locals["history"] = _locals["history_0"]

save_state(_locals)

result = {{
    "stdout": stdout_buf.getvalue(),
    "stderr": stderr_buf.getvalue(),
    "locals": serialize_locals(_locals),
}}
print(json.dumps(result))
'''
    )
