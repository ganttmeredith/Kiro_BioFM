"""AgentCore runtime entrypoint.

Deployed via direct_code_deploy: the contents of backend/ are zipped and
placed at /var/task/. So agentcore_app.py, models.py, services/, routers/
are all siblings at /var/task/ — there is no 'backend' package prefix.

We fix this by:
1. Adding /var/task to sys.path so bare imports (models, services/) resolve.
2. Registering a 'backend' package alias in sys.modules so all existing
   `from backend.xxx` imports across services/routers work unchanged.
3. Adding umap_retrieval_pkg to sys.path before any service imports fire.
4. Deferring ChatService instantiation to first invocation to avoid
   import-time failures from heavy deps (torch, umap-learn, etc.).
"""

import json
import sys
import types
from pathlib import Path

# ── 1. Path setup — must happen before ANY other local imports ────────────────
_HERE = Path(__file__).resolve().parent

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# umap_retrieval package — copied into backend/ by build.sh → /app/umap_retrieval_pkg/
# (also works for direct_code_deploy where it lands at /var/task/umap_retrieval_pkg/)
_UMAP_PKG_DIR = _HERE / "umap_retrieval_pkg"
if _UMAP_PKG_DIR.exists() and str(_UMAP_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_UMAP_PKG_DIR))

# ── 2. Register 'backend' as a package alias for the current directory ────────
# All service/router files use `from backend.models import ...` etc.
# Since the zip places everything flat at /var/task/, we register a fake
# 'backend' package whose __path__ points here so those imports resolve.
if "backend" not in sys.modules:
    _pkg = types.ModuleType("backend")
    _pkg.__path__ = [str(_HERE)]          # type: ignore[attr-defined]
    _pkg.__package__ = "backend"
    _pkg.__spec__ = None                  # type: ignore[assignment]
    sys.modules["backend"] = _pkg

# ── 3. AgentCore app ──────────────────────────────────────────────────────────
from bedrock_agentcore import BedrockAgentCoreApp  # noqa: E402
from backend.models import AppContext, ChatMessage  # noqa: E402

app = BedrockAgentCoreApp()

# Defer heavy service imports to first invocation — avoids startup crashes
# if optional deps (torch, umap-learn) are slow or partially unavailable.
_chat_service = None


def _get_chat_service():
    global _chat_service
    if _chat_service is None:
        from backend.services.chat_service import ChatService  # noqa: PLC0415
        _chat_service = ChatService()
    return _chat_service


# ── 4. Invocation handler ─────────────────────────────────────────────────────

@app.entrypoint
def handler(payload: dict, context) -> dict:  # noqa: ARG001
    """AgentCore invocation handler.

    Payload fields:
      - prompt (str): user message text
      - messages (list[dict]): optional conversation history
        Each item: {"role": "user"|"assistant", "content": "<text>"}
      - app_context (dict): optional AppContext fields
        (active_filters, query_patient_id, alpha, beta, gamma, etc.)

    Returns:
      - response (str): full assistant reply
      - artifacts (list): optional rendered artifact objects
    """
    prompt: str = payload.get("prompt", "")
    messages_raw: list = payload.get("messages", [])
    app_context_raw: dict = payload.get("app_context", {})

    if not messages_raw:
        messages_raw = [{"role": "user", "content": prompt}]

    messages = [ChatMessage(**m) for m in messages_raw]
    app_context = AppContext(**app_context_raw) if app_context_raw else AppContext()

    full_response = ""
    artifacts = []

    for chunk in _get_chat_service().stream_response(messages, app_context):
        if not chunk.startswith("data: "):
            continue
        raw = chunk[len("data: "):].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if "delta" in obj:
            full_response += obj["delta"]
        elif "artifact" in obj:
            artifacts.append(obj["artifact"])

    result: dict = {"response": full_response}
    if artifacts:
        result["artifacts"] = artifacts
    return result
