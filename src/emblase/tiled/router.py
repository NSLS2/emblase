"""FastAPI router that serves the embedding UI plugin's static assets and chat API.

Registered via Tiled's ``routers:`` config and served at ``/custom/emblase/``.
"""

import time
import uuid
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

router = APIRouter(prefix="/emblase", tags=["emblase"])

STATIC_DIR = Path(__file__).resolve().parent / "static"

# In-memory chat history, keyed by node_path. Will be replaced by a real
# service once available.
_chat_histories: dict[str, list[dict]] = defaultdict(list)

DUMMY_RESPONSES = [
    "Based on the embedding distribution, there appear to be {n} distinct clusters in this dataset.",
    "The projections show a spiral pattern in the latent space, which is consistent with the synthetic data generation process.",
    "Looking at the label distribution: the embeddings are evenly spread across categories. Consider filtering by a specific label to focus your analysis.",
    "The embedding dimensionality is {dim}D, projected down to {proj_dim}D for visualization. Some structural information may be lost in this reduction.",
    "I notice several points in the upper-right quadrant that might be outliers. You can click on them to inspect their thumbnails and annotations.",
    "To explore relationships between clusters, try zooming into the boundary regions where different label colors meet.",
    "The model '{model}' was used to generate these embeddings. Different models may produce different clustering patterns.",
]
_dummy_idx: dict[str, int] = defaultdict(int)


class ChatRequest(BaseModel):
    message: str
    node_path: str


class ChatMessage(BaseModel):
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: float


@router.post("/chat")
async def chat(req: ChatRequest) -> dict:
    """Accept a user message and return a dummy assistant response.

    In production this will proxy to an external chat service that has
    access to the data at ``node_path``.
    """
    ts = time.time()
    user_msg = {
        "id": uuid.uuid4().hex[:12],
        "role": "user",
        "content": req.message,
        "timestamp": ts,
    }
    _chat_histories[req.node_path].append(user_msg)

    # Pick a dummy response, cycling through the list
    idx = _dummy_idx[req.node_path] % len(DUMMY_RESPONSES)
    _dummy_idx[req.node_path] = idx + 1
    template = DUMMY_RESPONSES[idx]
    reply_text = template.format(
        n=5,
        dim=128,
        proj_dim=2,
        model="dummy-model",
    )

    assistant_msg = {
        "id": uuid.uuid4().hex[:12],
        "role": "assistant",
        "content": reply_text,
        "timestamp": time.time(),
    }
    _chat_histories[req.node_path].append(assistant_msg)

    return assistant_msg


@router.get("/chat/history/{path:path}")
async def chat_history(path: str) -> list[dict]:
    """Return the conversation history for a given node path."""
    return _chat_histories.get(path, [])


@router.delete("/chat/history/{path:path}")
async def clear_chat_history(path: str) -> dict:
    """Clear the conversation history for a given node path."""
    _chat_histories.pop(path, None)
    _dummy_idx.pop(path, None)
    return {"status": "cleared"}


@router.get("/main.js")
async def main_js():
    return FileResponse(
        STATIC_DIR / "main.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/ws-test")
async def ws_test_page():
    """Minimal page to debug WebSocket connections to projections stream."""
    return HTMLResponse("""<!DOCTYPE html>
<html><body>
<h3>WS Debug</h3>
<pre id="log" style="background:#eee;padding:12px;max-height:80vh;overflow:auto"></pre>
<script>
const log = document.getElementById('log');
function L(msg) { log.textContent += new Date().toISOString().slice(11,23) + ' ' + msg + '\\n'; }

const path = 'test_embed/embeddings/projections';
const wsUrl = (location.protocol==='https:'?'wss:':'ws:') + '//' + location.host
  + '/api/v1/stream/single/' + path + '?envelope_format=json';

L('Connecting to: ' + wsUrl);
const ws = new WebSocket(wsUrl);
ws.onopen = () => L('OPEN');
ws.onclose = (e) => L('CLOSE code=' + e.code + ' reason=' + e.reason);
ws.onerror = (e) => L('ERROR ' + JSON.stringify(e));
ws.onmessage = (e) => {
  const d = JSON.parse(e.data);
  L('MSG type=' + d.type + ' offset=' + JSON.stringify(d.offset) + ' shape=' + JSON.stringify(d.shape)
    + ' payload=' + (d.payload ? JSON.stringify(d.payload).slice(0,200) : 'none'));
};
</script>
</body></html>""")
