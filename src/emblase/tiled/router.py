"""FastAPI router that serves the embedding UI plugin's static assets and
proxies chat requests to the AmSC chat service.

Registered via Tiled's ``routers:`` config and served at ``/custom/emblase/``.

Chat credentials are read from environment variables so they never appear in
version-controlled config files or reach the browser:

  EMBLASE_CHATAPP_URL    Base URL of the AmSC chat service
                         (default: https://chat-amsc-dev.nsls2.bnl.gov)
  EMBLASE_CHATAPP_TOKEN  Bearer token for the chat service
  EMBLASE_CHATAPP_MODEL  Default model name
                         (default: openai/gpt-oss-120b)
"""

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Load .env from the working directory (or any parent) so EMBLASE_* vars are
# available even when the server is started without explicitly sourcing .env.
# Already-set env vars take priority (load_dotenv won't override them).
load_dotenv()

router = APIRouter(prefix="/emblase", tags=["emblase"])

STATIC_DIR = Path(__file__).resolve().parent / "static"

_CHAT_URL = os.environ.get(
    "EMBLASE_CHATAPP_URL", "https://chat-amsc-dev.nsls2.bnl.gov"
).rstrip("/")
_CHAT_TOKEN = os.environ.get("EMBLASE_CHATAPP_TOKEN", "")
_CHAT_MODEL = os.environ.get("EMBLASE_CHATAPP_MODEL", "openai/gpt-oss-20b")

if not _CHAT_TOKEN:
    import warnings

    warnings.warn(
        "EMBLASE_CHATAPP_TOKEN is not set — chat proxy endpoints will fail. "
        "Add it to your .env file or set it in the environment.",
        stacklevel=1,
    )

_AMSC_HEADERS = {
    "Authorization": f"Bearer {_CHAT_TOKEN}",
    "Content-Type": "application/json",
}

# In-memory store mapping chat_session_id → message_id of the hidden priming
# message. Used to filter it out when returning history to the browser.
_priming_message_ids: dict[str, int] = {}


def _build_priming_message(node_path: str, metadata: dict) -> str:
    """Build a context message describing the dataset for the LLM."""
    md = metadata.get("metadata", {})
    model_name = md.get("model_name", "unknown")
    embedding_dim = md.get("embedding_dim", "unknown")
    projection_dim = md.get("projection_dim", 2)
    description = md.get("description", "")
    param_specs = md.get("param_specs", {})

    params_lines = []
    for name, spec in param_specs.items():
        dtype = spec.get("dtype", "")
        units = spec.get("units", "")
        source = spec.get("source", "")
        params_lines.append(f"  - {name} ({dtype}, units: {units}, source: {source})")

    lines = [
        # Marker used by the history filter to strip this message after a restart.
        # The LLM treats it as an innocuous comment and ignores it.
        "<!-- emblase-priming -->",
        "",
        "The user is currently viewing a LatentSpaceEmbedding container in the Emblase Latent Space Explorer.",
        f"The Tiled path for this specific container is: {node_path}",
        "Please treat this as the primary dataset for this conversation — do not confuse it with other nodes you may find in Tiled.",
        "The only exception are the source datasets used to produce these embeddings (the `path` column in the `_index` table).",
        "",
        "Container metadata:",
        f"  - Embedding model: {model_name}",
        f"  - Embedding dimensionality: {embedding_dim}D, projected to {projection_dim}D for visualisation",
    ]
    if description:
        lines.append(f"  - Description: {description}")
    if params_lines:
        lines.append("  - Experimental parameters tracked per sample:")
        lines.extend(f"  {line}" for line in params_lines)
    lines += [
        "",
        f"You can query the contents of this container directly using your Tiled tools (path: {node_path}).",
        "",
        "Important: keep answers concise, up to 3 sentences. Expect follow-up questions.",
        "Avoid large headers and excessive structure — short paragraphs or brief bullet points are preferred.",
    ]
    return "\n".join(lines)


async def _fetch_tiled_metadata(request: Request, node_path: str) -> dict:
    """Fetch metadata for a Tiled node using the current request's auth token."""
    # Reuse the Authorization header from the incoming browser request so we
    # don't need a separate Tiled service account.
    auth = request.headers.get("Authorization", "")
    tiled_base = str(request.base_url).rstrip("/")
    url = f"{tiled_base}/api/v1/metadata/{node_path}"
    async with httpx.AsyncClient(timeout=10) as client:
        res = await client.get(url, headers={"Authorization": auth})
    if res.status_code == 200:
        return res.json().get("data", {}).get("attributes", {})
    return {}


async def _send_priming_message(
    session_id: str | None, priming_text: str
) -> str | None:
    """Send the hidden context message and return the resulting session_id."""
    payload = {
        "message": priming_text,
        "model_name": _CHAT_MODEL,
        "chat_session_id": session_id,
        "image_refs": [],
    }
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(
            f"{_CHAT_URL}/chat",
            json=payload,
            headers=_AMSC_HEADERS,
        )
    if res.status_code == 200:
        data = res.json()
        new_session_id = data.get("chat_session_id")
        # Record the assistant reply's message_id so we can filter both the
        # priming user turn and the assistant ack from history.
        # The API returns one message_id (the assistant reply); the user turn
        # will have message_id - 1.
        # We store both so the history endpoint can strip them.
        if new_session_id:
            msg_id = data.get("message_id")
            if msg_id is not None:
                _priming_message_ids[new_session_id] = msg_id  # assistant reply
        return new_session_id
    return session_id


class ChatRequest(BaseModel):
    message: str
    node_path: str
    chat_session_id: str | None = None


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """Proxy a chat message to the AmSC service and stream the SSE response back.

    On a new session (no chat_session_id), a hidden priming message describing
    the dataset is sent first so the LLM has context. The priming message is
    never shown to the user.

    The AmSC token never leaves the server — the browser only needs a valid
    Tiled session to reach this endpoint.
    """
    session_id = req.chat_session_id

    # New session: fetch dataset metadata and send a silent priming message.
    if not session_id:
        metadata = await _fetch_tiled_metadata(request, req.node_path)
        priming_text = _build_priming_message(req.node_path, metadata)
        session_id = await _send_priming_message(None, priming_text)

    payload = {
        "message": req.message,
        "model_name": _CHAT_MODEL,
        "chat_session_id": session_id,
        "image_refs": [],
    }

    async def stream():
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{_CHAT_URL}/chat/stream",
                json=payload,
                headers={**_AMSC_HEADERS, "Accept": "text/event-stream"},
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/chat")
async def chat(req: ChatRequest, request: Request) -> dict:
    """Non-streaming proxy: send a message and return the full assistant reply.

    Useful for scripting / debugging without SSE support.
    """
    session_id = req.chat_session_id

    if not session_id:
        metadata = await _fetch_tiled_metadata(request, req.node_path)
        priming_text = _build_priming_message(req.node_path, metadata)
        session_id = await _send_priming_message(None, priming_text)

    payload = {
        "message": req.message,
        "model_name": _CHAT_MODEL,
        "chat_session_id": session_id,
        "image_refs": [],
    }
    async with httpx.AsyncClient(timeout=120) as client:
        res = await client.post(
            f"{_CHAT_URL}/chat", json=payload, headers=_AMSC_HEADERS
        )
    res.raise_for_status()
    return res.json()


@router.get("/chat/history/{session_id}")
async def chat_history(session_id: str):
    """Fetch message history for a session from AmSC, stripping the priming messages."""
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.get(
            f"{_CHAT_URL}/sessions/{session_id}/messages",
            headers=_AMSC_HEADERS,
        )
    if res.status_code != 200:
        return []

    messages = res.json()

    # Filter out the hidden priming exchange (user context msg + assistant ack).
    # Primary filter: by stored message_id (works when server hasn't restarted).
    # Fallback filter: by content marker (works across restarts).
    priming_id = _priming_message_ids.get(session_id)
    if priming_id is not None:
        hidden_ids = {priming_id, priming_id - 1}
        messages = [m for m in messages if m.get("message_id") not in hidden_ids]
    else:
        # Fallback: strip any user message that starts with the priming marker
        messages = [
            m
            for m in messages
            if not (
                m.get("role") == "user"
                and isinstance(m.get("content"), str)
                and m["content"].startswith("<!-- emblase-priming -->")
            )
        ]

    return messages


@router.get("/main.js")
async def main_js():
    return FileResponse(
        STATIC_DIR / "main.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )
