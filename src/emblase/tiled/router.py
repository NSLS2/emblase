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
from fastapi import APIRouter
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Load .env from the working directory (or any parent) so EMBLASE_* vars are
# available even when the server is started without explicitly sourcing .env.
# Already-set env vars take priority (load_dotenv won't override them).
load_dotenv()

router = APIRouter(prefix="/emblase", tags=["emblase"])

STATIC_DIR = Path(__file__).resolve().parent / "static"

_CHAT_URL = os.environ.get("EMBLASE_CHATAPP_URL", "https://chat-amsc-dev.nsls2.bnl.gov").rstrip("/")
_CHAT_TOKEN = os.environ.get("EMBLASE_CHATAPP_TOKEN", "")
_CHAT_MODEL = os.environ.get("EMBLASE_CHATAPP_MODEL", "openai/gpt-oss-120b")

if not _CHAT_TOKEN:
    import warnings
    warnings.warn(
        "EMBLASE_CHATAPP_TOKEN is not set — chat proxy endpoints will fail. "
        "Add it to your .env file or set it in the environment.",
        stacklevel=1,
    )


class ChatRequest(BaseModel):
    message: str
    chat_session_id: str | None = None


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Proxy a chat message to the AmSC service and stream the SSE response back.

    The AmSC token never leaves the server — the browser only needs a valid
    Tiled session to reach this endpoint.
    """
    payload = {
        "message": req.message,
        "model_name": _CHAT_MODEL,
        "chat_session_id": req.chat_session_id,
        "image_refs": [],
    }
    headers = {
        "Authorization": f"Bearer {_CHAT_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    async def stream():
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{_CHAT_URL}/chat/stream",
                json=payload,
                headers=headers,
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/chat")
async def chat(req: ChatRequest) -> dict:
    """Non-streaming proxy: send a message and return the full assistant reply.

    Useful for scripting / debugging without SSE support.
    """
    payload = {
        "message": req.message,
        "model_name": _CHAT_MODEL,
        "chat_session_id": req.chat_session_id,
        "image_refs": [],
    }
    headers = {
        "Authorization": f"Bearer {_CHAT_TOKEN}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        res = await client.post(f"{_CHAT_URL}/chat", json=payload, headers=headers)
    res.raise_for_status()
    return res.json()


@router.get("/chat/history/{session_id}")
async def chat_history(session_id: str):
    """Fetch message history for a session from AmSC."""
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.get(
            f"{_CHAT_URL}/sessions/{session_id}/messages",
            headers={"Authorization": f"Bearer {_CHAT_TOKEN}"},
        )
    if res.status_code == 200:
        return res.json()
    return []


@router.get("/main.js")
async def main_js():
    return FileResponse(
        STATIC_DIR / "main.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )
