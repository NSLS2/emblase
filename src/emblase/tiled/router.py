"""FastAPI router that serves the embedding UI plugin's static assets and
proxies chat requests to the AmSC chat service.

Registered via Tiled's ``routers:`` config and served at ``/custom/emblase/``.

Chat credentials are read from environment variables so they never appear in
version-controlled config files or reach the browser:

  EMBLASE_CHATAPP_URL          Base URL of the AmSC chat service
                               (default: https://chat-amsc-dev.nsls2.bnl.gov)
  EMBLASE_CHATAPP_MODEL        Default model name
                               (default: openai/gpt-oss-20b)

OBO auth (preferred — no static token required):
  EMBLASE_ENTRA_TENANT_ID      Azure tenant ID
  EMBLASE_ENTRA_CLIENT_ID      Tiled's own Entra client ID (used as OBO actor)
  EMBLASE_ENTRA_CLIENT_SECRET  Tiled's client secret
  EMBLASE_CHATAPP_SCOPE        Scope for the chatapp's Entra app registration
                               e.g. api://<chatapp-client-id>/access_as_user

Fallback (local dev without full Entra setup):
  EMBLASE_CHATAPP_TOKEN        Static bearer token for the chat service

Security: the user's Entra access token is stored server-side in the Tiled
session JWT (HMAC-signed, HTTPS-only). The OBO exchange happens here on the
server; the browser never sees any chat service credential.
"""

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import httpx
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Load .env from the working directory (or any parent) so EMBLASE_* vars are
# available even when the server is started without explicitly sourcing .env.
# Already-set env vars take priority (load_dotenv won't override them).
load_dotenv()

log = logging.getLogger(__name__)

router = APIRouter(prefix="/emblase", tags=["emblase"])

STATIC_DIR = Path(__file__).resolve().parent / "static"

_CHAT_URL = os.environ.get("EMBLASE_CHATAPP_URL", "https://chat-amsc-dev.nsls2.bnl.gov").rstrip("/")
_CHAT_MODEL = os.environ.get("EMBLASE_CHATAPP_MODEL", "openai/gpt-oss-20b")
_CHAT_TOKEN = os.environ.get("EMBLASE_CHATAPP_TOKEN", "")  # fallback for local dev

_TENANT_ID = os.environ.get("EMBLASE_ENTRA_TENANT_ID", "")
_CLIENT_ID = os.environ.get("EMBLASE_ENTRA_CLIENT_ID", "")
_CLIENT_SECRET = os.environ.get("EMBLASE_ENTRA_CLIENT_SECRET", "")
_CHATAPP_SCOPE = os.environ.get("EMBLASE_CHATAPP_SCOPE", "")

_OBO_ENABLED = all([_TENANT_ID, _CLIENT_ID, _CLIENT_SECRET, _CHATAPP_SCOPE])

if not _OBO_ENABLED and not _CHAT_TOKEN:
    import warnings

    warnings.warn(
        "Neither OBO env vars (EMBLASE_ENTRA_TENANT_ID / EMBLASE_ENTRA_CLIENT_ID / "
        "EMBLASE_ENTRA_CLIENT_SECRET / EMBLASE_CHATAPP_SCOPE) nor EMBLASE_CHATAPP_TOKEN "
        "are set. Chat proxy endpoints will return 401.",
        stacklevel=1,
    )

_TOKEN_URL = (
    f"https://login.microsoftonline.com/{_TENANT_ID}/oauth2/v2.0/token" if _TENANT_ID else ""
)
_JSON_CT = {"Content-Type": "application/json"}

# ---------------------------------------------------------------------------
# OBO token cache
# TTL = 45 min (OBO tokens are valid for ~1 h; we refresh with 15 min margin).
# Key = SHA-256 of the user's Entra access token (never stored in plaintext).
# maxsize = 1024 covers ~1000 concurrent users with negligible memory cost.
# ---------------------------------------------------------------------------
_obo_cache: TTLCache = TTLCache(maxsize=1024, ttl=45 * 60)
_obo_cache_lock = threading.Lock()

# In-memory store mapping chat_session_id → message_id of the hidden priming
# message. Used to filter it out when returning history to the browser.
# NOTE: this is per-process; multi-worker deployments fall back to the
# content-marker filter for sessions created on a different worker.
_priming_message_ids: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Tiled session state dependency
# ---------------------------------------------------------------------------

try:
    from tiled.server.authentication import get_session_state as _tiled_get_session_state

    _session_state_dep = Depends(_tiled_get_session_state)
except ImportError:
    _tiled_get_session_state = None
    _session_state_dep = None

# If OBO is configured but we can't inject session state, the proxy will
# always fall through to the static token (or fail). Raise at startup so the
# misconfiguration is immediately visible rather than silently degrading.
if _OBO_ENABLED and _session_state_dep is None:
    raise RuntimeError(
        "EMBLASE OBO auth is configured (EMBLASE_ENTRA_* vars are set) but "
        "tiled.server.authentication.get_session_state could not be imported. "
        "Ensure the emblase router is loaded inside a running Tiled server."
    )


# ---------------------------------------------------------------------------
# OBO helpers
# ---------------------------------------------------------------------------


async def _exchange_obo(user_access_token: str) -> str:
    """Exchange a user's Entra access token for one scoped to the chatapp via OBO.

    Results are cached by SHA-256(user_access_token) with a 45-minute TTL so
    that the Entra token endpoint is not hammered on every request.

    Raises HTTPException(401) if the exchange fails.
    """
    cache_key = hashlib.sha256(user_access_token.encode()).hexdigest()
    with _obo_cache_lock:
        cached = _obo_cache.get(cache_key)
    if cached:
        return cached

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            _TOKEN_URL,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "client_id": _CLIENT_ID,
                "client_secret": _CLIENT_SECRET,
                "assertion": user_access_token,
                "scope": f"openid offline_access {_CHATAPP_SCOPE}",
                "requested_token_use": "on_behalf_of",
            },
        )
    if not resp.is_success:
        log.warning("OBO exchange failed %s: %s", resp.status_code, resp.text)
        raise HTTPException(
            status_code=401,
            detail="OBO token exchange failed — log out and back in to refresh your session.",
        )

    obo_token = resp.json()["access_token"]
    with _obo_cache_lock:
        _obo_cache[cache_key] = obo_token
    return obo_token


async def _get_chat_auth_header(session_state: Optional[dict]) -> dict:
    """Return an Authorization header for the chatapp.

    Priority:
    1. OBO exchange using the user's stored Entra access token (production).
    2. Static EMBLASE_CHATAPP_TOKEN fallback (local dev).
    3. Raise 401 if neither is available.
    """
    if _OBO_ENABLED and session_state:
        entra_token = session_state.get("entra_access_token")
        if entra_token:
            chat_token = await _exchange_obo(entra_token)
            return {"Authorization": f"Bearer {chat_token}"}
        else:
            log.warning(
                "OBO configured but no entra_access_token in session state — "
                "user may have logged in before OBO support was deployed. "
                "Falling back to static token if available."
            )

    if _CHAT_TOKEN:
        return {"Authorization": f"Bearer {_CHAT_TOKEN}"}

    raise HTTPException(
        status_code=401,
        detail=(
            "No chat credentials available. "
            "Log out and back in so the server can capture your Entra token, "
            "or set EMBLASE_CHATAPP_TOKEN for local development."
        ),
    )


# ---------------------------------------------------------------------------
# Priming message helpers
# ---------------------------------------------------------------------------


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
    """Fetch metadata for a Tiled node using the current request's auth token.

    Returns an empty dict on failure and logs a warning — priming will still
    proceed but with unknown metadata fields.
    """
    auth = request.headers.get("Authorization", "")
    tiled_base = str(request.base_url).rstrip("/")
    url = f"{tiled_base}/api/v1/metadata/{node_path}"
    async with httpx.AsyncClient(timeout=10) as client:
        res = await client.get(url, headers={"Authorization": auth})
    if res.status_code == 200:
        return res.json().get("data", {}).get("attributes", {})
    log.warning(
        "_fetch_tiled_metadata: got %s for path %r — priming will use empty metadata",
        res.status_code,
        node_path,
    )
    return {}


async def _send_priming_message(
    session_id: Optional[str],
    priming_text: str,
    auth_header: dict,
) -> str:
    """Send the hidden context message and return the resulting session_id.

    Raises HTTPException(502) if the upstream chat service rejects the priming
    message, so that the caller does not silently start a context-free session.
    """
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
            headers={**auth_header, **_JSON_CT},
        )
    if res.status_code != 200:
        log.error(
            "_send_priming_message: upstream returned %s — %s",
            res.status_code,
            res.text,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Chat service rejected priming message ({res.status_code}). Cannot start session.",
        )

    data = res.json()
    new_session_id = data.get("chat_session_id") or session_id
    # Record the assistant reply's message_id so we can filter both the
    # priming user turn and the assistant ack from history.
    msg_id = data.get("message_id")
    if new_session_id and msg_id is not None:
        _priming_message_ids[new_session_id] = msg_id  # assistant reply
    return new_session_id


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    node_path: str
    chat_session_id: Optional[str] = None


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    session_state: Optional[dict] = _session_state_dep,
):
    """Proxy a chat message to the AmSC service and stream the SSE response back.

    On a new session (no chat_session_id), a hidden priming message describing
    the dataset is sent first so the LLM has context. The priming message is
    never shown to the user.

    The AmSC token never leaves the server — the browser only needs a valid
    Tiled session to reach this endpoint.
    """
    auth_header = await _get_chat_auth_header(session_state)
    session_id = req.chat_session_id

    # New session: fetch dataset metadata and send a silent priming message.
    if not session_id:
        metadata = await _fetch_tiled_metadata(request, req.node_path)
        priming_text = _build_priming_message(req.node_path, metadata)
        session_id = await _send_priming_message(None, priming_text, auth_header)

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
                headers={**auth_header, **_JSON_CT, "Accept": "text/event-stream"},
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/chat")
async def chat(
    req: ChatRequest,
    request: Request,
    session_state: Optional[dict] = _session_state_dep,
) -> dict:
    """Non-streaming proxy: send a message and return the full assistant reply.

    Useful for scripting / debugging without SSE support.
    """
    auth_header = await _get_chat_auth_header(session_state)
    session_id = req.chat_session_id

    if not session_id:
        metadata = await _fetch_tiled_metadata(request, req.node_path)
        priming_text = _build_priming_message(req.node_path, metadata)
        session_id = await _send_priming_message(None, priming_text, auth_header)

    payload = {
        "message": req.message,
        "model_name": _CHAT_MODEL,
        "chat_session_id": session_id,
        "image_refs": [],
    }
    async with httpx.AsyncClient(timeout=120) as client:
        res = await client.post(
            f"{_CHAT_URL}/chat",
            json=payload,
            headers={**auth_header, **_JSON_CT},
        )
    res.raise_for_status()
    return res.json()


@router.get("/chat/history/{session_id}")
async def chat_history(
    session_id: str,
    session_state: Optional[dict] = _session_state_dep,
):
    """Fetch message history for a session from AmSC, stripping the priming messages."""
    auth_header = await _get_chat_auth_header(session_state)
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.get(
            f"{_CHAT_URL}/sessions/{session_id}/messages",
            headers={**auth_header, **_JSON_CT},
        )
    if res.status_code != 200:
        return []

    messages = res.json()

    # Filter out the hidden priming exchange (user context msg + assistant ack).
    # Primary filter: by stored message_id (works when server hasn't restarted).
    # Fallback filter: by content marker (works across restarts / workers).
    priming_id = _priming_message_ids.get(session_id)
    if priming_id is not None:
        hidden_ids = {priming_id, priming_id - 1}
        messages = [m for m in messages if m.get("message_id") not in hidden_ids]
    else:
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


# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------


@router.get("/main.js")
async def main_js():
    return FileResponse(
        STATIC_DIR / "main.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )
