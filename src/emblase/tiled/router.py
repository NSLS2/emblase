"""FastAPI router that serves the embedding UI plugin's static assets and
proxies chat requests to the AmSC chat service.

Registered via Tiled's ``routers:`` config and served at ``/custom/emblase/``.

Authentication design — On-Behalf-Of (OBO)
-------------------------------------------
When the browser authenticates to Tiled via Entra OIDC, Tiled's
``EntraAuthenticator`` stores the original Entra ``access_token`` and
``refresh_token`` inside the session's ``state`` dict.  This state is:

  1. Persisted in the Tiled session DB (``sessions.state`` JSON column).
  2. Embedded verbatim in the Tiled HMAC-HS256 access token sent to the
     browser — **the Entra access token is therefore visible to anyone who
     base64-decodes the Tiled JWT**.  This is acceptable because the Tiled
     access token is short-lived (default 15 minutes) and is only transmitted
     over HTTPS, but it is a conscious security trade-off that operators
     should be aware of.  The Entra access token itself also expires in ~1 h.

This router retrieves those tokens via ``get_session_state`` (which decodes
the Tiled JWT without a DB hit) and performs a Microsoft OBO exchange to
obtain a per-user Entra token scoped to the AmSC chat application (audience
``api://<chatapp_client_id>``).  That derived token is forwarded to AmSC in
``Authorization: Bearer`` — AmSC sees the user's real Entra identity via its
``bearer_obo`` auth path.

OBO token caching
~~~~~~~~~~~~~~~~~
OBO tokens are valid for ~1 hour.  To avoid hitting the Entra token endpoint
on every request (and risk rate-limiting), results are cached in a
process-local ``TTLCache`` keyed on ``SHA-256(entra_access_token)`` with a
45-minute TTL.  The user's raw Entra access token is never stored in the
cache — only its hash is used as the key.

Automatic token refresh
~~~~~~~~~~~~~~~~~~~~~~~
Entra access tokens expire in ~1 hour.  When the OBO exchange fails with
``invalid_grant``, the router silently:

  1. Calls the Entra token endpoint with ``grant_type=refresh_token`` to
     obtain a fresh access + refresh token pair.
  2. Writes the new tokens back to the Tiled session DB so that the next
     Tiled ``slide_session`` call picks them up and embeds them in the
     renewed Tiled HMAC JWT — keeping everything in sync without any
     browser-side action.
  3. Retries the OBO exchange with the fresh access token.

The user never needs to re-authenticate for the lifetime of the Tiled
session (controlled by Tiled's ``session_max_age`` setting).

DB write design
~~~~~~~~~~~~~~~
``_update_session_state`` receives a live ``db`` session via FastAPI
dependency injection (``get_database_session_factory``), the same mechanism
Tiled uses internally.  The function looks up the session by *principal UUID*
(the ``sub`` claim of the Tiled access token, available on every request),
not by session UUID — the ``sub`` claim (principal UUID) is present in access
tokens, whereas ``sid`` (session UUID) only appears in refresh tokens.
Because a principal may have multiple concurrent sessions, the state update is
applied to **all** valid non-revoked sessions for that principal.  In practice

Priming context design
~~~~~~~~~~~~~~~~~~~~~~
On the first message of a new session (no ``chat_session_id``), dataset
context is injected via the ``developer_prompt`` field of the AmSC request.
This keeps the context out of the visible message history and avoids a
separate hidden POST that could trigger AmSC's tool-calling layer on the
node path string.  The server-side ``developer_prompt`` (built from Tiled
metadata) is merged with any ``developer_prompt`` already sent by the browser
(which may include selection context from the UI).
a user has at most one active browser session, so this is equivalent to a
targeted update.

Fallback (local dev without Entra)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If any of the four OBO env vars are absent, the router falls back to a static
bearer token from ``EMBLASE_CHATAPP_TOKEN``.  A warning is emitted at startup.
This allows local development against a Tiled instance that is not wired to
Entra.

Required environment variables
--------------------------------
  EMBLASE_CHATAPP_URL          Base URL of the AmSC chat service
                               (default: https://chat-amsc-dev.nsls2.bnl.gov)
  EMBLASE_CHATAPP_MODEL        Default model name
                               (default: openai/gpt-oss-20b)
  EMBLASE_ENTRA_TENANT_ID      Azure AD tenant ID
  EMBLASE_ENTRA_CLIENT_ID      Tiled's own Entra app registration client ID
  EMBLASE_ENTRA_CLIENT_SECRET  Tiled's Entra client secret
  EMBLASE_CHATAPP_SCOPE        Scope for the AmSC app
                               e.g. api://<chatapp_client_id>/access_as_user
  EMBLASE_CHATAPP_TOKEN        Fallback static bearer token (local dev only)
"""

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Callable, Optional

import httpx
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Load .env from the working directory (or any parent) so EMBLASE_* vars are
# available even when the server is started without explicitly sourcing .env.
load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/emblase", tags=["emblase"])

STATIC_DIR = Path(__file__).resolve().parent / "static"

_CHAT_URL = os.environ.get("EMBLASE_CHATAPP_URL", "https://chat-amsc-dev.nsls2.bnl.gov").rstrip("/")
_CHAT_MODEL = os.environ.get("EMBLASE_CHATAPP_MODEL", "openai/gpt-oss-20b")

# OBO configuration — all four must be set to enable per-user auth.
_TENANT_ID = os.environ.get("EMBLASE_ENTRA_TENANT_ID", "")
_CLIENT_ID = os.environ.get("EMBLASE_ENTRA_CLIENT_ID", "")
_CLIENT_SECRET = os.environ.get("EMBLASE_ENTRA_CLIENT_SECRET", "")
_CHATAPP_SCOPE = os.environ.get("EMBLASE_CHATAPP_SCOPE", "")

# Fallback static token for local dev (no Entra configured).
_FALLBACK_TOKEN = os.environ.get("EMBLASE_CHATAPP_TOKEN", "")

_OBO_ENABLED = bool(_TENANT_ID and _CLIENT_ID and _CLIENT_SECRET and _CHATAPP_SCOPE)

if not _OBO_ENABLED:
    import warnings

    warnings.warn(
        "EMBLASE_ENTRA_TENANT_ID / EMBLASE_ENTRA_CLIENT_ID / "
        "EMBLASE_ENTRA_CLIENT_SECRET / EMBLASE_CHATAPP_SCOPE are not all set. "
        "Chat proxy will fall back to EMBLASE_CHATAPP_TOKEN (service account). "
        "Set all four variables to enable true per-user OBO authentication.",
        stacklevel=1,
    )
    if not _FALLBACK_TOKEN:
        warnings.warn(
            "EMBLASE_CHATAPP_TOKEN is also not set — chat endpoints will fail.",
            stacklevel=1,
        )

_TOKEN_ENDPOINT = (
    f"https://login.microsoftonline.com/{_TENANT_ID}/oauth2/v2.0/token" if _TENANT_ID else ""
)

# Common JSON content-type header added to every AmSC request.
_JSON_CT = {"Content-Type": "application/json"}

# ---------------------------------------------------------------------------
# OBO token cache
# TTL = 45 min (OBO tokens are valid for ~1 h; 15 min margin before expiry).
# Key = SHA-256(entra_access_token) — the raw token is never stored.
# maxsize = 1024 covers ~1000 concurrent users with negligible memory cost.
# ---------------------------------------------------------------------------
_obo_cache: TTLCache = TTLCache(maxsize=1024, ttl=45 * 60)
_obo_cache_lock = threading.Lock()




# ---------------------------------------------------------------------------
# OBO helpers
# ---------------------------------------------------------------------------


class _OBOError(Exception):
    """Raised when the OBO token exchange fails."""

    def __init__(self, error_code: str, body: dict):
        self.error_code = error_code
        self.body = body
        super().__init__(error_code)


async def _exchange_obo(entra_access_token: str) -> str:
    """Exchange an Entra access token for a chat-app-scoped token via OBO.

    Results are cached by SHA-256(entra_access_token) with a 45-minute TTL so
    that the Entra token endpoint is not hammered on every request.

    Raises ``_OBOError`` on failure so the caller can inspect the error code
    and decide whether to retry after refreshing the underlying token.
    """
    cache_key = hashlib.sha256(entra_access_token.encode()).hexdigest()
    with _obo_cache_lock:
        cached = _obo_cache.get(cache_key)
    if cached:
        return cached

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            _TOKEN_ENDPOINT,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "client_id": _CLIENT_ID,
                "client_secret": _CLIENT_SECRET,
                "assertion": entra_access_token,
                "requested_token_use": "on_behalf_of",
                "scope": _CHATAPP_SCOPE,
            },
        )
    body = resp.json()
    if resp.is_error:
        raise _OBOError(body.get("error", "unknown"), body)

    obo_token = body["access_token"]
    with _obo_cache_lock:
        _obo_cache[cache_key] = obo_token
    return obo_token


async def _refresh_entra_tokens(refresh_token: str) -> dict:
    """Use an Entra refresh token to obtain a fresh access + refresh token pair.

    Requests ``openid offline_access`` alongside the chat app scope so that
    the response always contains a new refresh token for the next cycle.
    Raises ``HTTPException(401)`` on failure so the error surfaces cleanly to
    the browser.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            _TOKEN_ENDPOINT,
            data={
                "grant_type": "refresh_token",
                "client_id": _CLIENT_ID,
                "client_secret": _CLIENT_SECRET,
                "refresh_token": refresh_token,
                # openid + offline_access ensure we get a fresh refresh_token
                # back; the chat app scope makes the new access_token usable
                # for OBO without a second round-trip.
                "scope": f"openid offline_access {_CHATAPP_SCOPE}",
            },
        )
    body = resp.json()
    if resp.is_error:
        logger.error("Entra refresh_token grant failed: %s", body)
        raise HTTPException(
            status_code=401,
            detail="Your session has expired and could not be renewed automatically. Please log in again.",
        )
    return body


async def _update_session_state(
    principal_uuid_hex: str,
    new_tokens: dict,
    db_factory: Callable,
) -> None:
    """Persist refreshed Entra tokens back to all Tiled sessions for this principal.

    Uses the *principal UUID* (the ``sub`` claim of the Tiled access token,
    available on every request) rather than the session UUID (``sid``), which
    only appears in refresh tokens and is therefore not available here.

    A principal may have multiple concurrent sessions (e.g. two browser tabs).
    We update all of them so they all benefit from the fresh tokens.  In the
    common single-session case this is exactly one row.

    The update is best-effort: a failure here is non-fatal.  The OBO exchange
    already succeeded and the response will be returned to the user.  The only
    consequence of a failed write is that the next OBO attempt (after the new
    access token also expires) will need to refresh again.
    """
    try:
        import uuid as _uuid

        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        from tiled.authn_database import orm

        async with db_factory() as db:
            principal = (
                await db.execute(
                    select(orm.Principal)
                    .options(selectinload(orm.Principal.sessions))
                    .filter(orm.Principal.uuid == _uuid.UUID(hex=principal_uuid_hex))
                )
            ).scalar()

            if principal is None:
                logger.warning(
                    "_update_session_state: principal %s not found in DB",
                    principal_uuid_hex,
                )
                return

            updated = 0
            for session in principal.sessions:
                if session.revoked:
                    continue
                merged = dict(session.state or {})
                merged.update(new_tokens)
                # Assign a new dict object so SQLAlchemy detects the mutation
                # on the JSON column.
                session.state = merged
                db.add(session)
                updated += 1

            if updated:
                await db.commit()
                logger.debug(
                    "Updated Entra tokens in %d Tiled session(s) for principal %s",
                    updated,
                    principal_uuid_hex,
                )
    except Exception:
        logger.warning(
            "Failed to write refreshed Entra tokens back to Tiled session DB "
            "(principal=%s). The current request will succeed, but the stored "
            "tokens are stale and will need refreshing again on the next OBO attempt.",
            principal_uuid_hex,
            exc_info=True,
        )


async def _get_chat_auth_header(
    session_state: Optional[dict],
    principal_uuid_hex: Optional[str],
    db_factory: Optional[Callable],
) -> dict:
    """Return an ``Authorization`` + ``Content-Type`` header dict for AmSC.

    Performs the full OBO flow when configured:
      1. Try OBO with the stored Entra access token (cache hit → no network).
      2. On ``invalid_grant``, refresh the Entra tokens and retry OBO once.
      3. Write fresh tokens back to the Tiled DB after a successful refresh.

    Falls back to the static service-account token when OBO is not configured
    or when session state is unavailable (e.g. API-key auth in local dev).
    """
    if _OBO_ENABLED and session_state:
        entra_access_token = session_state.get("entra_access_token")
        entra_refresh_token = session_state.get("entra_refresh_token")

        if entra_access_token:
            try:
                chat_token = await _exchange_obo(entra_access_token)
                return {"Authorization": f"Bearer {chat_token}", **_JSON_CT}
            except _OBOError as exc:
                if exc.error_code == "invalid_grant" and entra_refresh_token:
                    logger.info(
                        "OBO invalid_grant for principal=%s — refreshing Entra tokens",
                        principal_uuid_hex,
                    )
                    refreshed = await _refresh_entra_tokens(entra_refresh_token)
                    new_access = refreshed["access_token"]
                    new_refresh = refreshed.get("refresh_token", entra_refresh_token)
                    new_tokens = {
                        "entra_access_token": new_access,
                        "entra_refresh_token": new_refresh,
                    }

                    # Write fresh tokens to the DB (best-effort, non-blocking).
                    if principal_uuid_hex and db_factory:
                        await _update_session_state(principal_uuid_hex, new_tokens, db_factory)

                    # Retry OBO with the fresh access token.
                    try:
                        chat_token = await _exchange_obo(new_access)
                    except _OBOError as retry_exc:
                        logger.error(
                            "OBO exchange failed after token refresh for principal=%s: %s",
                            principal_uuid_hex,
                            retry_exc.body,
                        )
                        raise HTTPException(
                            status_code=401,
                            detail="Could not obtain a chat service token after refreshing your session. Please log in again.",
                        )
                    return {"Authorization": f"Bearer {chat_token}", **_JSON_CT}
                else:
                    logger.error(
                        "OBO exchange failed permanently for principal=%s: %s",
                        principal_uuid_hex,
                        exc.body,
                    )
                    raise HTTPException(
                        status_code=401,
                        detail="Could not obtain a chat service token for your account.",
                    )
        else:
            logger.warning(
                "OBO is enabled but session state has no entra_access_token "
                "(principal=%s). Falling back to service-account token.",
                principal_uuid_hex,
            )

    # Fallback path — static service-account PAT.
    if not _FALLBACK_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Chat service is not configured (no authentication token available).",
        )
    return {"Authorization": f"Bearer {_FALLBACK_TOKEN}", **_JSON_CT}


# ---------------------------------------------------------------------------
# Principal / session helpers
# ---------------------------------------------------------------------------


def _get_username(principal) -> Optional[str]:
    """Extract the username string from a Tiled Principal, or None."""
    if principal is None:
        return None
    identities = getattr(principal, "identities", [])
    if identities:
        return str(identities[0].id)
    return None


def _require_principal(principal) -> None:
    """Raise HTTP 401 if no authenticated principal (anonymous request)."""
    if principal is None:
        raise HTTPException(status_code=401, detail="Authentication required.")


def _get_principal_dependency():
    """Return the get_current_principal FastAPI dependency, or None for local dev."""
    try:
        from tiled.server.authentication import get_current_principal

        return Depends(get_current_principal)
    except ImportError:
        return None


def _get_session_state_dependency():
    """Return the get_session_state FastAPI dependency, or None for local dev."""
    try:
        from tiled.server.authentication import get_session_state

        return Depends(get_session_state)
    except ImportError:
        return None


def _get_decoded_token_dependency():
    """Return the get_decoded_access_token FastAPI dependency, or None for local dev."""
    try:
        from tiled.server.authentication import get_decoded_access_token

        return Depends(get_decoded_access_token)
    except ImportError:
        return None


def _get_db_factory_dependency():
    """Return the get_database_session_factory FastAPI dependency, or None for local dev."""
    try:
        from tiled.server.connection_pool import get_database_session_factory

        return Depends(get_database_session_factory)
    except ImportError:
        return None


# Resolve dependencies once at import time.
_principal_dep = _get_principal_dependency()
_session_state_dep = _get_session_state_dependency()
_decoded_token_dep = _get_decoded_token_dependency()
_db_factory_dep = _get_db_factory_dependency()

# If OBO is configured but session state injection is unavailable, raise at
# startup so the misconfiguration is immediately visible rather than silently
# degrading to the fallback token.
if _OBO_ENABLED and _session_state_dep is None:
    raise RuntimeError(
        "EMBLASE OBO auth is configured (EMBLASE_ENTRA_* vars are set) but "
        "tiled.server.authentication.get_session_state could not be imported. "
        "Ensure the emblase router is loaded inside a running Tiled server."
    )


# ---------------------------------------------------------------------------
# Priming message helpers
# ---------------------------------------------------------------------------


def _build_developer_prompt(node_path: str, metadata: dict) -> str:
    """Build a developer_prompt describing the dataset context for the LLM.

    This is injected on the first message of a new session via AmSC's
    ``developer_prompt`` field so the context is invisible in chat history
    and does not trigger AmSC's tool-calling layer.
    """
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
        "The user is currently viewing a LatentSpaceEmbedding container in the Emblase Latent Space Explorer.",
        f"The node for this container is: `{node_path}`",
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
        f"You can query the contents of this container using your tools (node: `{node_path}`).",
        "",
        "Important: keep answers concise, up to 3 sentences. Expect follow-up questions.",
        "Avoid large headers and excessive structure — short paragraphs or brief bullet points are preferred.",
    ]
    return "\n".join(lines)


async def _fetch_tiled_metadata(request: Request, node_path: str) -> dict:
    """Fetch metadata for a Tiled node, reusing the browser's auth token.

    Returns an empty dict on failure and logs a warning — context will still
    be injected but with unknown metadata fields.
    """
    auth = request.headers.get("Authorization", "")
    tiled_base = str(request.base_url).rstrip("/")
    url = f"{tiled_base}/api/v1/metadata/{node_path}"
    async with httpx.AsyncClient(timeout=10) as client:
        res = await client.get(url, headers={"Authorization": auth})
    if res.status_code == 200:
        return res.json().get("data", {}).get("attributes", {})
    logger.warning(
        "_fetch_tiled_metadata: got %s for path %r — developer_prompt will use empty metadata",
        res.status_code,
        node_path,
    )
    return {}


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    node_path: str
    chat_session_id: Optional[str] = None
    developer_prompt: Optional[str] = None


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    principal=_principal_dep,
    session_state=_session_state_dep,
    decoded_token=_decoded_token_dep,
    db_factory=_db_factory_dep,
):
    """Proxy a chat message to AmSC and stream the SSE response back.

    Performs a per-user Entra OBO exchange (when configured) so AmSC receives
    the user's real Entra identity rather than a service-account token.

    On a new session (no ``chat_session_id``), dataset context is injected via
    the ``developer_prompt`` field of the upstream request.  The server builds
    a ``developer_prompt`` from Tiled metadata and merges it with any
    ``developer_prompt`` already supplied by the browser (which may include
    selection context from the UI).
    """
    _require_principal(principal)
    username = _get_username(principal)
    # The ``sub`` claim in the Tiled access token is the principal UUID hex —
    # not the session UUID (``sid``), which only lives in refresh tokens.
    principal_uuid_hex = decoded_token.get("sub") if decoded_token else None

    auth_header = await _get_chat_auth_header(session_state, principal_uuid_hex, db_factory)
    session_id = req.chat_session_id

    developer_prompt: Optional[str] = None
    if not session_id:
        metadata = await _fetch_tiled_metadata(request, req.node_path)
        server_prompt = _build_developer_prompt(req.node_path, metadata)
        # Merge with any client-provided developer_prompt (e.g. selection context).
        if req.developer_prompt:
            developer_prompt = server_prompt + "\n\n" + req.developer_prompt
        else:
            developer_prompt = server_prompt

    payload: dict = {
        "message": req.message,
        "model_name": _CHAT_MODEL,
        "chat_session_id": session_id,
        "image_refs": [],
    }
    if developer_prompt is not None:
        payload["developer_prompt"] = developer_prompt
    if username:
        payload["username"] = username

    async def stream():
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{_CHAT_URL}/chat/stream",
                json=payload,
                headers={**auth_header, "Accept": "text/event-stream"},
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    logger.error(
                        "chat/stream upstream returned %s — %s",
                        response.status_code,
                        body[:500],
                    )
                    yield (
                        f'data: {{"type": "error", "message": "Upstream chat service error ({response.status_code})"}}\n\n'
                        .encode()
                    )
                    return
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat")
async def chat(
    req: ChatRequest,
    request: Request,
    principal=_principal_dep,
    session_state=_session_state_dep,
    decoded_token=_decoded_token_dep,
    db_factory=_db_factory_dep,
) -> dict:
    """Non-streaming proxy: send a message and return the full assistant reply.

    Useful for scripting and debugging without SSE support.
    See ``POST /emblase/chat/stream`` for the full behaviour description.
    """
    _require_principal(principal)
    username = _get_username(principal)
    principal_uuid_hex = decoded_token.get("sub") if decoded_token else None

    auth_header = await _get_chat_auth_header(session_state, principal_uuid_hex, db_factory)
    session_id = req.chat_session_id

    developer_prompt: Optional[str] = None
    if not session_id:
        metadata = await _fetch_tiled_metadata(request, req.node_path)
        server_prompt = _build_developer_prompt(req.node_path, metadata)
        if req.developer_prompt:
            developer_prompt = server_prompt + "\n\n" + req.developer_prompt
        else:
            developer_prompt = server_prompt

    payload: dict = {
        "message": req.message,
        "model_name": _CHAT_MODEL,
        "chat_session_id": session_id,
        "image_refs": [],
    }
    if developer_prompt is not None:
        payload["developer_prompt"] = developer_prompt
    if username:
        payload["username"] = username

    async with httpx.AsyncClient(timeout=120) as client:
        res = await client.post(
            f"{_CHAT_URL}/chat",
            json=payload,
            headers=auth_header,
        )
    res.raise_for_status()
    return res.json()


@router.get("/chat/history/{session_id}")
async def chat_history(
    session_id: str,
    principal=_principal_dep,
    session_state=_session_state_dep,
    decoded_token=_decoded_token_dep,
    db_factory=_db_factory_dep,
):
    """Fetch message history for a session from AmSC.

    Returns the message list from AmSC directly.  Since context is now
    injected via ``developer_prompt`` (invisible in history) rather than a
    hidden priming message, no filtering is required.
    """
    _require_principal(principal)
    principal_uuid_hex = decoded_token.get("sub") if decoded_token else None
    auth_header = await _get_chat_auth_header(session_state, principal_uuid_hex, db_factory)

    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.get(
            f"{_CHAT_URL}/sessions/{session_id}/messages",
            headers=auth_header,
        )
    if res.status_code != 200:
        return []

    return res.json()


# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------


@router.get("/main.js")
async def main_js():
    """Serve the compiled Emblase UI plugin bundle."""
    return FileResponse(
        STATIC_DIR / "main.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )
