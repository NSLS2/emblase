"""FastAPI router that serves the embedding UI plugin's static assets and chat config.

Registered via Tiled's ``routers:`` config and served at ``/custom/emblase/``.

Chat credentials are read from environment variables so they never appear in
version-controlled config files:

  EMBLASE_CHATAPP_URL    Base URL of the AmSC chat service
                         (default: https://chat-amsc-dev.nsls2.bnl.gov)
  EMBLASE_CHATAPP_TOKEN  Bearer token for the chat service
  EMBLASE_CHATAPP_MODEL  Model name to request
                         (default: openai/gpt-oss-120b)
"""

import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(prefix="/emblase", tags=["emblase"])

STATIC_DIR = Path(__file__).resolve().parent / "static"

_CHAT_URL = os.environ.get("EMBLASE_CHATAPP_URL", "https://chat-amsc-dev.nsls2.bnl.gov").rstrip("/")
_CHAT_TOKEN = os.environ.get("EMBLASE_CHATAPP_TOKEN", "")
_CHAT_MODEL = os.environ.get("EMBLASE_CHATAPP_MODEL", "openai/gpt-oss-120b")


@router.get("/config")
async def get_config() -> dict:
    """Return chat configuration for the plugin.

    The token is served here (over the same authenticated Tiled session) rather
    than embedded in ui_settings.yml so it stays out of version control.
    """
    return {
        "chat_url": _CHAT_URL,
        "chat_token": _CHAT_TOKEN,
        "chat_model": _CHAT_MODEL,
    }


@router.get("/main.js")
async def main_js():
    return FileResponse(
        STATIC_DIR / "main.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )
