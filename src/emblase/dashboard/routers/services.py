"""External service status routers for the EMBLASE dashboard.

Covers:
- MLflow model registry (status + model list)
- AmSC LLM chat app (status)
- OpenMetadata catalog (placeholder)
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter

from ...config import settings

router = APIRouter(tags=["services"])


# ── MLflow ────────────────────────────────────────────────────────────────────


@router.get("/mlflow/status")
async def mlflow_status() -> dict[str, Any]:
    """Return MLflow server reachability and basic info."""
    if not settings.mlflow_tracking_uri:
        return {"status": "unconfigured", "message": "EMBLASE_MLFLOW_TRACKING_URI not set"}
    try:
        uri = settings.mlflow_tracking_uri.rstrip("/")
        headers: dict[str, str] = {}
        if settings.mlflow_api_key:
            headers["X-Api-Key"] = settings.mlflow_api_key

        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            resp = await client.get(f"{uri}/api/2.0/mlflow/experiments/list")
            resp.raise_for_status()

        return {
            "status": "online",
            "tracking_uri": settings.mlflow_tracking_uri,
            "experiment": settings.mlflow_experiment,
            "model_prefix": settings.mlflow_model_prefix,
        }
    except Exception as exc:
        # Some servers don't expose /experiments/list — still report what we know
        return {
            "status": "error",
            "error": str(exc),
            "tracking_uri": settings.mlflow_tracking_uri,
        }


@router.get("/mlflow/models")
async def mlflow_models() -> dict[str, Any]:
    """List registered models with the configured prefix (e.g. 'bnl-nsls2-')."""
    if not settings.mlflow_tracking_uri:
        return {"models": [], "error": "EMBLASE_MLFLOW_TRACKING_URI not set"}
    try:
        import asyncio

        from ...mlflow_registry import list_models

        loop = asyncio.get_event_loop()
        model_list = await loop.run_in_executor(None, list_models)
        prefix = settings.mlflow_model_prefix
        filtered = [
            {
                "name": m.name,
                "latest_version": m.latest_version,
                "description": m.description,
            }
            for m in model_list
            if not prefix or m.name.startswith(prefix)
        ]
        return {"models": filtered, "total": len(filtered), "prefix_filter": prefix}
    except Exception as exc:
        return {"models": [], "error": str(exc)}


# ── Chatbot ───────────────────────────────────────────────────────────────────


@router.get("/chatbot/status")
async def chatbot_status() -> dict[str, Any]:
    """Return the status of the AmSC LLM chat service."""
    url = settings.chatapp_url
    if not url:
        return {"status": "unconfigured", "message": "EMBLASE_CHATAPP_URL not set"}
    try:
        headers: dict[str, str] = {}
        if settings.chatapp_token:
            headers["Authorization"] = f"Bearer {settings.chatapp_token}"
        async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
            resp = await client.get(url.rstrip("/") + "/")
            # Accept any 2xx or 3xx; the service is reachable regardless of body
            reachable = resp.status_code < 500
        return {
            "status": "online" if reachable else "degraded",
            "url": url,
            "model": settings.chatapp_model,
            "http_status": resp.status_code,
        }
    except httpx.TimeoutException:
        return {"status": "timeout", "url": url}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "url": url}


# ── OpenMetadata (placeholder) ────────────────────────────────────────────────


@router.get("/openmetadata/status")
async def openmetadata_status() -> dict[str, Any]:
    """Placeholder for the OpenMetadata catalog status (not yet implemented)."""
    return {
        "status": "placeholder",
        "message": (
            "OpenMetadata catalog integration is not yet configured. "
            "This service will receive metadata updates from Tiled via webhooks."
        ),
        "planned_features": [
            "Automatic dataset registration from Tiled write events",
            "Lineage tracking: input runs → embeddings → model versions",
            "Searchable catalog of all EMBLASE experiments",
            "Integration with AmSC data governance policies",
        ],
    }
