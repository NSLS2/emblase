"""External service status routers for the EMBLASE dashboard.

Covers:
- MLflow model registry (status + model list)
- AmSC LLM chat app (status)
- AmSC OpenMetadata catalog (status + recent artifacts)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import APIRouter

from ...config import settings

router = APIRouter(tags=["services"])


# ── MLflow helpers ────────────────────────────────────────────────────────────


@asynccontextmanager
async def _mlflow_client(timeout: float = 10.0):
    """Yield a configured AsyncClient for the MLflow REST API."""
    headers: dict[str, str] = {}
    if settings.mlflow_api_key:
        headers["X-Api-Key"] = settings.mlflow_api_key
    async with httpx.AsyncClient(
        base_url=settings.mlflow_tracking_uri.rstrip("/"),
        headers=headers,
        timeout=timeout,
    ) as client:
        yield client


# ── MLflow endpoints ──────────────────────────────────────────────────────────


@router.get("/mlflow/status")
async def mlflow_status() -> dict[str, Any]:
    """Return MLflow server reachability and basic info.

    Uses registered-models/search (a cheap, single-result probe) rather than
    experiments/list which returns 404 on some MLflow-compatible deployments.

    The MLflow REST API is called directly via httpx — no mlflow SDK needed.
    This keeps the dashboard environment lightweight and avoids version-pinning
    the heavy mlflow package just for a status check.
    """
    if not settings.mlflow_tracking_uri:
        return {"status": "unconfigured", "message": "EMBLASE_MLFLOW_TRACKING_URI not set"}
    try:
        async with _mlflow_client() as client:
            resp = await client.get(
                "/api/2.0/mlflow/registered-models/search",
                params={"max_results": 1},
            )
            resp.raise_for_status()
        return {
            "status": "online",
            "tracking_uri": settings.mlflow_tracking_uri,
            "experiment": settings.mlflow_experiment,
            "model_prefix": settings.mlflow_model_prefix,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "tracking_uri": settings.mlflow_tracking_uri,
        }


@router.get("/mlflow/models")
async def mlflow_models() -> dict[str, Any]:
    """List registered models with the configured prefix via MLflow REST API.

    Paginates through registered-models/search until exhausted.  A client-side
    prefix filter is applied as a fallback for servers that ignore the filter param.
    """
    if not settings.mlflow_tracking_uri:
        return {"models": [], "error": "EMBLASE_MLFLOW_TRACKING_URI not set"}
    try:
        prefix = settings.mlflow_model_prefix  # e.g. "bnl-nsls2-"
        models: list[dict[str, Any]] = []
        page_token: str | None = None

        async with _mlflow_client(timeout=15.0) as client:
            while True:
                params: dict[str, Any] = {"max_results": 200}
                if prefix:
                    params["filter"] = f"name LIKE '{prefix}%'"
                if page_token:
                    params["page_token"] = page_token

                resp = await client.get(
                    "/api/2.0/mlflow/registered-models/search",
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()

                for m in data.get("registered_models", []):
                    name: str = m.get("name", "")
                    # Client-side fallback filter (handles servers that ignore the filter param)
                    if prefix and not name.startswith(prefix):
                        continue
                    latest_versions = m.get("latest_versions", [])
                    latest = (
                        max(int(v["version"]) for v in latest_versions) if latest_versions else None
                    )
                    models.append(
                        {
                            "name": name,
                            "latest_version": latest,
                            "description": m.get("description") or None,
                        }
                    )

                page_token = data.get("next_page_token")
                if not page_token:
                    break

        models.sort(key=lambda m: m["name"])
        return {"models": models, "total": len(models), "prefix_filter": prefix}
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


# ── AmSC OpenMetadata catalog ─────────────────────────────────────────────────


@router.get("/openmetadata/status")
async def openmetadata_status() -> dict[str, Any]:
    """Check connectivity to the AmSC OpenMetadata catalog API."""
    import logging as _logging

    _log = _logging.getLogger(__name__)

    raw_catalog = settings.amsc_openmetadata_catalog_name or ""
    # "bnl-lse-demo-storage.bnl-lse-demo-data-catalog"
    #  └─ catalog_name (POST path segment) ──┘  └─ full root FQN ──────────────┘
    catalog_name = raw_catalog.split(".")[0] if raw_catalog else None
    root_fqn = raw_catalog if raw_catalog else (settings.amsc_openmetadata_parent_fqn or None)

    if not settings.amsc_openmetadata_token:
        return {
            "status": "unconfigured",
            "message": "EMBLASE_AMSC_OPENMETADATA_TOKEN is not set",
            "catalog_url": settings.amsc_openmetadata_catalog_url,
            "catalog_name": catalog_name,
            "root_fqn": root_fqn,
        }

    from ...catalog.amsc import AmscClient

    try:
        async with AmscClient() as client:
            projects = await client.list_projects()
        return {
            "status": "online",
            "catalog_url": settings.amsc_openmetadata_catalog_url,
            "catalog_name": catalog_name,
            "root_fqn": root_fqn,
            "parent_fqn": settings.amsc_openmetadata_parent_fqn or None,
            "projects": projects,
            "project_count": len(projects),
        }
    except Exception as exc:
        _log.warning("AmSC catalog connectivity check failed: %s", exc)
        return {
            "status": "error",
            "catalog_url": settings.amsc_openmetadata_catalog_url,
            "catalog_name": catalog_name,
            "root_fqn": root_fqn,
            "error": str(exc),
        }


@router.get("/openmetadata/artifacts")
async def openmetadata_artifacts(
    query: str = "",
    parent_fqn: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    """Search the AmSC catalog and return recent artifacts.

    Args:
        query:      Free-text search term (empty = all).
        parent_fqn: Restrict to children of this FQN.  Falls back to
                    ``EMBLASE_AMSC_OPENMETADATA_PARENT_FQN`` if not provided.
        limit:      Max results (default 20, max 100).
    """
    import logging as _logging

    _log = _logging.getLogger(__name__)

    if not settings.amsc_openmetadata_token:
        return {"artifacts": [], "total": 0, "error": "EMBLASE_AMSC_OPENMETADATA_TOKEN not set"}

    limit = min(max(1, limit), 100)
    raw_catalog = settings.amsc_openmetadata_catalog_name or ""
    root_fqn = raw_catalog if raw_catalog else (settings.amsc_openmetadata_parent_fqn or "")
    effective_parent = parent_fqn or root_fqn or None

    from ...catalog.amsc import AmscClient

    try:
        async with AmscClient() as client:
            results = await client.search(
                query=query,
                parent_fqn=effective_parent,
                limit=limit,
            )
        return {
            "artifacts": [
                {
                    "fqn": a.fqn,
                    "name": a.name,
                    "display_name": a.display_name,
                    "entity_type": a.entity_type,
                    "description": a.description,
                    "location": a.location,
                    "parent_fqn": a.parent_fqn,
                }
                for a in results
            ],
            "total": len(results),
            "query": query,
            "parent_fqn": effective_parent,
        }
    except Exception as exc:
        _log.warning("AmSC catalog search failed: %s", exc)
        return {
            "artifacts": [],
            "total": 0,
            "error": "Internal error while querying catalog",
        }
