"""Tiled status router for the EMBLASE dashboard."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import APIRouter

from ...config import settings

router = APIRouter(prefix="/tiled", tags=["tiled"])


async def _tiled_get(path: str, timeout: float = 8.0) -> dict[str, Any]:
    """Perform a GET request against the Tiled server and return parsed JSON."""
    uri = settings.tiled_server_uri.rstrip("/")
    headers: dict[str, str] = {}
    if settings.tiled_api_key:
        headers["Authorization"] = f"Apikey {settings.tiled_api_key}"
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        resp = await client.get(f"{uri}{path}")
        resp.raise_for_status()
        return resp.json()


@router.get("/status")
async def tiled_status() -> dict[str, Any]:
    """Return Tiled server status: reachability, version, and container info."""
    if not settings.tiled_server_uri:
        return {"status": "unconfigured", "message": "EMBLASE_TILED_SERVER_URI not set"}

    try:
        data = await _tiled_get("/api/v1/")

        # Fetch output container item count in parallel (best-effort)
        output_count: int | None = None
        output_path = settings.tiled_output_container
        if output_path:
            try:
                search = await _tiled_get(
                    f"/api/v1/search/{output_path.strip('/')}?page[limit]=0",
                    timeout=5.0,
                )
                output_count = search.get("meta", {}).get("count")
            except Exception:
                pass

        return {
            "status": "online",
            "server_uri": settings.tiled_server_uri,
            "tiled_version": data.get("library_version", "unknown"),
            "api_version": data.get("api_version", "unknown"),
            "output_container": output_path or None,
            "output_count": output_count,
        }
    except httpx.TimeoutException:
        return {"status": "timeout", "server_uri": settings.tiled_server_uri}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "server_uri": settings.tiled_server_uri}


@router.get("/container")
async def tiled_container_info() -> dict[str, Any]:
    """Return metadata about the input and output Tiled containers (frame/embedding counts)."""
    if not settings.tiled_server_uri:
        return {"input": None, "output": None}

    async def _container_info(path: str) -> dict[str, Any] | None:
        if not path:
            return None
        try:
            data = await _tiled_get(f"/api/v1/metadata/{path}")
            attributes = data.get("data", {}).get("attributes", {})
            return {
                "path": path,
                "structure_family": attributes.get("structure_family", "unknown"),
                "specs": [s.get("name") for s in attributes.get("specs", [])],
                "metadata": attributes.get("metadata", {}),
            }
        except Exception as exc:
            return {"path": path, "error": str(exc)}

    input_info, output_info = await asyncio.gather(
        _container_info(settings.tiled_input_container),
        _container_info(settings.tiled_output_container),
    )
    return {"input": input_info, "output": output_info}
