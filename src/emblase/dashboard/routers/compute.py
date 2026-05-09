"""Compute (Orion + NERSC) status and job management router for the EMBLASE dashboard."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

import httpx
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ...compute.base import JobStatus
from ...config import settings

router = APIRouter(prefix="/compute", tags=["compute"])

# ── Orion ─────────────────────────────────────────────────────────────────────


async def _orion_get(path: str, timeout: float = 8.0) -> dict[str, Any]:
    async with httpx.AsyncClient(
        timeout=timeout,
        headers={"x-api-key": settings.orion_api_key},
    ) as client:
        resp = await client.get(f"{settings.orion_api_url.rstrip('/')}{path}")
        resp.raise_for_status()
        return resp.json()


@router.get("/orion/status")
async def orion_status() -> dict[str, Any]:
    """Return Orion API reachability and config summary."""
    if not settings.orion_api_key:
        return {"status": "unconfigured", "message": "EMBLASE_ORION_API_KEY not set"}
    try:
        # Attempt to list jobs to verify connectivity
        data = await _orion_get(f"/api/v1/compute/{settings.orion_cluster}/jobs")
        jobs = data if isinstance(data, list) else data.get("jobs", [])
        running = [j for j in jobs if j.get("job_state") in ("PENDING", "RUNNING", "CONFIGURING")]
        return {
            "status": "online",
            "api_url": settings.orion_api_url,
            "cluster": settings.orion_cluster,
            "account": settings.orion_account,
            "active_jobs": len(running),
            "latest_job": running[0] if running else None,
        }
    except httpx.TimeoutException:
        return {"status": "timeout", "api_url": settings.orion_api_url}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "api_url": settings.orion_api_url}


@router.get("/orion/jobs")
async def orion_jobs() -> dict[str, Any]:
    """List recent Orion jobs."""
    if not settings.orion_api_key:
        return {"jobs": [], "error": "EMBLASE_ORION_API_KEY not set"}
    try:
        data = await _orion_get(f"/api/v1/compute/{settings.orion_cluster}/jobs")
        jobs = data if isinstance(data, list) else data.get("jobs", [])
        return {"jobs": jobs[:20]}  # Return last 20
    except Exception as exc:
        return {"jobs": [], "error": str(exc)}


@router.get("/orion/jobs/{job_id}/status")
async def orion_job_status(job_id: str) -> dict[str, Any]:
    """Get status for a specific Orion job."""
    try:
        data = await _orion_get(f"/api/v1/compute/{settings.orion_cluster}/jobs/{job_id}")
        return data
    except Exception as exc:
        return {"error": str(exc)}


@router.delete("/orion/jobs/{job_id}")
async def orion_cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel an Orion job."""
    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            headers={"x-api-key": settings.orion_api_key},
        ) as client:
            resp = await client.delete(
                f"{settings.orion_api_url.rstrip('/')}/api/v1/compute/{settings.orion_cluster}/jobs/{job_id}"
            )
            resp.raise_for_status()
            return {"status": "cancelled", "job_id": job_id}
    except Exception as exc:
        return {"error": str(exc)}


# ── NERSC ─────────────────────────────────────────────────────────────────────


async def _nersc_client():
    """Return a configured NERSCClient (lazy import to avoid heavy deps if unused)."""
    from ...compute.nersc import NERSCClient

    return NERSCClient()


@router.get("/nersc/status")
async def nersc_status() -> dict[str, Any]:
    """Return NERSC API reachability and config summary."""
    if not settings.nersc_api_token:
        return {"status": "unconfigured", "message": "EMBLASE_NERSC_API_TOKEN not set"}
    try:
        client = await _nersc_client()
        async with client:
            resources = await client.discover_resources()
        return {
            "status": "online",
            "api_uri": settings.nersc_api_uri,
            "resource_id": settings.nersc_resource_id,
            "account": settings.nersc_account,
            "queue": settings.nersc_queue,
            "container_image": settings.nersc_container_image,
            "time_limit": settings.nersc_time_limit,
            "available_resources": [r.get("name", r) if isinstance(r, dict) else r for r in resources],
        }
    except httpx.TimeoutException:
        return {"status": "timeout", "api_uri": settings.nersc_api_uri}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "api_uri": settings.nersc_api_uri}


@router.get("/nersc/jobs/{job_id}/status")
async def nersc_job_status(job_id: str) -> dict[str, Any]:
    """Get status for a specific NERSC job."""
    try:
        client = await _nersc_client()
        async with client:
            job = await client.get_job(job_id)
        return {"job_id": job.job_id, "state": job.state, "raw": job.raw}
    except Exception as exc:
        return {"error": str(exc)}


@router.delete("/nersc/jobs/{job_id}")
async def nersc_cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel a NERSC job."""
    try:
        client = await _nersc_client()
        async with client:
            await client.cancel_job(job_id)
        return {"status": "cancelled", "job_id": job_id}
    except Exception as exc:
        return {"error": str(exc)}


async def _nersc_log_generator(job_id: str, log_path: str, interval: float = 10.0) -> AsyncGenerator[str, None]:
    """SSE generator that polls NERSC job logs every *interval* seconds."""
    client = await _nersc_client()
    last_line_count = 0
    while True:
        try:
            async with client:
                # First get job state
                job = await client.get_job(job_id)
                state = job.state

                # Try to get log
                try:
                    content = await client.read_file_tail(log_path, lines=200)
                    lines = content.splitlines()
                    new_lines = lines[last_line_count:]
                    last_line_count = len(lines)
                    for line in new_lines:
                        yield f"data: {line}\n\n"
                except Exception as log_exc:
                    yield f"data: [log not yet available: {log_exc}]\n\n"

                # Signal terminal states
                if state in ("completed", "failed", "canceled"):
                    yield f"event: job_done\ndata: {state}\n\n"
                    break
                else:
                    yield f"event: state\ndata: {state}\n\n"

        except Exception as exc:
            yield f"data: [error polling logs: {exc}]\n\n"

        await asyncio.sleep(interval)


@router.get("/nersc/jobs/{job_id}/logs")
async def nersc_job_logs(job_id: str, log_path: str = "") -> StreamingResponse:
    """Stream NERSC job logs as Server-Sent Events (polled every 10 s)."""
    return StreamingResponse(
        _nersc_log_generator(job_id, log_path, interval=10.0),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
