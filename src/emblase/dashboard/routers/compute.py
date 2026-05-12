"""Compute (Orion + NERSC) status and job management router for the EMBLASE dashboard."""

from __future__ import annotations

import asyncio
import logging
import logging
from typing import Any, AsyncGenerator

import httpx
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ...config import settings

_log = logging.getLogger(__name__)
_log = logging.getLogger(__name__)

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
    except Exception:
        _log.exception("Failed to fetch Orion status")
        return {"status": "error", "error": "Internal server error", "api_url": settings.orion_api_url}


@router.get("/orion/jobs")
async def orion_jobs() -> dict[str, Any]:
    """List recent Orion jobs."""
    if not settings.orion_api_key:
        return {"jobs": [], "error": "EMBLASE_ORION_API_KEY not set"}
    except Exception:
        _log.exception("Failed to fetch Orion jobs from Orion API")
        return {"jobs": [], "error": "Failed to fetch Orion jobs"}
        jobs = data if isinstance(data, list) else data.get("jobs", [])
        return {"jobs": jobs[:20]}
    except Exception as exc:
        return {"jobs": [], "error": str(exc)}


@router.get("/orion/jobs/{job_id}/status")
async def orion_job_status(job_id: str) -> dict[str, Any]:
    """Get status for a specific Orion job — returns structured state/node/stdout.

    If the job is not found (404) it is treated as completed/purged so the
    frontend stops polling.
    """
    try:
        from ...compute.orion import OrionClient

        async with OrionClient() as client:
            job = await client.get_job(int(job_id))
        return {
            "job_id": job.job_id,
            "state": job.state,
            "node": job.node,
            "stdout": job.stdout,
        }
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            # Job no longer exists in the Orion API — treat as completed/purged
            return {"job_id": job_id, "state": "COMPLETED", "node": None, "not_found": True}
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": str(exc)}


@router.delete("/orion/jobs/{job_id}")
async def orion_cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel an Orion job."""
    import logging as _logging

    _log = _logging.getLogger(__name__)
    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            headers={"x-api-key": settings.orion_api_key},
        ) as client:
            resp = await client.delete(
                f"{settings.orion_api_url.rstrip('/')}/api/v1/compute/{settings.orion_cluster}/jobs/{job_id}"
            )
            resp.raise_for_status()
        _log.info("Cancelled Orion job %s", job_id)
        return {"status": "cancelled", "job_id": job_id}
    except Exception as exc:
        _log.error("Failed to cancel Orion job %s: %s", job_id, exc)
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=str(exc))


# ── NERSC ─────────────────────────────────────────────────────────────────────


async def _nersc_client():
    """Return a configured NERSCClient (lazy import to avoid heavy deps if unused)."""
    from ...compute.nersc import NERSCClient

    return NERSCClient()


def _extract_log_path_sync(raw: dict | None) -> str:
    """Strategy 1 only (sync): parse admincomment JSON → stdoutPath.

    Returns empty string if not found or not yet populated by SLURM.
    """
    if not raw:
        return ""
    import json as _json

    status_obj = raw.get("status") or {}
    meta = status_obj.get("meta_data") or {}

    # admincomment is a JSON string populated by SLURM once the job finishes
    try:
        admin_raw = meta.get("admincomment", "") or ""
        if admin_raw:
            admin = _json.loads(admin_raw)
            path = admin.get("stdoutPath", "")
            if path and isinstance(path, str) and path.strip():
                return path.strip()
    except (ValueError, TypeError):
        pass

    # Fallback: simple field candidates
    for candidate in [
        status_obj.get("stdoutPath"),
        status_obj.get("stdout"),
        raw.get("stdout_path"),
    ]:
        if candidate and isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    return ""


async def _resolve_log_path(client: Any, raw: dict | None) -> str:
    """Resolve log path using the same two strategies as scripts/submit_nersc.py.

    Strategy 1: parse admincomment JSON → stdoutPath  (populated after job ends)
    Strategy 2: ls {workdir}/scripts/ and find the matching timestamped directory
                (works while the job is still running)
    """
    path = _extract_log_path_sync(raw)
    if path:
        return path

    if not raw:
        return ""

    status_obj = raw.get("status") or {}
    meta = status_obj.get("meta_data") or {}
    workdir = meta.get("workdir", "")
    jobname = meta.get("jobname", "")  # e.g. "emblase-bnl-nsls2-smi-vit"
    slug = jobname.removeprefix("emblase-")  # e.g. "bnl-nsls2-smi-vit"

    if not (workdir and slug):
        return ""

    scripts_dir = workdir.rstrip("/") + "/scripts"
    try:
        entries = await client.ls(scripts_dir, filesystem_resource_id="scratch")
        matches = [
            e["name"]
            for e in entries
            if e.get("name", "").rstrip("/").endswith(f"_{slug}") and e.get("type") == "d"
        ]
        if matches:
            matches.sort(reverse=True)  # newest first (timestamp prefix)
            return f"{matches[0]}/job.out"
    except Exception:
        pass

    return ""


# Keep the old name as an alias for the status endpoint
_extract_log_path = _extract_log_path_sync


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
            "available_resources": [
                r.get("name", r) if isinstance(r, dict) else r for r in resources
            ],
        }
    except httpx.TimeoutException:
        return {"status": "timeout", "api_uri": settings.nersc_api_uri}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "api_uri": settings.nersc_api_uri}


@router.get("/nersc/jobs/{job_id}/status")
async def nersc_job_status(job_id: str) -> dict[str, Any]:
    """Get status for a specific NERSC job, including discovered log path."""
    try:
        client = await _nersc_client()
        async with client:
            job = await client.get_job(job_id)
        return {
            "job_id": job.job_id,
            "state": job.state,
            "log_path": _extract_log_path(job.raw),
            "raw": job.raw,
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.delete("/nersc/jobs/{job_id}")
async def nersc_cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel a NERSC job."""
    import logging as _logging

    _log = _logging.getLogger(__name__)
    try:
        from ...compute.nersc import NERSCClient

        async with NERSCClient() as client:
            await client.cancel_job(job_id)
        _log.info("Cancelled NERSC job %s", job_id)
        return {"status": "cancelled", "job_id": job_id}
    except Exception as exc:
        _log.error("Failed to cancel NERSC job %s: %s", job_id, exc)
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=str(exc))


async def _nersc_log_generator(
    job_id: str, log_path: str, interval: float = 10.0
) -> AsyncGenerator[str, None]:
    """SSE generator that polls NERSC job logs every *interval* seconds.

    If *log_path* is empty the generator tries to discover the stdout path using
    the same two-strategy lookup as scripts/submit_nersc.py:
      1. admincomment JSON → stdoutPath  (populated after job ends)
      2. ls {workdir}/scripts/ → find timestamped dir  (works while running)
    Emits a ``state`` event each cycle so the UI can track job state.
    """
    client = await _nersc_client()
    last_line_count = 0
    resolved_path = log_path.strip()

    # IRI terminal states (raw strings returned by the API)
    _terminal = {"completed", "failed", "canceled"}

    while True:
        try:
            async with client:
                job = await client.get_job(job_id)
                state = job.state  # raw IRI string: "active", "completed", etc.

                if not resolved_path:
                    resolved_path = await _resolve_log_path(client, job.raw)

                if resolved_path:
                    try:
                        content = await client.read_file_tail(resolved_path, lines=500)
                        lines = content.splitlines()
                        new_lines = lines[last_line_count:]
                        last_line_count = len(lines)
                        for line in new_lines:
                            yield f"data: {line}\n\n"
                    except Exception as log_exc:
                        yield f"data: [log not yet available: {log_exc}]\n\n"
                else:
                    yield f"data: [locating log file… state={state}]\n\n"

                if state in _terminal:
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
