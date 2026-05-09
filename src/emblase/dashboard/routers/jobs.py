"""Job submission router for the EMBLASE dashboard.

Supports:
- Batch inference jobs (Orion or NERSC)
- Streaming inference jobs (NERSC only)
- Real-time log streaming via SSE after submission
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncGenerator, Literal

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...config import settings, resolve_tiled_path

router = APIRouter(prefix="/jobs", tags=["jobs"])

# ── Request models ────────────────────────────────────────────────────────────


class ParamSpec(BaseModel):
    name: str
    source: str
    dtype: str = "float"
    units: str = ""


class BatchJobRequest(BaseModel):
    backend: Literal["orion", "nersc"]
    model_name: str = "vae"
    mlflow_version: str = ""
    input_container: str = ""
    output_container: str = ""
    batch_size: int = Field(default=1, ge=1, le=256)
    image_key: str = "primary/pil900KW_image"
    thumb_mode: str = "default"
    param_specs: list[ParamSpec] = []
    projector: str | None = None
    classifier: str | None = None
    # NERSC-specific
    nersc_queue: str = ""
    nersc_account: str = ""
    nersc_time_limit: str = ""
    nersc_constraint: str = ""
    # Orion-specific
    orion_account: str = ""


class StreamJobRequest(BaseModel):
    """Streaming inference job (NERSC only — subscribes to Tiled WebSocket)."""
    model_name: str = "vae"
    mlflow_version: str = ""
    run_path: str = ""
    output_container: str = ""
    batch_size: int = Field(default=8, ge=1, le=256)
    image_key: str = "primary/pil900KW_image"
    thumb_mode: str = "logroi"
    param_specs: list[ParamSpec] = []
    projector: str | None = None
    classifier: str | None = None
    nersc_queue: str = ""
    nersc_account: str = ""
    nersc_time_limit: str = "02:00:00"
    nersc_constraint: str = ""
    ws_max_size: int = 64 * 1024 * 1024


# ── Helpers ───────────────────────────────────────────────────────────────────


def _param_specs_dict(specs: list[ParamSpec]) -> dict:
    return {s.name: {"source": s.source, "dtype": s.dtype, "units": s.units} for s in specs}


async def _submit_batch_nersc(req: BatchJobRequest) -> tuple[str, str]:
    """Submit a batch job to NERSC. Returns (job_id, log_path)."""
    from ...compute.nersc import NERSCBackend

    backend = NERSCBackend(
        account=req.nersc_account or None,
        queue=req.nersc_queue or None,
        time_limit=req.nersc_time_limit or None,
        constraint=req.nersc_constraint or None,
    )
    output = resolve_tiled_path(req.output_container or None, settings.tiled_output_container)
    run_path = resolve_tiled_path(req.input_container or None, settings.tiled_input_container)
    job_id = await backend.submit(
        model_name=req.model_name,
        batch_size=req.batch_size,
        run_path=run_path,
        image_key=req.image_key,
        output=output,
        mlflow_version=req.mlflow_version,
        thumb_mode=req.thumb_mode,
        param_specs=_param_specs_dict(req.param_specs) or None,
        projector=req.projector,
        classifier=req.classifier,
    )
    log_path = backend._jobs.get(job_id, {}).get("log_path", "")
    return job_id, log_path


async def _submit_stream_nersc(req: StreamJobRequest) -> tuple[str, str]:
    """Submit a streaming job to NERSC. Returns (job_id, log_path)."""
    from ...compute.nersc import NERSCBackend

    backend = NERSCBackend(
        account=req.nersc_account or None,
        queue=req.nersc_queue or None,
        time_limit=req.nersc_time_limit or None,
        constraint=req.nersc_constraint or None,
    )
    output = resolve_tiled_path(req.output_container or None, settings.tiled_output_container)
    run_path = resolve_tiled_path(req.run_path or None, settings.tiled_input_container)
    job_id = await backend.submit_streaming(
        run_path=run_path,
        output=output,
        model_name=req.model_name,
        batch_size=req.batch_size,
        mlflow_version=req.mlflow_version,
        thumb_mode=req.thumb_mode,
        image_key=req.image_key,
        ws_max_size=req.ws_max_size,
        param_specs=_param_specs_dict(req.param_specs) or None,
        projector=req.projector,
        classifier=req.classifier,
        time_limit=req.nersc_time_limit or None,
    )
    log_path = backend._jobs.get(job_id, {}).get("log_path", "")
    return job_id, log_path


async def _submit_batch_orion(req: BatchJobRequest) -> tuple[str, str]:
    """Submit a batch job to Orion. Returns (job_id, log_path)."""
    from ...compute.orion import OrionBackend

    backend = OrionBackend(
        account=req.orion_account or None,
    )
    output = resolve_tiled_path(req.output_container or None, settings.tiled_output_container)
    run_path = resolve_tiled_path(req.input_container or None, settings.tiled_input_container)
    job_id = await backend.submit(
        model_name=req.model_name,
        batch_size=req.batch_size,
        run_path=run_path,
        image_key=req.image_key,
        output=output,
        mlflow_version=req.mlflow_version,
        thumb_mode=req.thumb_mode,
        param_specs=_param_specs_dict(req.param_specs) or None,
        projector=req.projector,
        classifier=req.classifier,
    )
    return job_id, ""


# ── SSE job log streaming ─────────────────────────────────────────────────────


async def _orion_log_generator(job_id: str, interval: float = 5.0) -> AsyncGenerator[str, None]:
    """SSE generator that polls Orion job status every *interval* seconds."""
    from ...compute.orion import OrionClient

    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}
    while True:
        try:
            async with OrionClient() as client:
                job = await client.get_job(int(job_id))
            state = job.state if hasattr(job, "state") else "unknown"
            yield f"event: state\ndata: {state}\n\n"
            if state in terminal:
                yield f"event: job_done\ndata: {state}\n\n"
                break
        except Exception as exc:
            yield f"data: [error: {exc}]\n\n"
        await asyncio.sleep(interval)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/batch")
async def submit_batch(req: BatchJobRequest) -> dict[str, Any]:
    """Submit a batch inference job and return the job ID."""
    try:
        if req.backend == "nersc":
            job_id, log_path = await _submit_batch_nersc(req)
        elif req.backend == "orion":
            job_id, log_path = await _submit_batch_orion(req)
        else:
            raise HTTPException(400, f"Unknown backend: {req.backend}")
        return {
            "job_id": job_id,
            "backend": req.backend,
            "log_path": log_path,
            "submitted_at": time.time(),
        }
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.post("/stream")
async def submit_stream(req: StreamJobRequest) -> dict[str, Any]:
    """Submit a streaming inference job (NERSC only) and return the job ID."""
    try:
        job_id, log_path = await _submit_stream_nersc(req)
        return {
            "job_id": job_id,
            "backend": "nersc",
            "log_path": log_path,
            "submitted_at": time.time(),
        }
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/orion/{job_id}/logs")
async def orion_logs_sse(job_id: str) -> StreamingResponse:
    """Stream Orion job state updates as SSE (polled every 5 s)."""
    return StreamingResponse(
        _orion_log_generator(job_id, interval=5.0),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
