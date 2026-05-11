"""Job submission router for the EMBLASE dashboard.

Supports:
- Batch inference jobs (Orion or NERSC)
- Streaming inference jobs (NERSC or Orion)
"""

from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
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
    model_name: str = "bnl-nsls2-smi-vit"
    mlflow_version: str = ""
    input_container: str = ""
    output_container: str = ""
    batch_size: int = Field(default=1, ge=1, le=256)
    image_key: str = "primary/pil900KW_image"
    thumb_mode: str = "logroi"
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
    backend: Literal["nersc", "orion"] = "nersc"
    model_name: str = "bnl-nsls2-smi-vit"
    mlflow_version: str = ""
    run_path: str = ""
    output_container: str = ""
    batch_size: int = Field(default=8, ge=1, le=256)
    image_key: str = "primary/pil900KW_image"
    thumb_mode: str = "logroi"
    param_specs: list[ParamSpec] = []
    projector: str | None = None
    classifier: str | None = None
    # NERSC-specific
    nersc_queue: str = ""
    nersc_account: str = ""
    nersc_time_limit: str = "02:00:00"
    nersc_constraint: str = ""
    ws_max_size: int = 64 * 1024 * 1024
    # Orion-specific
    orion_account: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _param_specs_dict(specs: list[ParamSpec]) -> dict | None:
    """Convert ParamSpec list to the dict format expected by backend submit()."""
    if not specs:
        return None
    return {s.name: {"source": s.source, "dtype": s.dtype, "units": s.units} for s in specs}


def _or_none(value: str) -> str | None:
    """Return None for empty strings (avoids overriding backend defaults)."""
    return value or None


async def _submit_batch_nersc(req: BatchJobRequest) -> tuple[str, str]:
    """Submit a batch job to NERSC. Returns (job_id, log_path)."""
    from ...compute.nersc import NERSCBackend

    backend = NERSCBackend(
        account=_or_none(req.nersc_account),
        queue=_or_none(req.nersc_queue),
        time_limit=_or_none(req.nersc_time_limit),
        constraint=_or_none(req.nersc_constraint),
    )
    job_id = await backend.submit(
        model_name=req.model_name,
        batch_size=req.batch_size,
        run_path=resolve_tiled_path(_or_none(req.input_container), settings.tiled_input_container),
        image_key=req.image_key,
        output=resolve_tiled_path(_or_none(req.output_container), settings.tiled_output_container),
        mlflow_version=req.mlflow_version,
        thumb_mode=req.thumb_mode,
        param_specs=_param_specs_dict(req.param_specs),
        projector=req.projector,
        classifier=req.classifier,
    )
    return job_id, backend._jobs.get(job_id, {}).get("log_path", "")


async def _submit_stream_nersc(req: StreamJobRequest) -> tuple[str, str]:
    """Submit a streaming job to NERSC. Returns (job_id, log_path)."""
    from ...compute.nersc import NERSCBackend

    backend = NERSCBackend(
        account=_or_none(req.nersc_account),
        queue=_or_none(req.nersc_queue),
        time_limit=_or_none(req.nersc_time_limit),
        constraint=_or_none(req.nersc_constraint),
    )
    job_id = await backend.submit_streaming(
        run_path=resolve_tiled_path(_or_none(req.run_path), settings.tiled_input_container),
        output=resolve_tiled_path(_or_none(req.output_container), settings.tiled_output_container),
        model_name=req.model_name,
        batch_size=req.batch_size,
        mlflow_version=req.mlflow_version,
        thumb_mode=req.thumb_mode,
        image_key=req.image_key,
        ws_max_size=req.ws_max_size,
        param_specs=_param_specs_dict(req.param_specs),
        projector=req.projector,
        classifier=req.classifier,
        time_limit=_or_none(req.nersc_time_limit),
    )
    return job_id, backend._jobs.get(job_id, {}).get("log_path", "")


async def _submit_batch_orion(req: BatchJobRequest) -> tuple[str, str]:
    """Submit a batch job to Orion. Returns (job_id, log_path)."""
    from ...compute.orion import OrionBackend

    backend = OrionBackend(account=_or_none(req.orion_account))
    job_id = await backend.submit(
        model_name=req.model_name,
        batch_size=req.batch_size,
        run_path=resolve_tiled_path(_or_none(req.input_container), settings.tiled_input_container),
        image_key=req.image_key,
        output=resolve_tiled_path(_or_none(req.output_container), settings.tiled_output_container),
        mlflow_version=req.mlflow_version,
        thumb_mode=req.thumb_mode,
        param_specs=_param_specs_dict(req.param_specs),
        projector=req.projector,
        classifier=req.classifier,
    )
    return job_id, ""


async def _submit_stream_orion(req: StreamJobRequest) -> tuple[str, str]:
    """Submit a streaming job to Orion. Returns (job_id, log_path)."""
    from ...compute.orion import OrionBackend

    backend = OrionBackend(account=_or_none(req.orion_account))
    job_id = await backend.submit_streaming(
        run_path=resolve_tiled_path(_or_none(req.run_path), settings.tiled_input_container),
        output=resolve_tiled_path(_or_none(req.output_container), settings.tiled_output_container),
        model_name=req.model_name,
        batch_size=req.batch_size,
        mlflow_version=req.mlflow_version,
        thumb_mode=req.thumb_mode,
        image_key=req.image_key,
        param_specs=_param_specs_dict(req.param_specs),
        projector=req.projector,
        classifier=req.classifier,
    )
    return job_id, ""


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/batch")
async def submit_batch(req: BatchJobRequest) -> dict[str, Any]:
    """Submit a batch inference job and return the job ID."""
    try:
        if req.backend == "nersc":
            job_id, log_path = await _submit_batch_nersc(req)
        else:
            job_id, log_path = await _submit_batch_orion(req)
        return {"job_id": job_id, "backend": req.backend, "log_path": log_path, "submitted_at": time.time()}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.post("/stream")
async def submit_stream(req: StreamJobRequest) -> dict[str, Any]:
    """Submit a streaming inference job (NERSC or Orion) and return the job ID."""
    try:
        if req.backend == "nersc":
            job_id, log_path = await _submit_stream_nersc(req)
        else:
            job_id, log_path = await _submit_stream_orion(req)
        return {"job_id": job_id, "backend": req.backend, "log_path": log_path, "submitted_at": time.time()}
    except Exception as exc:
        raise HTTPException(500, str(exc))
