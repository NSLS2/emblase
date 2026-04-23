"""Emblase — EMBeddings and LAtent Space Explorer — FastAPI application."""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException

from .compute.base import ComputeBackend, JobResult
from .compute.local import LocalBackend
from .compute.orion import OrionBackend
from .config import settings
from .schemas import EvaluateRequest, EvaluateResponse


def _create_backend() -> ComputeBackend:
    return OrionBackend() if settings.compute_backend == "orion" else LocalBackend()


def _to_response(job_id: str, result: JobResult) -> EvaluateResponse:
    return EvaluateResponse(
        job_id=job_id,
        status=result.status,
        output_data=result.output_data.tolist()
        if result.output_data is not None
        else None,
        error=result.error,
    )


backend: ComputeBackend = _create_backend()

app = FastAPI(
    title="Emblase",
    description="EMBeddings and LAtent Space Explorer — Dimensionality reduction service for synchrotron images",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {"status": "ok", "backend": settings.compute_backend}


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    """Submit images for dimensionality reduction."""
    if req.images is not None:
        images = np.array(req.images, dtype=np.float32)
    elif req.dummy_images:
        images = np.random.rand(req.dummy_images, *req.image_size).astype(np.float32)
    else:
        raise HTTPException(400, "Provide either `images` or `dummy_images`")

    job_id = await backend.submit(
        model_name=req.model.value,
        image_data=images,
        image_size=req.image_size,
        latent_dim=req.latent_dim,
        output_mode=req.output_mode.value,
        tiled_result_path=req.tiled_result_path or "",
    )
    return _to_response(job_id, await backend.result(job_id))


@app.get("/jobs/{job_id}", response_model=EvaluateResponse)
async def get_job(job_id: str):
    """Check job status and retrieve results."""
    try:
        return _to_response(job_id, await backend.result(job_id))
    except KeyError:
        raise HTTPException(404, f"Job {job_id} not found")


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    try:
        await backend.cancel(job_id)
    except KeyError:
        raise HTTPException(404, f"Job {job_id} not found")
    return {"status": "cancelled", "job_id": job_id}
