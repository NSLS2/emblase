"""Emblase — EMBeddings and LAtent Space Explorer — FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from .compute.base import ComputeBackend
from .compute.local import LocalBackend
from .compute.orion import OrionBackend
from .config import settings
from .schemas import EvaluateRequest, EvaluateResponse


def _create_backend() -> ComputeBackend:
    if settings.compute_backend == "orion":
        return OrionBackend()
    return LocalBackend()


backend: ComputeBackend = _create_backend()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Emblase",
    description="EMBeddings and LAtent Space Explorer — Dimensionality reduction service for synchrotron images",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "backend": settings.compute_backend}


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    """Submit images for dimensionality reduction.

    Either provide `images` (list of 2D arrays) or set `dummy_images` to
    generate random test images.
    """
    if req.images is not None:
        images = np.array(req.images, dtype=np.float32)
    elif req.dummy_images:
        images = np.random.rand(req.dummy_images, *req.image_size).astype(np.float32)
    else:
        raise HTTPException(400, "Provide either `images` or `dummy_images`")

    job_id = await backend.submit(
        model_name=req.model.value,
        images=images,
        latent_dim=req.latent_dim,
    )

    result = await backend.result(job_id)
    return EvaluateResponse(
        job_id=job_id,
        status=result.status,
        latent_vectors=result.latent_vectors.tolist()
        if result.latent_vectors is not None
        else None,
        error=result.error,
    )


@app.get("/jobs/{job_id}", response_model=EvaluateResponse)
async def get_job(job_id: str):
    """Check job status and retrieve results."""
    try:
        result = await backend.result(job_id)
    except KeyError:
        raise HTTPException(404, f"Job {job_id} not found")

    return EvaluateResponse(
        job_id=job_id,
        status=result.status,
        latent_vectors=result.latent_vectors.tolist()
        if result.latent_vectors is not None
        else None,
        error=result.error,
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    try:
        await backend.cancel(job_id)
    except KeyError:
        raise HTTPException(404, f"Job {job_id} not found")
    return {"status": "cancelled", "job_id": job_id}
