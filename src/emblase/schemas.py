"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from .compute.base import JobStatus  # single source of truth

__all__ = ["ModelName", "JobStatus", "EvaluateRequest", "EvaluateResponse"]


class ModelName(str, Enum):
    vae = "vae"
    vit = "vit"


class EvaluateRequest(BaseModel):
    """Request to run dimensionality reduction on images."""

    model: ModelName = ModelName.vae
    images: list[list[list[float]]] | None = None
    dummy_images: int | None = 4
    image_size: tuple[int, int] = (512, 512)
    latent_dim: int = 512


class EvaluateResponse(BaseModel):
    job_id: str
    status: JobStatus
    latent_vectors: list[list[float]] | None = None
    error: str | None = None
