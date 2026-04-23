"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel


class ModelName(str, Enum):
    vae = "vae"
    vit = "vit"


class EvaluateRequest(BaseModel):
    """Request to run dimensionality reduction on images."""

    model: ModelName = ModelName.vae
    # Images as nested lists (batch x H x W), or we accept numpy-compatible input
    images: list[list[list[float]]] | None = None
    # Alternatively, generate N random dummy images of given size
    dummy_images: int | None = 4
    image_size: tuple[int, int] = (512, 512)
    latent_dim: int = 512


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class EvaluateResponse(BaseModel):
    job_id: str
    status: JobStatus
    latent_vectors: list[list[float]] | None = None
    error: str | None = None
