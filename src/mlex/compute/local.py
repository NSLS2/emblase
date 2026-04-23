"""Local compute backend — runs inference in-process."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import torch

from .base import ComputeBackend, JobResult, JobStatus


class LocalBackend(ComputeBackend):
    """Runs model inference locally (in the FastAPI process). For dev/testing."""

    def __init__(self):
        self._jobs: dict[str, JobResult] = {}

    async def submit(
        self,
        model_name: str,
        images: np.ndarray,
        latent_dim: int = 512,
        **kwargs: Any,
    ) -> str:
        job_id = str(uuid.uuid4())[:8]
        self._jobs[job_id] = JobResult(job_id=job_id, status=JobStatus.running)

        try:
            from ..models import encode, load_model

            model = load_model(
                model_name,
                latent_dim=latent_dim,
                image_size=images.shape[-2:],
            )
            # images: (B, H, W) -> (B, 1, H, W)
            if images.ndim == 3:
                images = images[:, np.newaxis, :, :]
            tensor = torch.from_numpy(images).float()

            latent = encode(model, tensor, model_name)
            self._jobs[job_id] = JobResult(
                job_id=job_id,
                status=JobStatus.completed,
                latent_vectors=latent,
            )
        except Exception as e:
            self._jobs[job_id] = JobResult(
                job_id=job_id, status=JobStatus.failed, error=str(e)
            )

        return job_id

    async def status(self, job_id: str) -> JobStatus:
        return self._jobs[job_id].status

    async def result(self, job_id: str) -> JobResult:
        return self._jobs[job_id]

    async def cancel(self, job_id: str) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.failed
            self._jobs[job_id].error = "Cancelled"
