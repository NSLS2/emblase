"""Local compute backend — runs inference in-process."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import numpy as np

from .base import ComputeBackend, JobResult, JobStatus


class LocalBackend(ComputeBackend):
    """Runs model inference locally in a thread pool. For dev/testing."""

    def __init__(self):
        self._jobs: dict[str, JobResult] = {}

    async def submit(
        self,
        model_name: str,
        images: np.ndarray | None = None,
        latent_dim: int = 512,
        image_size: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> str:
        job_id = str(uuid.uuid4())[:8]
        self._jobs[job_id] = JobResult(job_id=job_id, status=JobStatus.running)

        def _run() -> JobResult:
            import torch

            from ..models import encode, load_model

            if images is None:
                raise ValueError("images is required for LocalBackend")
            imgs = images[:, np.newaxis] if images.ndim == 3 else images
            size = image_size or imgs.shape[-2:]
            model = load_model(model_name, latent_dim=latent_dim, image_size=size)
            output = encode(model, torch.from_numpy(imgs).float())
            return JobResult(
                job_id=job_id, status=JobStatus.completed, output_data=output
            )

        try:
            self._jobs[job_id] = await asyncio.to_thread(_run)
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
        if job_id not in self._jobs:
            raise KeyError(f"Unknown job_id: {job_id!r}")
        self._jobs[job_id] = JobResult(
            job_id=job_id, status=JobStatus.failed, error="Cancelled"
        )
