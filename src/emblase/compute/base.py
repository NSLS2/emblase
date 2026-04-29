"""Abstract base class for compute backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


@dataclass
class JobResult:
    job_id: str
    status: JobStatus
    output_data: np.ndarray | None = None
    error: str | None = None


class ComputeBackend(ABC):
    """Interface for submitting and tracking inference jobs."""

    @abstractmethod
    async def submit(self, model_name: str, **kwargs: Any) -> str:
        """Submit an inference job. Returns a job_id."""

    @abstractmethod
    async def status(self, job_id: str) -> JobStatus:
        """Check the status of a submitted job."""

    @abstractmethod
    async def result(self, job_id: str) -> JobResult:
        """Retrieve the result of a completed job."""

    @abstractmethod
    async def cancel(self, job_id: str) -> None:
        """Cancel a running job."""
