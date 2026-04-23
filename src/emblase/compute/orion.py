"""Orion compute backend for inference jobs."""

from __future__ import annotations

import base64
import io
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from ..config import settings
from .base import ComputeBackend, JobResult, JobStatus

_TEMPLATE = (Path(__file__).parent.parent / "worker" / "inference.py.tmpl").read_text()

_SLURM_STATE_MAP = {
    "PENDING": JobStatus.pending,
    "CONFIGURING": JobStatus.pending,
    "RUNNING": JobStatus.running,
    "COMPLETING": JobStatus.running,
    "COMPLETED": JobStatus.completed,
    "FAILED": JobStatus.failed,
    "CANCELLED": JobStatus.failed,
    "TIMEOUT": JobStatus.failed,
    "NODE_FAIL": JobStatus.failed,
    "OUT_OF_MEMORY": JobStatus.failed,
}


def _render_inference_script(
    job_dir: str,
    model_name: str,
    latent_dim: int,
    image_size: tuple[int, int],
    models_dir: str,
) -> str:
    """Render the inference template with concrete values."""
    return _TEMPLATE.format(
        job_dir=job_dir,
        model_name=model_name,
        latent_dim=latent_dim,
        image_size=image_size,
        models_dir=models_dir,
    )


def _build_sbatch_script(
    job_dir: str,
    python_script: str,
    images_b64: str,
    job_name: str = "emblase",
    time_limit: str = "0-00:10:00",
) -> str:
    """Build the sbatch script that decodes input data and runs inference."""
    return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail
mkdir -p {job_dir}
cd {job_dir}

# Decode input images
python3 -c "
import base64, io, numpy as np
data = base64.b64decode('{images_b64}')
arr = np.load(io.BytesIO(data))
np.save('input.npy', arr)
print(f'Input saved: {{arr.shape}}')
"

# Run inference
python3 << 'EMBLASE_INFERENCE_EOF'
{python_script}
EMBLASE_INFERENCE_EOF
"""


class OrionBackend(ComputeBackend):
    """Submit inference jobs to Orion (Slurm) via the Orion REST API."""

    def __init__(
        self,
        client: OrionClient | None = None,
        working_dir: str | None = None,
        models_dir: str | None = None,
        account: str | None = None,
    ):
        self.client = client or OrionClient()
        self.working_dir = working_dir or settings.orion_working_dir
        self.models_dir = models_dir or "~/code/emblase/models"
        self.account = account or settings.orion_account
        self._jobs: dict[int, dict[str, Any]] = {}

    async def submit(
        self,
        model_name: str,
        images: np.ndarray,
        latent_dim: int = 512,
        **kwargs: Any,
    ) -> str:
        tag = uuid.uuid4().hex[:8]
        job_dir = f"{self.working_dir}/job_{tag}"

        # Render the inference script from template
        py_script = _render_inference_script(
            job_dir=job_dir,
            model_name=model_name,
            latent_dim=latent_dim,
            image_size=images.shape[-2:],
            models_dir=self.models_dir,
        )

        # Encode images as base64 for embedding in the sbatch script
        buf = io.BytesIO()
        np.save(buf, images)
        images_b64 = base64.b64encode(buf.getvalue()).decode()

        script = _build_sbatch_script(
            job_dir=job_dir,
            python_script=py_script,
            images_b64=images_b64,
            job_name=f"emblase-{model_name}",
        )

        job_id = await self.client.submit_job(
            script=script,
            working_dir=job_dir,
            overrides={"tres_per_job": "gres/gpu:1", "account": self.account},
        )

        self._jobs[job_id] = {"job_dir": job_dir, "model_name": model_name, "tag": tag}
        return str(job_id)

    async def status(self, job_id: str) -> JobStatus:
        info = await self.client.get_job(int(job_id))
        return _SLURM_STATE_MAP.get(info.state, JobStatus.pending)

    async def result(self, job_id: str) -> JobResult:
        st = await self.status(job_id)
        if st not in (JobStatus.completed, JobStatus.failed):
            return JobResult(job_id=job_id, status=st)

        meta = self._jobs.get(int(job_id))
        if not meta:
            return JobResult(
                job_id=job_id, status=JobStatus.failed, error="Job metadata lost"
            )

        # TODO: remote filesystem access — for now assumes shared mount
        output_path = Path(meta["job_dir"]).expanduser() / "output.npy"
        if output_path.exists():
            latent = np.load(str(output_path))
            return JobResult(
                job_id=job_id, status=JobStatus.completed, latent_vectors=latent
            )

        return JobResult(
            job_id=job_id,
            status=st,
            error=f"Output not found at {output_path}"
            if st == JobStatus.completed
            else None,
        )

    async def cancel(self, job_id: str) -> None:
        await self.client.cancel_job(int(job_id))


@dataclass
class OrionJob:
    """Parsed Orion job info."""

    job_id: int
    state: str
    node: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    raw: dict[str, Any] | None = None


class OrionClient:
    """Async HTTP client for the Orion compute REST API (Slurm job submission).

    Usage::

        async with OrionClient() as client:
            job_id = await client.submit_job(script="#!/bin/bash\\nhostname")
            info = await client.get_job(job_id)
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        cluster: str | None = None,
    ):
        self.api_url = (api_url or settings.orion_api_url).rstrip("/")
        self.api_key = api_key or settings.orion_api_key
        self.cluster = cluster or settings.orion_cluster
        self._client: httpx.AsyncClient | None = None

    # -- context manager for connection reuse --

    async def __aenter__(self) -> OrionClient:
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "Use 'async with OrionClient() as client:' or call ._ensure_client()"
            )
        return self._client

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Lazily create a client if not using context manager."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            )
        return self._client

    # -- API methods --

    async def submit_job(
        self,
        script: str,
        working_dir: str = "/tmp",
        overrides: dict[str, str] | None = None,
    ) -> int:
        """Submit a job script to Orion. Returns the Slurm job ID."""
        client = await self._ensure_client()
        payload: dict[str, Any] = {
            "script": script,
            "working_dir_path": working_dir,
        }
        if overrides:
            payload["overrides"] = overrides

        resp = await client.post(
            f"{self.api_url}/api/v1/compute/{self.cluster}/jobs",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["job_id"]

    async def get_job(self, job_id: int) -> OrionJob:
        """Get info for a single job."""
        client = await self._ensure_client()
        resp = await client.get(
            f"{self.api_url}/api/v1/compute/{self.cluster}/jobs/{job_id}",
        )
        resp.raise_for_status()
        data = resp.json()["jobs"][0]

        state = data.get("state", ["UNKNOWN"])
        if isinstance(state, list):
            state = state[0] if state else "UNKNOWN"

        return OrionJob(
            job_id=data["job_id"],
            state=state,
            node=data.get("nodes"),
            stdout=data.get("stdout"),
            stderr=data.get("stderr"),
            raw=data,
        )

    async def cancel_job(self, job_id: int) -> None:
        """Cancel a running job."""
        client = await self._ensure_client()
        resp = await client.delete(
            f"{self.api_url}/api/v1/compute/{self.cluster}/jobs/{job_id}",
        )
        resp.raise_for_status()

    async def wait_for_job(
        self,
        job_id: int,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> OrionJob:
        """Poll until job reaches a terminal state."""
        import asyncio

        elapsed = 0.0
        while elapsed < timeout:
            info = await self.get_job(job_id)
            if info.state in (
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "NODE_FAIL",
            ):
                return info
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        raise TimeoutError(
            f"Job {job_id} did not complete within {timeout}s (last state: {info.state})"
        )
