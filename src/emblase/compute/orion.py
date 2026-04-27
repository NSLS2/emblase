"""Orion compute backend for inference jobs."""

from __future__ import annotations

import asyncio
import base64
import io
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
    model_name: str,
    models_dir: str,
    batch_size: int = 1,
    output_mode: str = "none",
    tiled_entries: list[str | tuple[str, str]] | None = None,
    tiled_result_path: str = "",
    mlflow_version: str = "",
) -> str:
    """Render the inference template with concrete values."""
    return _TEMPLATE.format(
        model_name=model_name,
        models_dir=models_dir,
        batch_size=batch_size,
        output_mode=output_mode,
        tiled_entries=repr(tiled_entries or []),
        tiled_result_path=tiled_result_path,
        mlflow_version=mlflow_version,
    )


def _build_sbatch_script(
    working_dir: str,
    python_script: str,
    project_dir: str,
    job_name: str = "emblase",
    time_limit: str = "0-00:10:00",
    images_b64: str | None = None,
    image_path: str | None = None,
    tiled_mode: bool = False,
) -> str:
    """Build the sbatch script. Exactly one image source must be indicated:

    - images_b64: encoded client-side, embedded as heredoc, decoded on the node.
    - image_path: .npy already on Orion; symlinked into the job dir.
    - tiled_mode: input is read directly by the inference script via tiled_io;
      no pre-fetch step needed in the sbatch preamble.
    """
    sources = [bool(images_b64), bool(image_path), tiled_mode]
    if sum(sources) != 1:
        raise ValueError("Provide exactly one of images_b64, image_path, tiled_mode=True")

    if images_b64:
        input_section = f"""\
cat > _input_b64.txt << 'EMBLASE_B64_EOF'
{images_b64}
EMBLASE_B64_EOF

cd {project_dir} && pixi run python << 'EMBLASE_DECODE_EOF'
import base64, io, numpy as np, os
job_dir = os.environ["JOB_DIR"]
with open(f"{{job_dir}}/_input_b64.txt") as f:
    data = base64.b64decode(f.read().strip())
arr = np.load(io.BytesIO(data))
np.save(f"{{job_dir}}/input.npy", arr)
print(f"Input saved: {{arr.shape}}")
EMBLASE_DECODE_EOF

rm -f "$JOB_DIR/_input_b64.txt"
"""
    elif image_path:
        input_section = f'ln -sf {image_path} "$JOB_DIR/input.npy"\n'
    else:
        input_section = ""  # tiled_mode: inference script reads directly from Tiled

    return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output={working_dir}/slurm-%j.out
#SBATCH --error={working_dir}/slurm-%j.out

set -euo pipefail
set -x

JOB_DIR={working_dir}/job_${{SLURM_JOB_ID}}
export JOB_DIR
mkdir -p "$JOB_DIR"
cd "$JOB_DIR"

{input_section}
cd {project_dir} && pixi run python << 'EMBLASE_INFERENCE_EOF'
{python_script}
EMBLASE_INFERENCE_EOF
"""


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

    async def __aenter__(self) -> OrionClient:
        await self._ensure_client()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            )
        return self._client

    async def submit_job(
        self,
        script: str,
        working_dir: str = "/tmp",
        overrides: dict[str, str] | None = None,
        environment: list[str] | None = None,
    ) -> int:
        """Submit a job script to Orion. Returns the Slurm job ID."""
        client = await self._ensure_client()
        payload: dict[str, Any] = {"script": script, "working_dir_path": working_dir}
        if overrides:
            payload["overrides"] = overrides
        if environment:
            payload["environment"] = environment
        resp = await client.post(
            f"{self.api_url}/api/v1/compute/{self.cluster}/jobs", json=payload
        )
        resp.raise_for_status()
        return resp.json()["job_id"]

    async def get_job(self, job_id: int) -> OrionJob:
        """Get info for a single job."""
        client = await self._ensure_client()
        resp = await client.get(
            f"{self.api_url}/api/v1/compute/{self.cluster}/jobs/{job_id}"
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
            f"{self.api_url}/api/v1/compute/{self.cluster}/jobs/{job_id}"
        )
        resp.raise_for_status()

    async def wait_for_job(
        self,
        job_id: int,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> OrionJob:
        """Poll until job reaches a terminal state."""
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


class OrionBackend(ComputeBackend):
    """Submit inference jobs to Orion (Slurm) via the Orion REST API."""

    def __init__(
        self,
        client: OrionClient | None = None,
        working_dir: str | None = None,
        models_dir: str | None = None,
        project_dir: str | None = None,
        home: str | None = None,
        account: str | None = None,
        path: str | None = None,
    ):
        self.client = client or OrionClient()
        self.working_dir = working_dir or settings.orion_working_dir
        self.models_dir = models_dir or settings.orion_models_dir
        self.project_dir = project_dir or settings.orion_project_dir
        self.home = home or settings.orion_home
        self.account = account or settings.orion_account
        self.path = path or settings.orion_path
        self._jobs: dict[int, dict[str, Any]] = {}

    async def submit(
        self,
        model_name: str,
        batch_size: int = 1,
        image_data: np.ndarray | None = None,
        image_path: str | None = None,
        tiled_entries: list[str | tuple[str, str]] | None = None,
        output_mode: str = "none",
        tiled_result_path: str = "",
        mlflow_version: str = "",
        **kwargs: Any,
    ) -> str:
        """Submit an inference job. Exactly one image source must be provided:

        - image_data: numpy array — encoded client-side and embedded in the script.
        - image_path: absolute path to a .npy file already on Orion.
        - tiled_entries: list of Tiled path strings or (path, slice) tuples —
          read on the node via tiled_io.

        output_mode:
        - "none"  — output.npy saved on the node only.
        - "tiled" — read from tiled_entries, write output to tiled_result_path.
        """
        py_script = _render_inference_script(
            model_name=model_name,
            models_dir=self.models_dir,
            batch_size=batch_size,
            output_mode=output_mode,
            tiled_entries=tiled_entries,
            tiled_result_path=tiled_result_path,
            mlflow_version=mlflow_version,
        )

        if image_data is not None:
            buf = io.BytesIO()
            np.save(buf, image_data)
            raw = buf.getvalue()
            b64 = base64.b64encode(raw).decode()
            _MAX_EMBED_BYTES = 10 * 1024 * 1024  # 10 MB — Orion API request size limit
            if len(b64) > _MAX_EMBED_BYTES:
                raise ValueError(
                    f"Image payload too large to embed ({len(raw) / 1024 / 1024:.1f} MB raw, "
                    f"{len(b64) / 1024 / 1024:.1f} MB base64). "
                    "Copy the .npy file to Orion and use --orion-path instead."
                )
            script_kwargs: dict[str, Any] = {"images_b64": b64}
        elif image_path:
            script_kwargs = {"image_path": image_path}
        else:
            script_kwargs = {"tiled_mode": True}

        script = _build_sbatch_script(
            working_dir=self.working_dir,
            python_script=py_script,
            project_dir=self.project_dir,
            job_name=f"emblase-{model_name}",
            **script_kwargs,
        )

        environment = [f"PATH={self.path}", f"HOME={self.home}", "SLURM_EXPORT_ENV=ALL"]
        if output_mode == "tiled" or tiled_entries:
            if settings.tiled_server_url:
                environment.append(f"EMBLASE_TILED_SERVER_URL={settings.tiled_server_url}")
            if settings.tiled_api_key:
                environment.append(f"EMBLASE_TILED_API_KEY={settings.tiled_api_key}")
        if settings.mlflow_tracking_uri:
            environment.append(f"EMBLASE_MLFLOW_TRACKING_URI={settings.mlflow_tracking_uri}")
            if settings.mlflow_api_key:
                environment.append(f"EMBLASE_MLFLOW_API_KEY={settings.mlflow_api_key}")
            if settings.model_cache_dir:
                environment.append(f"EMBLASE_MODEL_CACHE_DIR={settings.model_cache_dir}")

        job_id = await self.client.submit_job(
            script=script,
            working_dir=self.working_dir,
            overrides={"tres_per_job": "gres/gpu:1", "account": self.account},
            environment=environment,
        )

        job_dir = f"{self.working_dir}/job_{job_id}"
        self._jobs[job_id] = {
            "job_dir": job_dir,
            "model_name": model_name,
            "output_mode": output_mode,
        }
        return str(job_id)

    async def status(self, job_id: str) -> JobStatus:
        info = await self.client.get_job(int(job_id))
        return _SLURM_STATE_MAP.get(info.state, JobStatus.pending)

    async def result(self, job_id: str) -> JobResult:
        st = await self.status(job_id)
        meta = self._jobs.get(int(job_id))
        if not meta:
            return JobResult(
                job_id=job_id, status=JobStatus.failed, error="Job metadata lost"
            )

        if st not in (JobStatus.completed, JobStatus.failed):
            return JobResult(job_id=job_id, status=st)

        output_mode = meta.get("output_mode", "none")
        if output_mode == "tiled":
            # Results were written to Tiled on the node — nothing to return here.
            return JobResult(job_id=job_id, status=st)

        # "none" mode — try shared filesystem.
        output_path = Path(meta["job_dir"]) / "output.npy"
        if output_path.exists():
            output = np.load(str(output_path))
            return JobResult(
                job_id=job_id, status=JobStatus.completed, output_data=output
            )

        return JobResult(
            job_id=job_id,
            status=st,
            error=f"Output not found at {output_path}"
            if st == JobStatus.completed
            else None,
        )

    async def wait(
        self,
        job_id: str,
        poll_interval: float = 3.0,
        timeout: float = 300.0,
    ) -> JobStatus:
        """Poll until the job reaches a terminal state. Returns final JobStatus."""
        info = await self.client.wait_for_job(
            int(job_id), poll_interval=poll_interval, timeout=timeout
        )
        return _SLURM_STATE_MAP.get(info.state, JobStatus.failed)

    async def cancel(self, job_id: str) -> None:
        await self.client.cancel_job(int(job_id))
