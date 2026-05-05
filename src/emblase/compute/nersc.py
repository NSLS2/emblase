"""NERSC compute backend for inference jobs via the IRI API + Shifter containers.

Auth:
    A Globus bearer token scoped to the IRI API is required.  Set
    ``EMBLASE_NERSC_API_TOKEN`` in your ``.env`` file.  The token must have
    scope ``https://auth.globus.org/scopes/ed3e577d-f7f3-4639-b96e-ff5a8445d699/iri_api``.

Job delivery:
    The rendered Python inference script is uploaded to
    ``{nersc_working_dir}/job_{task_id}/inference.py`` via the IRI filesystem
    upload endpoint, then executed inside a Shifter container::

        shifter --image=<container_image> python /path/to/inference.py

Resource discovery:
    Call ``await NERSCClient.discover_resources()`` to list available resource
    IDs (e.g. "perlmutter").  The default ``resource_id`` ("perlmutter") is
    configurable via ``EMBLASE_NERSC_RESOURCE_ID``.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from ..config import settings
from .base import ComputeBackend, JobResult, JobStatus
from .orion import (
    _projector_mode_and_name,
    _render_inference_script,
    _render_streaming_inference_script,
)

_IRI_BASE = "https://api.nersc.gov/api/v1.2"

_NERSC_STATE_MAP: dict[str, JobStatus] = {
    # SLURM states forwarded by IRI
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
    # IRI wrapper states
    "queued": JobStatus.pending,
    "running": JobStatus.running,
    "completed": JobStatus.completed,
    "failed": JobStatus.failed,
}


@dataclass
class NERSCJob:
    """Parsed NERSC IRI job info."""

    job_id: str
    state: str
    raw: dict[str, Any] | None = None


class NERSCClient:
    """Async HTTP client for the NERSC IRI REST API.

    Usage::

        async with NERSCClient() as client:
            resources = await client.discover_resources()
            task_id = await client.submit_job(script="#!/bin/bash\\nhostname")
            info = await client.get_job(task_id)
    """

    def __init__(
        self,
        api_token: str | None = None,
        resource_id: str | None = None,
        base_url: str = _IRI_BASE,
    ):
        self.api_token = api_token or settings.nersc_api_token
        self.resource_id = resource_id or settings.nersc_resource_id
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> NERSCClient:
        await self._ensure_client()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            if not self.api_token:
                raise ValueError(
                    "NERSC API token is not set.  "
                    "Set EMBLASE_NERSC_API_TOKEN in your .env file."
                )
            self._client = httpx.AsyncClient(
                timeout=60.0,
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
        return self._client

    # ------------------------------------------------------------------
    # Resource discovery
    # ------------------------------------------------------------------

    async def discover_resources(self) -> list[dict[str, Any]]:
        """Return the list of available compute resources (IDs, names, etc.)."""
        client = await self._ensure_client()
        resp = await client.get(f"{self.base_url}/status")
        resp.raise_for_status()
        data = resp.json()
        # IRI v1.2 returns a list under the top-level key, or the list itself
        if isinstance(data, list):
            return data
        return data.get("items", [data])

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    async def mkdir(self, remote_path: str) -> None:
        """Create a directory on the NERSC filesystem (no-op if it exists)."""
        client = await self._ensure_client()
        resp = await client.put(
            f"{self.base_url}/utilities/command/{self.resource_id}",
            json={"command": f"mkdir -p {remote_path}"},
        )
        resp.raise_for_status()

    async def upload(self, remote_path: str, content: str) -> None:
        """Upload a text file to *remote_path* on the NERSC filesystem.

        Uses the IRI ``/utilities/upload`` endpoint (max 5 MB).  The
        containing directory must already exist (call ``mkdir`` first).
        """
        client = await self._ensure_client()
        # IRI upload is a multipart form — we need to send raw bytes.
        upload_client = httpx.AsyncClient(
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Accept": "application/json",
            },
        )
        async with upload_client:
            resp = await upload_client.post(
                f"{self.base_url}/utilities/upload/{self.resource_id}",
                data={"target_path": remote_path},
                files={"file": ("inference.py", content.encode(), "text/plain")},
            )
        resp.raise_for_status()

    # ------------------------------------------------------------------
    # Job submission / management
    # ------------------------------------------------------------------

    async def submit_job(
        self,
        script: str,
        working_dir: str,
        constraint: str = "gpu",
        account: str = "",
        time_limit: str = "00:30:00",
        nodes: int = 1,
        tasks_per_node: int = 1,
        environment: dict[str, str] | None = None,
    ) -> str:
        """Submit a job to NERSC via the IRI compute API.  Returns the task ID."""
        client = await self._ensure_client()
        payload: dict[str, Any] = {
            "script": script,
            "isPath": False,
            "constraint": constraint,
            "timelimit": time_limit,
            "nnodes": nodes,
            "tasks_per_node": tasks_per_node,
        }
        if account:
            payload["account"] = account
        if environment:
            payload["env_vars"] = environment

        resp = await client.post(
            f"{self.base_url}/compute/jobs/{self.resource_id}",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        # IRI wraps the Slurm job_id in a task; return the task_id for polling.
        return str(data.get("task_id", data.get("jobid", "")))

    async def get_task(self, task_id: str) -> dict[str, Any]:
        """Return the raw task dict (task status + optional Slurm job info)."""
        client = await self._ensure_client()
        resp = await client.get(f"{self.base_url}/tasks/{task_id}")
        resp.raise_for_status()
        return resp.json()

    async def get_job(self, task_id: str) -> NERSCJob:
        """Return a ``NERSCJob`` for the given task ID."""
        data = await self.get_task(task_id)
        # IRI task status is under "status"; Slurm state may be in "result"
        status_raw = data.get("status", "")
        result = data.get("result") or {}
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                result = {}
        slurm_state = ""
        if isinstance(result, dict):
            slurm_state = result.get("status", result.get("state", ""))
        # Prefer the Slurm state if available and mapped
        state = slurm_state if slurm_state in _NERSC_STATE_MAP else status_raw
        return NERSCJob(job_id=task_id, state=state, raw=data)

    async def cancel_job(self, task_id: str) -> None:
        """Cancel a running job (deletes the Slurm job)."""
        client = await self._ensure_client()
        task_data = await self.get_task(task_id)
        result = task_data.get("result") or {}
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                result = {}
        slurm_id = result.get("jobid", result.get("job_id", "")) if isinstance(result, dict) else ""
        if slurm_id:
            resp = await client.delete(
                f"{self.base_url}/compute/jobs/{self.resource_id}/{slurm_id}"
            )
            resp.raise_for_status()

    async def wait_for_job(
        self,
        task_id: str,
        poll_interval: float = 10.0,
        timeout: float = 1800.0,
    ) -> NERSCJob:
        """Poll until the job reaches a terminal state."""
        deadline = time.monotonic() + timeout
        info: NERSCJob | None = None
        while time.monotonic() < deadline:
            info = await self.get_job(task_id)
            if _NERSC_STATE_MAP.get(info.state) in (JobStatus.completed, JobStatus.failed):
                return info
            await asyncio.sleep(poll_interval)
        raise TimeoutError(
            f"NERSC job {task_id} did not complete within {timeout}s "
            f"(last state: {info.state if info else 'unknown'})"
        )


def _build_nersc_script(
    working_dir: str,
    script_path: str,
    container_image: str,
    job_name: str = "emblase",
    time_limit: str = "00:30:00",
    constraint: str = "gpu",
    account: str = "",
    nodes: int = 1,
    ntasks: int = 1,
) -> str:
    """Build the NERSC Slurm batch script that runs the inference script in Shifter.

    The rendered Python inference/streaming script is already uploaded to
    *script_path* on the NERSC filesystem before this is submitted.
    """
    account_line = f"#SBATCH --account={account}" if account else ""
    return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time_limit}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --constraint={constraint}
#SBATCH --gpus=1
#SBATCH --output={working_dir}/slurm-%j.out
#SBATCH --error={working_dir}/slurm-%j.out
{account_line}

set -euo pipefail
set -x

JOB_DIR={working_dir}/job_${{SLURM_JOB_ID}}
export JOB_DIR
mkdir -p "$JOB_DIR"

module load shifter

shifter --image={container_image} python {script_path}
"""


class NERSCBackend(ComputeBackend):
    """Submit inference jobs to NERSC (Perlmutter) via the IRI REST API + Shifter.

    The rendered Python inference script is uploaded to
    ``{working_dir}/scripts/<timestamp>/inference.py`` on the NERSC
    filesystem, then executed inside a Shifter container with the emblase
    image.  All secrets (Tiled URI, MLflow URI, API keys) are injected as
    environment variables into the Slurm job.
    """

    def __init__(
        self,
        client: NERSCClient | None = None,
        working_dir: str | None = None,
        models_dir: str | None = None,
        account: str | None = None,
        container_image: str | None = None,
        time_limit: str | None = None,
        constraint: str | None = None,
    ):
        self.client = client or NERSCClient()
        self.working_dir = working_dir or settings.nersc_working_dir
        self.models_dir = models_dir or settings.nersc_models_dir
        self.account = account or settings.nersc_account
        self.container_image = container_image or settings.nersc_container_image
        self.time_limit = time_limit or settings.nersc_time_limit
        self.constraint = constraint or settings.nersc_constraint
        self._jobs: dict[str, dict[str, Any]] = {}

    def _build_environment(self, *, require_tiled: bool = False) -> dict[str, str]:
        """Build the environment variable dict for a NERSC job."""
        if require_tiled and not settings.tiled_server_uri:
            raise ValueError(
                "EMBLASE_TILED_SERVER_URI is not set — required for this job type."
            )
        env: dict[str, str] = {}
        if settings.tiled_server_uri:
            env["EMBLASE_TILED_SERVER_URI"] = settings.tiled_server_uri
            if settings.tiled_api_key:
                env["EMBLASE_TILED_API_KEY"] = settings.tiled_api_key
            if settings.tiled_access_tags:
                env["EMBLASE_TILED_ACCESS_TAGS"] = settings.tiled_access_tags
        if settings.mlflow_tracking_uri:
            env["EMBLASE_MLFLOW_TRACKING_URI"] = settings.mlflow_tracking_uri
            if settings.mlflow_api_key:
                env["EMBLASE_MLFLOW_API_KEY"] = settings.mlflow_api_key
            if settings.model_cache_dir:
                env["EMBLASE_MODEL_CACHE_DIR"] = settings.model_cache_dir
        return env

    async def _upload_script(self, py_script: str, label: str) -> str:
        """Upload *py_script* to NERSC and return its remote path."""
        ts = int(time.time() * 1000)
        script_dir = f"{self.working_dir}/scripts/{ts}_{label}"
        script_path = f"{script_dir}/inference.py"
        await self.client.mkdir(script_dir)
        await self.client.upload(script_path, py_script)
        return script_path

    async def submit(
        self,
        model_name: str,
        batch_size: int = 1,
        run_path: str = "",
        image_key: str = "pil900KW_image",
        inputs: list[str | tuple[str, str]] | None = None,
        output: str = "",
        mlflow_version: str = "",
        thumb_mode: str = "default",
        param_specs: dict | None = None,
        projector: str | None = None,
        classifier: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit a batch inference job to NERSC.

        Image source — at least one of ``run_path``, ``inputs``, or neither
        (dummy images generated on the node) must be provided.  Unlike the
        Orion backend, direct numpy array upload is not supported — use Tiled
        or write a .npy to ``$PSCRATCH`` and pass via ``run_path``.

        See :meth:`OrionBackend.submit` for parameter documentation.
        """
        py_script = _render_inference_script(
            model_name=model_name,
            models_dir=self.models_dir,
            batch_size=batch_size,
            run_path=run_path,
            image_key=image_key,
            inputs=inputs,
            output=output,
            mlflow_version=mlflow_version,
            thumb_mode=thumb_mode,
            param_specs=param_specs,
            projector=projector,
            classifier=classifier,
        )

        script_path = await self._upload_script(py_script, model_name)
        sbatch = _build_nersc_script(
            working_dir=self.working_dir,
            script_path=script_path,
            container_image=self.container_image,
            job_name=f"emblase-{model_name}",
            time_limit=self.time_limit,
            constraint=self.constraint,
            account=self.account,
        )

        environment = self._build_environment(require_tiled=bool(run_path or inputs or output))
        task_id = await self.client.submit_job(
            script=sbatch,
            working_dir=self.working_dir,
            constraint=self.constraint,
            account=self.account,
            time_limit=self.time_limit,
            environment=environment,
        )

        self._jobs[task_id] = {
            "model_name": model_name,
            "output": output,
            "script_path": script_path,
        }
        return task_id

    async def submit_streaming(
        self,
        run_path: str,
        output: str,
        model_name: str,
        batch_size: int = 8,
        mlflow_version: str = "",
        thumb_mode: str = "logroi",
        image_key: str = "pil900KW_image",
        ws_max_size: int = 64 * 1024 * 1024,
        param_specs: dict | None = None,
        projector: str | None = None,
        classifier: str | None = None,
        time_limit: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit a streaming inference job to NERSC (full parity with Orion).

        See :meth:`OrionBackend.submit_streaming` for parameter documentation.
        """
        py_script = _render_streaming_inference_script(
            model_name=model_name,
            models_dir=self.models_dir,
            run_path=run_path,
            output=output,
            batch_size=batch_size,
            mlflow_version=mlflow_version,
            thumb_mode=thumb_mode,
            image_key=image_key,
            ws_max_size=ws_max_size,
            param_specs=param_specs,
            projector=projector,
            classifier=classifier,
        )

        script_path = await self._upload_script(py_script, f"stream-{model_name}")
        sbatch = _build_nersc_script(
            working_dir=self.working_dir,
            script_path=script_path,
            container_image=self.container_image,
            job_name=f"emblase-stream-{model_name}",
            time_limit=time_limit or "02:00:00",
            constraint=self.constraint,
            account=self.account,
        )

        environment = self._build_environment(require_tiled=True)
        task_id = await self.client.submit_job(
            script=sbatch,
            working_dir=self.working_dir,
            constraint=self.constraint,
            account=self.account,
            time_limit=time_limit or "02:00:00",
            environment=environment,
        )

        self._jobs[task_id] = {
            "model_name": model_name,
            "output": output,
            "script_path": script_path,
        }
        return task_id

    async def status(self, job_id: str) -> JobStatus:
        info = await self.client.get_job(job_id)
        return _NERSC_STATE_MAP.get(info.state, JobStatus.pending)

    async def result(self, job_id: str) -> JobResult:
        st = await self.status(job_id)
        meta = self._jobs.get(job_id)
        if not meta:
            return JobResult(job_id=job_id, status=JobStatus.failed, error="Job metadata lost")

        if st not in (JobStatus.completed, JobStatus.failed):
            return JobResult(job_id=job_id, status=st)

        if meta.get("output"):
            return JobResult(job_id=job_id, status=st)

        return JobResult(
            job_id=job_id,
            status=st,
            error="No local output path available — results written to Tiled or PSCRATCH"
            if st == JobStatus.completed
            else None,
        )

    async def wait(
        self,
        job_id: str,
        poll_interval: float = 10.0,
        timeout: float = 1800.0,
    ) -> JobStatus:
        """Poll until the job reaches a terminal state. Returns final JobStatus."""
        info = await self.client.wait_for_job(job_id, poll_interval=poll_interval, timeout=timeout)
        return _NERSC_STATE_MAP.get(info.state, JobStatus.failed)

    async def cancel(self, job_id: str) -> None:
        await self.client.cancel_job(job_id)
