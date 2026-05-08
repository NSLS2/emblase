"""NERSC compute backend for inference jobs via the IRI API + podman containers.

Auth:
    A Globus bearer token scoped to the IRI API is required.  Set
    ``EMBLASE_NERSC_API_TOKEN`` in your ``.env`` file.  The token must have
    the ``iri_api`` Globus scope — obtain it from
    https://iris.nersc.gov → Superfacility API → Get API Token.

API:
    Base URL: ``https://api.iri.nersc.gov/api/v1``
    Spec:     https://api.iri.nersc.gov/openapi.json  (Swagger UI at /docs)

    Endpoints used:

    =============================================  ======  ====================================================
    Endpoint                                       Method  Purpose
    =============================================  ======  ====================================================
    ``/status/resources``                          GET     List available resource IDs (``resources`` command)
    ``/compute/job/{resource_id}``                 POST    Submit a structured JobSpec
    ``/compute/status/{resource_id}/{job_id}``     GET     Poll job state (``monitor_job`` loop)
    ``/compute/cancel/{resource_id}/{job_id}``     DELETE  Cancel a running job
    ``/filesystem/ls/{resource_id}``               GET     List a directory (``ls`` command)
    ``/filesystem/mkdir/{resource_id}``            POST    Create directory on ``/pscratch``
    ``/filesystem/upload/{resource_id}``           POST    Upload a file (≤ 5 MB) to any filesystem
    ``/filesystem/download/{resource_id}``         GET     Download a file (≤ 5 MB) from ``/pscratch``
    ``/filesystem/chmod/{resource_id}``            PUT     Set file permissions
    ``/filesystem/tail/{resource_id}``             GET     Read last N lines of a file (job logs)
    ``/task/{task_id}``                            GET     Poll an async filesystem task to completion
    =============================================  ======  ====================================================

    Known resource IDs on Perlmutter (from ``GET /status/resources``):

    ============  =====================================================
    Resource ID   Filesystem / purpose
    ============  =====================================================
    ``scratch``   ``/pscratch`` — fast Lustre scratch
    ``homes``     ``/global/u2/...`` — user home dirs
    ``cfs``       ``/global/cfs/...`` — community/project storage
    ``archive``   HPSS tape archive
    ``compute``   Perlmutter compute (jobs)
    ============  =====================================================

    Home directory paths follow the pattern ``/global/u2/<initial>/<username>``
    (e.g. ``/global/u2/j/jdoe``).  Use ``ls /global/u2/<i>/<user> --resource homes``
    to confirm your path.

Job delivery:
    The rendered Python inference script is uploaded to ``/pscratch`` via the
    IRI filesystem API before the job is submitted::

        POST /filesystem/mkdir/scratch  {"path": "<script_dir>", "parent": true}
        POST /filesystem/upload/scratch?path=<script_path>   (multipart, ≤ 5 MB)
        PUT  /filesystem/chmod/scratch   {"path": "<script_path>", "mode": "400"}

    The job then runs inside a podman-hpc container::

        executable: python
        arguments:  [<script_path>]

    ``/pscratch`` is bind-mounted at the same path inside the container so the
    script is visible.  Only one volume mount is supported by podman-hpc on
    Perlmutter — two mounts produce an ``invalid reference format`` error.

Secrets injection:
    All secrets (``EMBLASE_TILED_*``, ``EMBLASE_MLFLOW_*``) are baked at
    submit time into a short Python preamble that is prepended to the rendered
    inference script::

        import os as _os
        _os.environ.setdefault('EMBLASE_MLFLOW_TRACKING_URI', '...')
        ...

    The script is then uploaded to ``/pscratch`` and ``chmod 400`` (owner
    read-only).  The secrets exist on disk only for the duration of the job;
    once the job completes the file stays on ``/pscratch`` but is readable only
    by the submitting user (standard Lustre POSIX permissions).

    Approaches investigated and rejected:

    * ``$HOME`` bind-mount — ``$HOME`` inside the container is ``/root``,
      not the user's Perlmutter home; the home filesystem is not mounted.
    * Two volume mounts (``/pscratch`` + ``/global/u2/...``) — podman-hpc
      treats the second entry as a container image name and fails with
      ``invalid reference format``.
    * IRI ``container.env`` field — silently ignored by podman-hpc
      (confirmed: probe job 52700081, zero vars reached the container).
    * IRI top-level ``environment`` field — also silently ignored inside the
      container (confirmed: probe job 52708889).
    * Bare-metal execution (no container) — rejected by Slurm policy on
      Perlmutter GPU nodes; ``sbatch`` returns "does not match any supported
      policy" without a container image.
    * ``pre_launch`` heredoc — silently ignored for podman container jobs.
    * ``bash -c "<cmd>"`` in arguments — ``-c`` is intercepted by podman as
      ``--cpu-shares`` before reaching bash.

Perlmutter queues:
    The ``queue_name`` field in the IRI JobSpec maps to Slurm partition names
    (not QOS names — QOS names cause a 400 error).  Valid values:

    ==================  ====================  ================  ================================
    queue_name          Slurm partition       Slurm QOS         Notes
    ==================  ====================  ================  ================================
    ``"shared"``        ``shared_gpu_ss11``   gpu_shared        **Best for single-GPU jobs**
                                                                — shares nodes, lowest wait
    ``"debug"``         ``gpu_ss11``          gpu_debug         Fast dispatch, ≤ 30 min cap
    ``""``              ``gpu_ss11``          gpu_debug         Scheduler default (= debug)
    ``"regular"``       ``gpu_ss11``          gpu_regular       Standard, longer queue
    ``"premium"``       ``gpu_ss11``          gpu_premium       Faster turn-around, higher cost;
                                                                requires account ``amsc006_g``.
    ``"express_amsc_g"``  ``gpu_ss11``        express_amsc_g    Real-time GPU access; pool of 32
                                                                reserved nodes shared by AMSC
                                                                projects.  Requires account
                                                                ``amsc006_g``.  Avoid >4 nodes
                                                                or >2 h per job.
    ``"express_amsc"``  ``cpu_ss11``          express_amsc      Same reserved pool, CPU nodes.
                                                                Requires account ``amsc006``.
                                                                Same usage limits apply.
    ==================  ====================  ================  ================================

    Default: ``"shared"`` (set via ``EMBLASE_NERSC_QUEUE``).

    Account note: GPU jobs on Perlmutter require the ``_g`` suffix on the
    project code (e.g. ``m3792_g``, not ``m3792``).  Set
    ``EMBLASE_NERSC_ACCOUNT=<project>_g`` in your ``.env``.

Resource discovery:
    Call ``await NERSCClient.discover_resources()`` to list available resource
    IDs.  The default compute ``resource_id`` (``"perlmutter"``) is
    configurable via ``EMBLASE_NERSC_RESOURCE_ID``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from ..config import settings
from .base import ComputeBackend, JobResult, JobStatus
from .orion import (
    _render_inference_script,
    _render_streaming_inference_script,
)

_IRI_BASE = "https://api.iri.nersc.gov/api/v1"
logger = logging.getLogger(__name__)
_TERMINAL_STATES = {JobStatus.completed, JobStatus.failed}


def _iri_base() -> str:
    """Return the IRI API base URL from settings, ensuring /api/v1 suffix."""
    uri = (settings.nersc_api_uri or _IRI_BASE).rstrip("/")
    if not uri.endswith("/api/v1"):
        uri = f"{uri}/api/v1"
    return uri


# IRI JobState enum values → our JobStatus
_NERSC_STATE_MAP: dict[str, JobStatus] = {
    "new": JobStatus.pending,
    "queued": JobStatus.pending,
    "held": JobStatus.pending,
    "active": JobStatus.running,
    "completed": JobStatus.completed,
    "failed": JobStatus.failed,
    "canceled": JobStatus.failed,
}


@dataclass
class NERSCJob:
    """Parsed NERSC IRI job info."""

    job_id: str
    state: str
    raw: dict[str, Any] | None = None


class NERSCClient:
    """Async HTTP client for the NERSC IRI REST API (api.iri.nersc.gov/api/v1).

    Usage::

        async with NERSCClient() as client:
            resources = await client.discover_resources()
            job_id = await client.submit_job(...)
            info = await client.get_job(job_id)
    """

    def __init__(
        self,
        api_token: str | None = None,
        resource_id: str | None = None,
        base_url: str | None = None,
    ):
        self.api_token = api_token or settings.nersc_api_token
        self.resource_id = resource_id or settings.nersc_resource_id
        self.base_url = (base_url or _iri_base()).rstrip("/")
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
                    "NERSC API token is not set.  Set EMBLASE_NERSC_API_TOKEN in your .env file."
                )
            self._client = httpx.AsyncClient(
                timeout=60.0,
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Accept": "application/json",
                },
            )
        return self._client

    # ------------------------------------------------------------------
    # Resource discovery
    # ------------------------------------------------------------------

    async def discover_resources(self) -> list[dict[str, Any]]:
        """Return the list of available compute resources."""
        client = await self._ensure_client()
        resp = await client.get(f"{self.base_url}/status/resources")
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        # paginated response has items under a key
        return data.get("items", [data])

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    async def _poll_task(self, task_id: str, timeout: float = 60.0) -> dict[str, Any]:
        """Poll /task/{task_id} until complete and return the result dict.

        Raises ``RuntimeError`` if the task fails.
        """
        client = await self._ensure_client()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            resp = await client.get(f"{self.base_url}/task/{task_id}")
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "pending")
            if status not in ("pending", "active", "running"):
                result = data.get("result") or {}
                if status == "failed" or "error" in result:
                    raise RuntimeError(result.get("error") or f"Task {task_id} failed: {result}")
                return result
            if loop.time() > deadline:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            await asyncio.sleep(2.0)

    async def ls(
        self,
        remote_path: str,
        filesystem_resource_id: str = "homes",
        show_hidden: bool = True,
    ) -> list[dict]:
        """List the contents of *remote_path* and return the entries as a list of dicts."""
        client = await self._ensure_client()
        resp = await client.get(
            f"{self.base_url}/filesystem/ls/{filesystem_resource_id}",
            params={"path": remote_path, "showHidden": show_hidden},
        )
        resp.raise_for_status()
        task_id = resp.json()["task_id"]
        result = await self._poll_task(task_id)
        # Result may be under "entries", "files", "output", or raw list
        if isinstance(result, list):
            return result
        for key in ("entries", "files", "output", "content"):
            if key in result:
                v = result[key]
                if isinstance(v, list):
                    return v
                if isinstance(v, str):
                    return [{"name": line} for line in v.splitlines() if line]
        return [result]

    async def read_file_tail(
        self,
        remote_path: str,
        lines: int = 100,
        filesystem_resource_id: str = "scratch",
    ) -> str:
        """Return the last *lines* lines of *remote_path* via the IRI filesystem API."""
        client = await self._ensure_client()
        resp = await client.get(
            f"{self.base_url}/filesystem/tail/{filesystem_resource_id}",
            params={"path": remote_path, "lines": lines},
        )
        resp.raise_for_status()
        task_id = resp.json()["task_id"]
        result = await self._poll_task(task_id, timeout=120.0)
        out = result.get("output", "")
        if isinstance(out, dict):
            out = out.get("content", "") or out.get("output", "") or str(out)
        return out or result.get("content", "") or str(result)

    async def upload_script(
        self,
        content: str,
        remote_path: str,
        filesystem_resource_id: str = "scratch",
        mode: str = "400",
    ) -> None:
        """Upload *content* as a text file to *remote_path* on the remote filesystem.

        Creates all parent directories first (``mkdir -p`` semantics).
        Uses the IRI filesystem API (max 5 MB per file).

        The file is chmod'd to *mode* (default ``400``, owner read-only) after
        upload.  Scripts contain credentials baked into the preamble, so
        read-only by owner is the tightest practical permission on Lustre.
        """
        client = await self._ensure_client()
        remote_dir = remote_path.rsplit("/", 1)[0]

        # mkdir -p
        resp = await client.post(
            f"{self.base_url}/filesystem/mkdir/{filesystem_resource_id}",
            json={"path": remote_dir, "parent": True},
        )
        resp.raise_for_status()
        await self._poll_task(resp.json()["task_id"])

        # upload
        resp = await client.post(
            f"{self.base_url}/filesystem/upload/{filesystem_resource_id}",
            params={"path": remote_path},
            files={"file": (remote_path.rsplit("/", 1)[-1], content.encode(), "text/plain")},
        )
        resp.raise_for_status()
        await self._poll_task(resp.json()["task_id"])

        # chmod — owner read-only so credentials in the preamble are not world-readable
        resp = await client.put(
            f"{self.base_url}/filesystem/chmod/{filesystem_resource_id}",
            json={"path": remote_path, "mode": mode},
        )
        resp.raise_for_status()
        await self._poll_task(resp.json()["task_id"])

    async def download_file(
        self,
        remote_path: str,
        filesystem_resource_id: str = "scratch",
    ) -> bytes:
        """Download a small file (≤ 5 MB) from *remote_path* and return its bytes.

        Uses ``GET /filesystem/download/{resource_id}?path=<path>``.
        The response follows the same task-polling pattern as upload.
        """
        client = await self._ensure_client()
        resp = await client.get(
            f"{self.base_url}/filesystem/download/{filesystem_resource_id}",
            params={"path": remote_path},
        )
        resp.raise_for_status()
        task_id = resp.json()["task_id"]
        result = await self._poll_task(task_id)
        # The API returns the file content base64-encoded under "file" or raw under "content"
        if "file" in result:
            return base64.b64decode(result["file"])
        raw = result.get("content") or result.get("output") or ""
        return raw.encode() if isinstance(raw, str) else raw

    # ------------------------------------------------------------------
    # Job submission / management
    # ------------------------------------------------------------------

    async def submit_job(
        self,
        executable: str,
        arguments: list[str],
        working_dir: str,
        container_image: str,
        name: str = "emblase",
        account: str = "",
        time_limit_s: int = 1800,
        nodes: int = 1,
        gpus_per_process: int = 1,
        constraint: str = "",
        queue_name: str = "",
        pre_launch: str = "",
        stdout_path: str = "",
        stderr_path: str = "",
        volume_mounts: list[dict] | None = None,
    ) -> str:
        """Submit a job via the IRI structured JobSpec.  Returns the job ID."""
        client = await self._ensure_client()

        attributes: dict[str, Any] = {
            "duration": time_limit_s,
        }
        custom: dict[str, str] = {}
        if constraint:
            custom["constraint"] = constraint
        if queue_name:
            attributes["queue_name"] = queue_name
        if custom:
            attributes["custom_attributes"] = custom
        if account:
            attributes["account"] = account

        resources: dict[str, Any] = {
            "node_count": nodes,
        }
        if gpus_per_process >= 1:
            resources["gpu_cores_per_process"] = gpus_per_process

        # Only /pscratch is mounted — two volume mounts cause podman-hpc to
        # fail with "invalid reference format" (treats the second entry as an
        # image name).  Scripts and model weights both live on /pscratch.
        if volume_mounts is None:
            volume_mounts = [
                {"source": "/pscratch", "target": "/pscratch", "read_only": False},
            ]

        payload: dict[str, Any] = {
            "name": name,
            "executable": executable,
            "arguments": arguments,
            "directory": working_dir,
            "container": {
                "image": container_image,
                "volume_mounts": volume_mounts,
            },
            "resources": resources,
            "attributes": attributes,
        }
        if pre_launch:
            payload["pre_launch"] = pre_launch
        if stdout_path:
            payload["stdout_path"] = stdout_path
        if stderr_path:
            payload["stderr_path"] = stderr_path

        resp = await client.post(
            f"{self.base_url}/compute/job/{self.resource_id}",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data["id"])

    async def get_job(self, job_id: str) -> NERSCJob:
        """Return a ``NERSCJob`` for *job_id*."""
        client = await self._ensure_client()
        resp = await client.get(f"{self.base_url}/compute/status/{self.resource_id}/{job_id}")
        resp.raise_for_status()
        data = resp.json()
        state = ""
        status_obj = data.get("status")
        if isinstance(status_obj, dict):
            state = status_obj.get("state", "")
        return NERSCJob(job_id=job_id, state=state, raw=data)

    async def cancel_job(self, job_id: str) -> None:
        """Cancel a running job."""
        client = await self._ensure_client()
        resp = await client.delete(f"{self.base_url}/compute/cancel/{self.resource_id}/{job_id}")
        resp.raise_for_status()

    async def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 15.0,
        timeout: float = 1800.0,
    ) -> NERSCJob:
        """Poll until the job reaches a terminal state."""
        deadline = time.monotonic() + timeout
        info: NERSCJob | None = None
        while time.monotonic() < deadline:
            info = await self.get_job(job_id)
            if _NERSC_STATE_MAP.get(info.state) in _TERMINAL_STATES:
                return info
            await asyncio.sleep(poll_interval)
        raise TimeoutError(
            f"NERSC job {job_id} did not complete within {timeout}s "
            f"(last state: {info.state if info else 'unknown'})"
        )


class NERSCBackend(ComputeBackend):
    """Submit inference jobs to NERSC (Perlmutter) via the IRI REST API.

    Secrets (``EMBLASE_TILED_*``, ``EMBLASE_MLFLOW_*``) are baked at submit
    time into a short Python preamble prepended to the rendered inference
    script.  The script is uploaded to ``/pscratch`` and immediately
    ``chmod 400`` (owner read-only).  ``JOB_DIR`` (non-secret) is included
    in the same preamble.

    See the module docstring for a full account of secrets delivery approaches
    that were investigated and rejected.
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
        queue: str | None = None,
    ):
        self.client = client or NERSCClient()
        self.working_dir = working_dir or settings.nersc_working_dir
        self.models_dir = models_dir or str(settings.nersc_models_dir)
        self.account = account or settings.nersc_account
        self.container_image = container_image or settings.nersc_container_image
        self.time_limit = time_limit or settings.nersc_time_limit
        self.constraint = constraint or settings.nersc_constraint
        self.queue = queue if queue is not None else settings.nersc_queue
        self._jobs: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _parse_time_limit(time_limit: str) -> int:
        """Convert ``HH:MM:SS`` time limit string to seconds."""
        parts = time_limit.split(":")
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            h, m, s = 0, int(parts[0]), int(parts[1])
        else:
            return int(parts[0])
        return h * 3600 + m * 60 + s

    def _validate_environment(self, *, require_tiled: bool = False) -> None:
        """Raise early if required settings are missing."""
        if require_tiled and not settings.tiled_server_uri:
            raise ValueError("EMBLASE_TILED_SERVER_URI is not set — required for this job type.")

    def _secrets_preamble(self, extra_env: dict[str, str] | None = None) -> str:
        """Return a Python preamble that injects credentials into ``os.environ``.

        All values are baked in at submit time as ``os.environ.setdefault``
        calls.  The script is uploaded ``chmod 400`` (owner read-only) so that
        the credentials are not visible to other users on the Lustre filesystem.

        *extra_env* holds non-secret key/value pairs (e.g. ``JOB_DIR``) that
        are included in the same block for convenience.
        """
        env: dict[str, str] = {
            "EMBLASE_MLFLOW_TRACKING_URI": settings.mlflow_tracking_uri,
            "EMBLASE_MLFLOW_API_KEY": settings.mlflow_api_key,
            "EMBLASE_TILED_SERVER_URI": settings.tiled_server_uri,
            "EMBLASE_TILED_API_KEY": settings.tiled_api_key,
            "EMBLASE_NERSC_MODELS_DIR": str(settings.nersc_models_dir),
        }
        if settings.tiled_access_tags:
            env["EMBLASE_TILED_ACCESS_TAGS"] = settings.tiled_access_tags
        if extra_env:
            env.update(extra_env)
        lines = ["# --- emblase job environment ---", "import os as _os"]
        for k, v in env.items():
            if v:
                lines.append(f"_os.environ.setdefault({k!r}, {v!r})")
        lines.append("# --- end emblase job environment ---\n")
        return "\n".join(lines) + "\n"

    def _job_paths(self, slug: str) -> tuple[str, str, str]:
        """Return (script_path, log_path, script_dir) for a timestamped job slug."""
        ts = int(time.time() * 1000)
        base = f"{self.working_dir}/scripts/{ts}_{slug}"
        script_path = f"{base}/inference.py"
        return script_path, f"{base}/job.out", base

    async def _submit_script(
        self,
        py_script: str,
        script_path: str,
        log_path: str,
        script_dir: str,
        model_name: str,
        output: str,
        time_limit: str,
        job_name: str,
    ) -> str:
        """Upload *py_script*, submit the job, record metadata, and return the job ID."""
        preamble = self._secrets_preamble(extra_env={"JOB_DIR": script_dir})
        await self.client.upload_script(preamble + py_script, script_path)
        job_id = await self.client.submit_job(
            executable="python",
            arguments=[script_path],
            working_dir=self.working_dir,
            container_image=self.container_image,
            name=job_name,
            account=self.account,
            time_limit_s=self._parse_time_limit(time_limit),
            constraint=self.constraint,
            queue_name=self.queue,
            stdout_path=log_path,
            stderr_path=log_path,
        )
        self._jobs[job_id] = {
            "model_name": model_name,
            "output": output,
            "script_path": script_path,
            "log_path": log_path,
        }
        return job_id

    async def submit(
        self,
        model_name: str,
        batch_size: int = 1,
        run_path: str = "",
        image_key: str = "primary/pil900KW_image",
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

        See :meth:`OrionBackend.submit` for parameter documentation.
        """
        self._validate_environment(require_tiled=bool(run_path or inputs or output))
        script_path, log_path, script_dir = self._job_paths(model_name)
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
        job_id = await self._submit_script(
            py_script,
            script_path,
            log_path,
            script_dir,
            model_name=model_name,
            output=output,
            time_limit=self.time_limit,
            job_name=f"emblase-{model_name}",
        )
        logger.info(
            "Submitted batch job %s  model=%r  queue=%s  account=%s  limit=%s  log=%s",
            job_id,
            model_name,
            self.queue or "(default)",
            self.account,
            self.time_limit,
            log_path,
        )
        return job_id

    async def submit_streaming(
        self,
        run_path: str,
        output: str,
        model_name: str,
        batch_size: int = 8,
        mlflow_version: str = "",
        thumb_mode: str = "logroi",
        image_key: str = "primary/pil900KW_image",
        ws_max_size: int = 64 * 1024 * 1024,
        param_specs: dict | None = None,
        projector: str | None = None,
        classifier: str | None = None,
        time_limit: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit a streaming inference job to NERSC.

        See :meth:`OrionBackend.submit_streaming` for parameter documentation.
        """
        self._validate_environment(require_tiled=True)
        effective_limit = time_limit or "02:00:00"
        script_path, log_path, script_dir = self._job_paths(f"stream-{model_name}")
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
        job_id = await self._submit_script(
            py_script,
            script_path,
            log_path,
            script_dir,
            model_name=model_name,
            output=output,
            time_limit=effective_limit,
            job_name=f"emblase-stream-{model_name}",
        )
        logger.info(
            "Submitted streaming job %s  model=%r  queue=%s  account=%s  limit=%s  log=%s",
            job_id,
            model_name,
            self.queue or "(default)",
            self.account,
            effective_limit,
            log_path,
        )
        return job_id

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
            # Results written to Tiled — nothing to download
            return JobResult(job_id=job_id, status=st)

        # No Tiled output → try to download output.npy from the job dir on /pscratch
        script_path = meta.get("script_path", "")
        if script_path and st == JobStatus.completed:
            job_dir = script_path.rsplit("/", 1)[0]
            output_path = f"{job_dir}/output.npy"
            try:
                data = await self.client.download_file(output_path)
                arr = np.load(io.BytesIO(data))
                return JobResult(job_id=job_id, status=JobStatus.completed, output_data=arr)
            except Exception as exc:
                return JobResult(
                    job_id=job_id,
                    status=JobStatus.completed,
                    error=f"Job completed but output.npy download failed: {exc}",
                )

        return JobResult(job_id=job_id, status=st)

    async def wait(
        self,
        job_id: str,
        poll_interval: float = 15.0,
        timeout: float = 1800.0,
    ) -> JobStatus:
        """Poll until the job reaches a terminal state. Returns final JobStatus."""
        info = await self.client.wait_for_job(job_id, poll_interval=poll_interval, timeout=timeout)
        return _NERSC_STATE_MAP.get(info.state, JobStatus.failed)

    async def cancel(self, job_id: str) -> None:
        await self.client.cancel_job(job_id)

    def monitor_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 15.0,
        log_prefix: str = "",
        on_status: Any = None,
    ) -> None:
        """Block until *job_id* reaches a terminal state, logging state after each poll.

        Polls the NERSC IRI REST API every *poll_interval* seconds.  Returns
        once the job completes or fails.  No SSH used.

        Parameters
        ----------
        job_id:
            NERSC IRI job ID (string returned by ``submit()``).
        poll_interval:
            Seconds between API polls (default 15 s).
        log_prefix:
            Optional string prepended to every status line.
        on_status:
            Optional ``callable(state: str, node: None, elapsed_s: int)``
            invoked after each poll.
        """
        prefix = f"{log_prefix} " if log_prefix else ""
        t0 = time.monotonic()
        _loop = asyncio.new_event_loop()
        try:
            while True:
                time.sleep(poll_interval)
                try:
                    info = _loop.run_until_complete(self.client.get_job(job_id))
                except Exception as exc:
                    logger.warning("%sCould not poll job %s: %s", prefix, job_id, exc)
                    continue
                elapsed = int(time.monotonic() - t0)
                logger.info("%s[%4ds] state=%s", prefix, elapsed, info.state)
                if on_status:
                    on_status(info.state, None, elapsed)
                if _NERSC_STATE_MAP.get(info.state) in _TERMINAL_STATES:
                    logger.info("%sJob %s finished: %s", prefix, job_id, info.state)
                    break
        finally:
            _loop.close()
