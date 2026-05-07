"""NERSC compute backend for inference jobs via the IRI API + Shifter containers.

Auth:
    A Globus bearer token scoped to the IRI API is required.  Set
    ``EMBLASE_NERSC_API_TOKEN`` in your ``.env`` file.  The token must have
    the ``iri_api`` Globus scope — obtain it from
    https://iris.nersc.gov → Superfacility API → Get API Token.

API:
    Base URL: ``https://api.iri.nersc.gov/api/v1``
    Spec:     https://api.iri.nersc.gov/openapi.json  (Swagger UI at /docs)

    Endpoints used:

    ============================================  ======  ===================================================
    Endpoint                                      Method  Purpose
    ============================================  ======  ===================================================
    ``/status/resources``                         GET     List available resource IDs (``resources`` command)
    ``/compute/job/{resource_id}``                POST    Submit a structured JobSpec
    ``/compute/status/{resource_id}/{job_id}``    GET     Poll job state (``monitor_job`` loop)
    ``/compute/cancel/{resource_id}/{job_id}``    POST    Cancel a running job
    ============================================  ======  ===================================================

    Note: filesystem endpoints (``/filesystem/mkdir``, ``/filesystem/upload``)
    are intentionally not used — the inference script is embedded directly in
    the job payload via a ``pre_launch`` heredoc.

Job delivery:
    The rendered Python inference script is embedded directly in the job
    submission payload via a ``pre_launch`` heredoc — no filesystem API calls
    are needed.  A structured ``JobSpec`` runs the script inside a container::

        pre_launch: "cat > /path/to/inference.py << '__EMBLASE_EOF__' ..."
        container:  {image: <container_image>}
        executable: python
        arguments:  [/path/to/inference.py]

Resource discovery:
    Call ``await NERSCClient.discover_resources()`` to list available resource
    IDs (e.g. "perlmutter").  The default ``resource_id`` ("perlmutter") is
    configurable via ``EMBLASE_NERSC_RESOURCE_ID``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

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
    "new":       JobStatus.pending,
    "queued":    JobStatus.pending,
    "held":      JobStatus.pending,
    "active":    JobStatus.running,
    "completed": JobStatus.completed,
    "failed":    JobStatus.failed,
    "canceled":  JobStatus.failed,
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
                    "NERSC API token is not set.  "
                    "Set EMBLASE_NERSC_API_TOKEN in your .env file."
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
        import asyncio as _asyncio
        client = await self._ensure_client()
        deadline = _asyncio.get_event_loop().time() + timeout
        while True:
            resp = await client.get(f"{self.base_url}/task/{task_id}")
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "pending")
            if status not in ("pending", "active", "running"):
                result = data.get("result") or {}
                if status == "failed" or "error" in result:
                    raise RuntimeError(
                        result.get("error") or f"Task {task_id} failed: {result}"
                    )
                return result
            if _asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            await _asyncio.sleep(2.0)

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
        result = await self._poll_task(task_id)
        return result.get("output", "") or result.get("content", "") or str(result)

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
        constraint: str = "gpu",
        environment: dict[str, str] | None = None,
        pre_launch: str = "",
        stdout_path: str = "",
        stderr_path: str = "",
    ) -> str:
        """Submit a job via the IRI structured JobSpec.  Returns the job ID."""
        client = await self._ensure_client()

        attributes: dict[str, Any] = {
            "duration": time_limit_s,
            "custom_attributes": {"constraint": constraint},
        }
        if account:
            attributes["account"] = account

        resources: dict[str, Any] = {
            "node_count": nodes,
        }
        if gpus_per_process >= 1:
            resources["gpu_cores_per_process"] = gpus_per_process

        payload: dict[str, Any] = {
            "name": name,
            "executable": executable,
            "arguments": arguments,
            "directory": working_dir,
            "container": {"image": container_image},
            "resources": resources,
            "attributes": attributes,
        }
        if environment:
            payload["environment"] = environment
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
        resp = await client.get(
            f"{self.base_url}/compute/status/{self.resource_id}/{job_id}"
        )
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
        resp = await client.delete(
            f"{self.base_url}/compute/cancel/{self.resource_id}/{job_id}"
        )
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

    The rendered Python inference script is embedded in the job submission
    payload via a ``pre_launch`` heredoc and written to
    ``{working_dir}/scripts/<timestamp>/inference.py`` on the compute node
    at job start.  The job then runs the script inside a container image.

    All secrets (Tiled URI, MLflow URI, API keys) are injected as
    ``environment`` in the job spec.
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

    def _script_pre_launch(self, py_script: str, script_path: str) -> str:
        """Return a pre_launch shell snippet that writes *py_script* to *script_path*.

        This avoids any filesystem API calls — the script content is embedded
        directly in the job submission payload via a heredoc.
        """
        # Escape any occurrence of the heredoc delimiter inside the script.
        safe = py_script.replace("__EMBLASE_EOF__", "__EMBLASE_EOF_ESC__")
        return (
            f"mkdir -p {self.working_dir}/scripts\n"
            f"cat > {script_path} << '__EMBLASE_EOF__'\n"
            f"{safe}\n"
            f"__EMBLASE_EOF__"
        )

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

        ts = int(time.time() * 1000)
        script_path = f"{self.working_dir}/scripts/{ts}_{model_name}/inference.py"
        log_path = f"{self.working_dir}/scripts/{ts}_{model_name}/job.out"
        pre_launch = self._script_pre_launch(py_script, script_path)
        environment = self._build_environment(require_tiled=bool(run_path or inputs or output))
        job_id = await self.client.submit_job(
            executable="python",
            arguments=[script_path],
            working_dir=self.working_dir,
            container_image=self.container_image,
            name=f"emblase-{model_name}",
            account=self.account,
            time_limit_s=self._parse_time_limit(self.time_limit),
            constraint=self.constraint,
            environment=environment,
            pre_launch=pre_launch,
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

        ts = int(time.time() * 1000)
        script_path = f"{self.working_dir}/scripts/{ts}_stream-{model_name}/inference.py"
        log_path = f"{self.working_dir}/scripts/{ts}_stream-{model_name}/job.out"
        pre_launch = self._script_pre_launch(py_script, script_path)
        environment = self._build_environment(require_tiled=True)
        effective_limit = time_limit or "02:00:00"
        job_id = await self.client.submit_job(
            executable="python",
            arguments=[script_path],
            working_dir=self.working_dir,
            container_image=self.container_image,
            name=f"emblase-stream-{model_name}",
            account=self.account,
            time_limit_s=self._parse_time_limit(effective_limit),
            constraint=self.constraint,
            environment=environment,
            pre_launch=pre_launch,
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
