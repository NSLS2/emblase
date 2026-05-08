"""Orion compute backend — Slurm job submission via the Orion REST API."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import httpx
import numpy as np

from ..config import settings
from .base import ComputeBackend, JobResult, JobStatus

logger = logging.getLogger(__name__)

_TEMPLATE = (Path(__file__).parent.parent / "worker" / "inference.py.tmpl").read_text()
_STREAMING_TEMPLATE = (
    Path(__file__).parent.parent / "worker" / "streaming_inference.py.tmpl"
).read_text()

_SLURM_STATE_MAP: dict[str, JobStatus] = {
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

_TERMINAL_STATES = {JobStatus.completed, JobStatus.failed}


# ---------------------------------------------------------------------------
# Script rendering
# ---------------------------------------------------------------------------


def _projector_mode_and_name(projector: str | None) -> tuple[str, str]:
    """Translate the ``--projector`` CLI value into ``(mode, name)``.

    ``None``                  → ``("scratch", "")`` — fit UMAP from scratch
    ``"false"``/``"0"``/etc.  → ``("false",  "")`` — write NaN, skip projector
    ``"<name>"``              → ``("name",   name)`` — resolve saved model by name
    """
    if projector is None:
        return "scratch", ""
    if projector.lower() in ("false", "0", "no", "none"):
        return "false", ""
    return "name", projector


def _render_worker_script(template: str, **kwargs: Any) -> str:
    """Render a worker template, converting projector/classifier kwargs to template slots."""
    projector_mode, projector_name = _projector_mode_and_name(kwargs.pop("projector", None))
    classifier = kwargs.pop("classifier", None)
    param_specs = kwargs.pop("param_specs", None)
    return template.format(
        **kwargs,
        param_specs_json=json.dumps(param_specs or {}),
        projector_mode=projector_mode,
        projector_name=projector_name,
        classifier_name=classifier or "",
    )


def _render_inference_script(
    model_name: str,
    models_dir: str,
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
) -> str:
    return _render_worker_script(
        _TEMPLATE,
        model_name=model_name,
        models_dir=models_dir,
        batch_size=batch_size,
        run_path=run_path,
        image_key=image_key,
        inputs=repr(inputs or []),
        output=output,
        mlflow_version=mlflow_version,
        thumb_mode=thumb_mode,
        param_specs=param_specs,
        projector=projector,
        classifier=classifier,
    )


def _render_streaming_inference_script(
    model_name: str,
    models_dir: str,
    run_path: str,
    output: str,
    batch_size: int = 8,
    mlflow_version: str = "",
    thumb_mode: str = "logroi",
    image_key: str = "primary/pil900KW_image",
    ws_max_size: int = 64 * 1024 * 1024,
    param_specs: dict | None = None,
    projector: str | None = None,
    classifier: str | None = None,
) -> str:
    return _render_worker_script(
        _STREAMING_TEMPLATE,
        model_name=model_name,
        models_dir=models_dir,
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


# ---------------------------------------------------------------------------
# sbatch script builder
# ---------------------------------------------------------------------------


def _build_sbatch_script(
    working_dir: str,
    python_script: str,
    project_dir: str,
    job_name: str = "emblase",
    time_limit: str = "0-00:10:00",
    mem: str = "16G",
    payload_b64: str | None = None,
    npy_path: str | None = None,
    tiled_input: bool = False,
) -> str:
    """Build a self-contained sbatch script.

    Exactly one image-source flag must be set:
    - ``payload_b64``: base64-encoded numpy array embedded as a heredoc.
    - ``npy_path``: path to an ``.npy`` file already on the cluster (symlinked).
    - ``tiled_input``: inference script reads frames directly from Tiled (no pre-fetch).
    """
    if sum([bool(payload_b64), bool(npy_path), tiled_input]) != 1:
        raise ValueError("Provide exactly one of payload_b64, npy_path, tiled_input=True")

    if payload_b64:
        input_section = f"""\
cat > _input_b64.txt << 'EMBLASE_B64_EOF'
{payload_b64}
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
    elif npy_path:
        input_section = f'ln -sf {npy_path} "$JOB_DIR/input.npy"\n'
    else:
        input_section = ""

    return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem={mem}
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


# ---------------------------------------------------------------------------
# Orion REST client
# ---------------------------------------------------------------------------


@dataclass
class OrionJob:
    """Parsed response from the Orion jobs endpoint."""

    job_id: int
    state: str
    node: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    raw: dict[str, Any] | None = None


class OrionClient:
    """Async HTTP client for the Orion REST API.

    Use as an async context manager to ensure the underlying session is closed::

        async with OrionClient() as c:
            job_id = await c.submit_job(script="#!/bin/bash\\nhostname")
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        cluster: str | None = None,
    ) -> None:
        self.api_url = (api_url or settings.orion_api_url).rstrip("/")
        self.api_key = api_key or settings.orion_api_key
        self.cluster = cluster or settings.orion_cluster
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OrionClient:
        await self._open()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._close()

    async def _open(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            )
        return self._client

    async def _close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def submit_job(
        self,
        script: str,
        working_dir: str = "/tmp",
        overrides: dict[str, str] | None = None,
        environment: list[str] | None = None,
    ) -> int:
        """Submit a job script. Returns the Slurm job ID."""
        client = await self._open()
        payload: dict[str, Any] = {"script": script, "working_dir_path": working_dir}
        if overrides:
            payload["overrides"] = overrides
        if environment:
            payload["environment"] = environment
        resp = await client.post(f"{self.api_url}/api/v1/compute/{self.cluster}/jobs", json=payload)
        resp.raise_for_status()
        return resp.json()["job_id"]

    async def get_job(self, job_id: int) -> OrionJob:
        """Fetch info for a single job."""
        client = await self._open()
        resp = await client.get(f"{self.api_url}/api/v1/compute/{self.cluster}/jobs/{job_id}")
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
        client = await self._open()
        resp = await client.delete(f"{self.api_url}/api/v1/compute/{self.cluster}/jobs/{job_id}")
        resp.raise_for_status()

    async def wait_for_job(
        self,
        job_id: int,
        poll_interval: float = 5.0,
        timeout: float = 1800.0,
    ) -> OrionJob:
        """Poll until the job reaches a terminal state. Raises ``TimeoutError`` on timeout."""
        deadline = time.monotonic() + timeout
        info: OrionJob | None = None
        while time.monotonic() < deadline:
            info = await self.get_job(job_id)
            if _SLURM_STATE_MAP.get(info.state) in _TERMINAL_STATES:
                return info
            await asyncio.sleep(poll_interval)
        raise TimeoutError(
            f"Job {job_id} did not complete within {timeout}s "
            f"(last state: {info.state if info else 'unknown'})"
        )


# ---------------------------------------------------------------------------
# SSH log streaming
# ---------------------------------------------------------------------------


def _ssh_host() -> str:
    if settings.orion_ssh_host:
        return settings.orion_ssh_host
    return urllib.parse.urlparse(settings.orion_api_url).hostname or "orion-staging.nsls2.bnl.gov"


def _ssh_user() -> str:
    return settings.orion_ssh_user or os.environ.get("USER", "")


def _log_path(job_id: int) -> str:
    return f"{settings.orion_working_dir}/slurm-{job_id}.out"


def stream_logs(
    job_id: int,
    tail_n: int = 50,
    ssh_host: str | None = None,
    ssh_user: str | None = None,
) -> Generator[str, None, None]:
    """Stream the Slurm log for *job_id* from the Orion login node via SSH.

    Runs ``ssh <host> tail -f -n <tail_n> <log_file>`` and yields one
    stripped line at a time.  Blocks until the caller breaks or SSH exits.
    ``tail -f`` waits silently if the log file does not yet exist.
    """
    host = ssh_host or _ssh_host()
    user = ssh_user or _ssh_user()
    destination = f"{user}@{host}" if user else host
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        destination,
        f"tail -f -n {tail_n} -- {_log_path(job_id)}",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1
    )
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            yield line.rstrip("\n")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# OrionBackend
# ---------------------------------------------------------------------------


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
    ) -> None:
        self.client = client or OrionClient()
        self.working_dir = working_dir or settings.orion_working_dir
        self.models_dir = models_dir or str(settings.models_dir)
        self.project_dir = project_dir or settings.orion_project_dir
        self.home = home or settings.orion_home
        self.account = account or settings.orion_account
        self.path = path or settings.orion_path
        self._jobs: dict[int, dict[str, Any]] = {}

    # -- job monitoring (REST API only, no SSH) --------------------------------

    def monitor_job(
        self,
        job_id: str | int,
        *,
        poll_interval: float = 10.0,
        log_prefix: str = "",
        on_status: Any = None,
    ) -> None:
        """Block until *job_id* reaches a terminal state, logging state after each poll.

        Polls the Orion REST API every *poll_interval* seconds and logs the
        current state and node.  Returns once the job completes or fails.
        No SSH connection is made.

        Parameters
        ----------
        job_id:
            Slurm job ID (int or str).
        poll_interval:
            Seconds between API polls (default 10 s).
        log_prefix:
            Optional string prepended to every status line (e.g. ``"[run_xyz]"``).
        on_status:
            Optional ``callable(state: str, node: str | None, elapsed_s: int)``
            invoked after each poll.
        """
        job_id = int(job_id)
        prefix = f"{log_prefix} " if log_prefix else ""
        t0 = time.monotonic()
        # Create a dedicated client + event loop for this thread.
        # The backend's self.client was opened on the main asyncio loop and
        # cannot be reused here (httpx.AsyncClient is not cross-loop safe).
        _loop = asyncio.new_event_loop()
        _client = OrionClient()
        try:
            _loop.run_until_complete(_client._open())
            while True:
                time.sleep(poll_interval)
                try:
                    info = _loop.run_until_complete(_client.get_job(job_id))
                except Exception as exc:
                    logger.warning("%sCould not poll job %s: %s", prefix, job_id, exc)
                    continue
                elapsed = int(time.monotonic() - t0)
                logger.info(
                    "%s[%4ds] state=%s  node=%s",
                    prefix,
                    elapsed,
                    info.state,
                    info.node or "(queued)",
                )
                if on_status:
                    on_status(info.state, info.node, elapsed)
                if _SLURM_STATE_MAP.get(info.state) in _TERMINAL_STATES:
                    logger.info("%sJob %s finished: %s", prefix, job_id, info.state)
                    break
        finally:
            _loop.run_until_complete(_client._close())
            _loop.close()

    # -- internal helpers ----------------------------------------------------

    def _build_environment(self, *, require_tiled: bool = False) -> list[str]:
        if require_tiled and not settings.tiled_server_uri:
            raise ValueError("EMBLASE_TILED_SERVER_URI is not set — required for this job type.")
        env = [
            f"PATH={self.path}",
            f"HOME={self.home}",
            "SLURM_EXPORT_ENV=ALL",
            "TRANSFORMERS_OFFLINE=1",
            "HF_DATASETS_OFFLINE=1",
        ]
        if settings.tiled_server_uri:
            env.append(f"EMBLASE_TILED_SERVER_URI={settings.tiled_server_uri}")
            if settings.tiled_api_key:
                env.append(f"EMBLASE_TILED_API_KEY={settings.tiled_api_key}")
            if settings.tiled_access_tags:
                env.append(f"EMBLASE_TILED_ACCESS_TAGS={settings.tiled_access_tags}")
        if settings.mlflow_tracking_uri:
            env.append(f"EMBLASE_MLFLOW_TRACKING_URI={settings.mlflow_tracking_uri}")
            if settings.mlflow_api_key:
                env.append(f"EMBLASE_MLFLOW_API_KEY={settings.mlflow_api_key}")
        if settings.models_dir:
            env.append(f"EMBLASE_MODELS_DIR={settings.models_dir}")
        return env

    async def _submit_job(
        self,
        script: str,
        model_name: str,
        output: str,
        *,
        require_tiled: bool,
    ) -> str:
        """Submit *script* to Orion, store metadata, return string job_id."""
        job_id = await self.client.submit_job(
            script=script,
            working_dir=self.working_dir,
            overrides={"tres_per_job": "gres/gpu:1", "account": self.account},
            environment=self._build_environment(require_tiled=require_tiled),
        )
        self._jobs[job_id] = {
            "job_dir": f"{self.working_dir}/job_{job_id}",
            "model_name": model_name,
            "output": output,
        }
        return str(job_id)

    # -- ComputeBackend interface --------------------------------------------

    async def submit(
        self,
        model_name: str,
        batch_size: int = 1,
        images: np.ndarray | None = None,
        npy_path: str | None = None,
        run_path: str = "",
        image_key: str = "primary/pil900KW_image",
        inputs: list[str | tuple[str, str]] | None = None,
        output: str = "",
        mlflow_version: str = "",
        thumb_mode: str = "default",
        param_specs: dict | None = None,
        projector: str | None = None,
        classifier: str | None = None,
        **_: Any,
    ) -> str:
        """Submit a batch inference job.

        Provide exactly one image source: ``run_path``, ``images``, ``npy_path``,
        or ``inputs``.  If none are given, dummy images are generated on the node.
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

        if images is not None:
            buf = io.BytesIO()
            np.save(buf, images)
            b64 = base64.b64encode(buf.getvalue()).decode()
            if len(b64) > 10 * 1024 * 1024:
                raise ValueError(
                    f"Image payload too large ({len(buf.getvalue()) / 1024 / 1024:.1f} MB). "
                    "Copy the .npy to Orion and use --npy-path instead."
                )
            script_kwargs: dict[str, Any] = {"payload_b64": b64}
        elif npy_path:
            script_kwargs = {"npy_path": npy_path}
        else:
            script_kwargs = {"tiled_input": True}

        script = _build_sbatch_script(
            working_dir=self.working_dir,
            python_script=py_script,
            project_dir=self.project_dir,
            job_name=f"emblase-{model_name}",
            **script_kwargs,
        )
        return await self._submit_job(
            script,
            model_name,
            output,
            require_tiled=bool(run_path or inputs or output),
        )

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
        mem: str = "32G",
        param_specs: dict | None = None,
        projector: str | None = None,
        classifier: str | None = None,
        **_: Any,
    ) -> str:
        """Submit a streaming inference job that subscribes to a live BlueskyRun."""
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
        script = _build_sbatch_script(
            working_dir=self.working_dir,
            python_script=py_script,
            project_dir=self.project_dir,
            job_name=f"emblase-stream-{model_name}",
            time_limit="0-02:00:00",
            tiled_input=True,
            mem=mem,
        )
        return await self._submit_job(script, model_name, output, require_tiled=True)

    async def status(self, job_id: str) -> JobStatus:
        info = await self.client.get_job(int(job_id))
        return _SLURM_STATE_MAP.get(info.state, JobStatus.pending)

    async def result(self, job_id: str) -> JobResult:
        st = await self.status(job_id)
        meta = self._jobs.get(int(job_id))
        if not meta:
            return JobResult(job_id=job_id, status=JobStatus.failed, error="Job metadata lost")
        if st not in _TERMINAL_STATES:
            return JobResult(job_id=job_id, status=st)
        if meta.get("output"):
            return JobResult(job_id=job_id, status=st)
        output_path = Path(meta["job_dir"]) / "output.npy"
        if output_path.exists():
            return JobResult(
                job_id=job_id,
                status=JobStatus.completed,
                output_data=np.load(str(output_path)),
            )
        return JobResult(
            job_id=job_id,
            status=st,
            error=f"Output not found at {output_path}" if st == JobStatus.completed else None,
        )

    async def wait(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        timeout: float = 1800.0,
    ) -> JobStatus:
        info = await self.client.wait_for_job(
            int(job_id), poll_interval=poll_interval, timeout=timeout
        )
        return _SLURM_STATE_MAP.get(info.state, JobStatus.failed)

    async def cancel(self, job_id: str) -> None:
        await self.client.cancel_job(int(job_id))
