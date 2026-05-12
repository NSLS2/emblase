"""Watcher router — manage the InputsWatcher (Tiled WebSocket subscription).

The watcher subscribes to EMBLASE_TILED_INPUT_CONTAINER via WebSocket.
Each time a new BlueskyRun appears it submits a streaming inference job to the
configured compute backend.

Endpoints
---------
POST /watcher/start   — start the watcher (idempotent if already running)
POST /watcher/stop    — stop the watcher
GET  /watcher/status  — current state and job count
GET  /watcher/jobs    — list of all job IDs submitted by the watcher this session
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...config import resolve_tiled_path, settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/watcher", tags=["watcher"])


# ── Request model ─────────────────────────────────────────────────────────────


class ParamSpec(BaseModel):
    name: str
    source: str
    dtype: str = "float"
    units: str = ""


class WatcherStartRequest(BaseModel):
    backend: str = Field(default="orion", description="Compute backend: orion or nersc")
    model_name: str = "bnl-nsls2-smi-vit"
    mlflow_version: str = ""
    output_container: str = ""
    batch_size: int = Field(default=1, ge=1, le=256)
    image_key: str = "primary/pil900KW_image"
    thumb_mode: str = "logroi"
    param_specs: list[ParamSpec] = []
    projector: str | None = None
    classifier: str | None = None
    # NERSC-specific
    nersc_queue: str = ""
    nersc_account: str = ""
    nersc_time_limit: str = "02:00:00"
    nersc_constraint: str = ""
    # Orion-specific
    orion_account: str = ""
    # Watcher behaviour
    replay_existing: bool = False


# ── Watcher state ────────────────────────────────────────────────────────────


class WatcherState:
    """Thread-safe holder for watcher lifecycle state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status: str = "idle"  # idle | running | error
        self.backend_name: str = ""
        self.inputs_path: str = ""
        self.jobs_submitted: int = 0
        self.job_records: list[dict] = []
        self.error: str = ""
        self._watcher = None  # InputsWatcher instance while running
        self._thread: threading.Thread | None = None
        self._loop = None  # asyncio event loop for the watcher thread

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "backend": self.backend_name,
                "inputs_path": self.inputs_path,
                "jobs_submitted": self.jobs_submitted,
                "error": self.error,
            }


_ws = WatcherState()


# ── Watcher thread ────────────────────────────────────────────────────────────


def _run_watcher(req: WatcherStartRequest) -> None:
    """Start InputsWatcher in a background thread. Updates _ws state."""
    try:
        import os

        try:
            import certifi as _certifi

            os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
        except ImportError:
            pass

        from tiled.client import from_uri

        from ...pipeline.streaming import InputsWatcher

        tiled_uri = settings.tiled_server_uri
        inputs_path = settings.tiled_input_container
        output_path = resolve_tiled_path(
            req.output_container or None, settings.tiled_output_container
        )

        if not tiled_uri:
            raise RuntimeError("EMBLASE_TILED_SERVER_URI is not set")
        if not inputs_path:
            raise RuntimeError("EMBLASE_TILED_INPUT_CONTAINER is not set")
        if not output_path:
            raise RuntimeError("EMBLASE_TILED_OUTPUT_CONTAINER is not set")

        client = from_uri(tiled_uri, api_key=settings.tiled_api_key or None)
        segs = [s for s in inputs_path.split("/") if s]
        try:
            inputs_node = client[tuple(segs)]
        except KeyError:
            raise RuntimeError(f"Inputs container not found in Tiled: {inputs_path}")

        # Build backend with any per-request overrides, same pattern as jobs.py
        if req.backend == "nersc":
            from ...compute.nersc import NERSCBackend

            backend = NERSCBackend(
                account=req.nersc_account or None,
                queue=req.nersc_queue or None,
                time_limit=req.nersc_time_limit or None,
                constraint=req.nersc_constraint or None,
            )
        else:
            from ...compute.orion import OrionBackend

            backend = OrionBackend(account=req.orion_account or None)

        loop = asyncio.new_event_loop()
        with _ws._lock:
            _ws._loop = loop

        threading.Thread(
            target=lambda: (asyncio.set_event_loop(loop), loop.run_forever()),
            daemon=True,
            name="emblase-watcher-asyncio",
        ).start()

        # Wrap submit_streaming to track job count and record job IDs
        _original_submit = backend.submit_streaming

        async def _counted_submit(*args, **kwargs):
            # Extract run_path from kwargs to find any matching scheduled record
            run_path: str = kwargs.get("run_path", args[0] if args else "")
            run_key = run_path.rsplit("/", 1)[-1] if run_path else ""
            job_id = await _original_submit(*args, **kwargs)
            record: dict[str, Any] = {
                "job_id": str(job_id),
                "backend": req.backend,
                "mode": "stream",
                "model_name": req.model_name,
                "submitted_at": time.time(),
                "state": "",
            }
            with _ws._lock:
                _ws.jobs_submitted += 1
                # Replace placeholder scheduled record if present
                scheduled_id = f"scheduled-{run_key}"
                replaced = False
                for i, r in enumerate(_ws.job_records):
                    if r.get("job_id") == scheduled_id:
                        _ws.job_records[i] = record
                        replaced = True
                        break
                if not replaced:
                    _ws.job_records.append(record)
            return job_id

        def _on_queued(run_key: str) -> None:
            """Record a placeholder entry for a run waiting in the queue."""
            record: dict[str, Any] = {
                "job_id": f"scheduled-{run_key}",
                "backend": req.backend,
                "mode": "stream",
                "model_name": req.model_name,
                "submitted_at": time.time(),
                "state": "scheduled",
            }
            with _ws._lock:
                _ws.job_records.append(record)

        backend.submit_streaming = _counted_submit

        watcher = InputsWatcher(
            inputs_node=inputs_node,
            output_root=output_path,
            backend=backend,
            model_name=req.model_name,
            batch_size=req.batch_size,
            image_key=req.image_key,
            thumb_mode=req.thumb_mode,
            mlflow_version=req.mlflow_version,
            param_specs={
                s.name: {"source": s.source, "dtype": s.dtype, "units": s.units}
                for s in req.param_specs
                if s.name and s.source
            }
            or None,
            projector=req.projector,
            classifier=req.classifier,
            loop=loop,
            on_queued=_on_queued,
        )

        with _ws._lock:
            _ws._watcher = watcher
            _ws.inputs_path = inputs_path

        watcher.start(replay_existing=req.replay_existing)  # blocks until stop()

        with _ws._lock:
            _ws.status = "idle"
            _ws._watcher = None
            loop.call_soon_threadsafe(loop.stop)

    except Exception as exc:
        logger.error("Watcher thread failed: %s", exc)
        with _ws._lock:
            _ws.status = "error"
            _ws.error = "Watcher failed — see server logs for details"
            _ws._watcher = None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/start")
async def watcher_start(req: WatcherStartRequest) -> dict[str, Any]:
    """Start the InputsWatcher."""
    with _ws._lock:
        if _ws.status == "running":
            raise HTTPException(409, "Watcher is already running — stop it first")
        _ws.status = "running"
        _ws.backend_name = req.backend
        _ws.jobs_submitted = 0
        _ws.error = ""

    _ws._thread = threading.Thread(target=_run_watcher, args=(req,), daemon=True)
    _ws._thread.start()
    return _ws.to_dict()


@router.post("/stop")
async def watcher_stop() -> dict[str, Any]:
    """Stop the InputsWatcher."""
    with _ws._lock:
        if _ws.status != "running" or _ws._watcher is None:
            return {"status": _ws.status, "message": "Watcher is not running"}
        watcher = _ws._watcher

    def _stop():
        try:
            watcher.stop()
        except Exception:
            pass

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _stop)
    return {"status": "stopping"}


@router.get("/status")
async def watcher_status() -> dict[str, Any]:
    """Return current watcher state."""
    return _ws.to_dict()


@router.get("/jobs")
async def watcher_jobs() -> dict[str, Any]:
    """Return all job records submitted by the watcher this server session."""
    with _ws._lock:
        return {"jobs": list(_ws.job_records)}
