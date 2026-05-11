"""Instrument router — replay existing Tiled data into the inputs container.

This simulates live acquisition by copying a source BlueskyRun into the
configured EMBLASE_TILED_INPUT_CONTAINER one frame at a time, with an
optional delay between frames.

Endpoints
---------
POST /instrument/start   — begin a copy job (returns immediately, copy runs in background thread)
POST /instrument/stop    — cancel a running copy
GET  /instrument/status  — current state (idle/running/done/error)
GET  /instrument/progress — SSE stream of progress events while running
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...config import settings

router = APIRouter(prefix="/instrument", tags=["instrument"])


# ── State ─────────────────────────────────────────────────────────────────────

class _State:
    """Process-global instrument state (single concurrent copy supported)."""

    def __init__(self) -> None:
        self.status: str = "idle"       # idle | running | done | error
        self.src: str = ""
        self.rename: str = ""
        self.frames_written: int = 0
        self.frames_total: int = 0
        self.elapsed: float = 0.0
        self.error: str = ""
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "src": self.src,
                "rename": self.rename,
                "frames_written": self.frames_written,
                "frames_total": self.frames_total,
                "elapsed": round(self.elapsed, 1),
                "error": self.error,
                "inputs_container": settings.tiled_input_container,
            }


_state = _State()


# ── Request model ─────────────────────────────────────────────────────────────

class InstrumentStartRequest(BaseModel):
    src: str = Field(..., description="Source Tiled run path to replay")
    image_key: str = "primary/pil900KW_image"
    batch_delay: float = Field(default=0.1, ge=0.0, le=60.0, description="Seconds between frames")


# ── Copy thread ───────────────────────────────────────────────────────────────

def _run_copy(req: InstrumentStartRequest) -> None:
    """Run deepcopy in a background thread; updates _state throughout."""
    try:
        import os
        try:
            import certifi as _certifi
            os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
        except ImportError:
            pass

        from tiled.client import from_uri
        from ...pipeline.copy_tiled import deepcopy

        tiled_uri = settings.tiled_server_uri
        tiled_key = settings.tiled_api_key or None
        dst_path = settings.tiled_input_container

        if not tiled_uri:
            raise RuntimeError("EMBLASE_TILED_SERVER_URI is not set")
        if not dst_path:
            raise RuntimeError("EMBLASE_TILED_INPUT_CONTAINER is not set")

        client = from_uri(tiled_uri, api_key=tiled_key)

        src_segs = [s for s in req.src.split("/") if s]
        dst_segs = [s for s in dst_path.split("/") if s]
        try:
            src_node = client[tuple(src_segs)]
        except KeyError:
            raise RuntimeError(f"Source not found in Tiled: {req.src}")
        try:
            dst_node = client[tuple(dst_segs)]
        except KeyError:
            raise RuntimeError(f"Inputs container not found in Tiled: {dst_path}")

        # Auto-generate rename: <last_segment>_live_<timestamp>
        base = src_segs[-1] + "_live"
        rename = f"{base}_{int(time.time())}"

        t_start = time.monotonic()

        with _state._lock:
            _state.rename = rename
            _state.frames_written = 0
            _state.frames_total = 0

        def _progress(written: int, total: int) -> None:
            if _state._stop_event.is_set():
                raise InterruptedError("copy cancelled")
            with _state._lock:
                _state.frames_written = written
                _state.frames_total = total
                _state.elapsed = time.monotonic() - t_start

        deepcopy(
            src_node,
            dst_node,
            rename=rename,
            batch_size=1,
            batch_delay=req.batch_delay,
            image_key=req.image_key,
            on_progress=_progress,
        )

        with _state._lock:
            _state.status = "done"
            _state.elapsed = time.monotonic() - t_start

    except InterruptedError:
        with _state._lock:
            _state.status = "idle"
    except Exception as exc:
        with _state._lock:
            _state.status = "error"
            _state.error = str(exc)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/start")
async def instrument_start(req: InstrumentStartRequest) -> dict[str, Any]:
    """Begin replaying a source run into the inputs container."""
    with _state._lock:
        if _state.status == "running":
            raise HTTPException(409, "A copy is already running — stop it first")
        _state.status = "running"
        _state.src = req.src
        _state.error = ""
        _state._stop_event.clear()

    _state._thread = threading.Thread(target=_run_copy, args=(req,), daemon=True)
    _state._thread.start()
    return {"status": "running", "src": req.src, "inputs_container": settings.tiled_input_container}


@router.post("/stop")
async def instrument_stop() -> dict[str, Any]:
    """Cancel a running copy."""
    with _state._lock:
        if _state.status != "running":
            return {"status": _state.status, "message": "Nothing to stop"}
        _state._stop_event.set()
    return {"status": "stopping"}


@router.get("/status")
async def instrument_status() -> dict[str, Any]:
    """Return current instrument state."""
    return _state.to_dict()


async def _progress_generator() -> AsyncGenerator[str, None]:
    """SSE generator that emits state updates ~2 s until done/error/idle."""
    while True:
        data = _state.to_dict()
        yield f"data: {data}\n\n"
        if data["status"] in ("done", "error", "idle"):
            break
        await asyncio.sleep(2.0)


@router.get("/progress")
async def instrument_progress() -> StreamingResponse:
    """Stream instrument progress as SSE (one event per ~2 s)."""
    return StreamingResponse(
        _progress_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
