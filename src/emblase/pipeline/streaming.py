"""Streaming pipeline: watch a Tiled inputs container for new BlueskyRuns and
submit one streaming-inference job per run to the configured compute backend.

Architecture
------------
InputsWatcher (runs locally)
  └─ inputs_node.subscribe()
       on child_created(run_key) →
           backend.submit_streaming(run_path, output_path)  ← one job per run

The compute job (streaming_inference.py.tmpl) does everything else:
  - subscribes to primary → image array via WebSocket on the compute node
  - encodes frames in batches as they arrive
  - writes embeddings incrementally to Tiled
  - exits when the stop document is received

Key rules
---------
- inputs_node must be a **dedicated** Tiled client — not shared with any writer.
- The watcher never touches the data; it only reacts to container-level events.
- Any backend implementing ``ComputeBackend.submit_streaming`` is supported
  (Orion, local, NERSC, …).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import urllib.parse
from typing import Any

# websockets uses ssl.create_default_context() which on some systems may not
# include the CA chain for tiled.nsls2.bnl.gov (InCommon RSA).
# Force certifi if available and SSL_CERT_FILE not already set.
try:
    import certifi as _certifi

    os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
except ImportError:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helper — used to build Tiled path strings for Orion job parameters
# ---------------------------------------------------------------------------


def _tiled_path(node: Any) -> str:
    """Return the slash-joined Tiled path for a node."""
    parsed = urllib.parse.urlparse(str(node.uri))
    marker = "/api/v1/metadata/"
    if marker not in parsed.path:
        raise ValueError(f"Cannot parse path from URI: {node.uri}")
    return parsed.path[parsed.path.index(marker) + len(marker) :].strip("/")


# ---------------------------------------------------------------------------
# InputsWatcher
# ---------------------------------------------------------------------------


class InputsWatcher:
    """Watch a Tiled inputs container; submit one Orion streaming job per new run.

    Parameters
    ----------
    inputs_node:
        Tiled container node to watch (e.g. ``inputs_copy``).
        Must be a **dedicated** client — not shared with any writer.
    output_root:
        Tiled path of the results container.  Each run's output is placed at
        ``{output_root}/{run_key}``.
    backend:
        A ``ComputeBackend`` instance (Orion, local, NERSC, …) with a
        ``submit_streaming`` coroutine.
    model_name, batch_size, image_key, thumb_mode, mlflow_version:
        Forwarded to the Orion streaming job.
    param_specs:
        Optional dict mapping parameter names to ParamSpec dicts.  Forwarded
        verbatim to ``submit_streaming`` (and hence to the Orion job).
    projector:
        Projector behaviour forwarded to the Orion job.  ``None`` (default) = fit
        from scratch after the run completes; ``"<name>"`` = use saved
        approximator; ``"false"``/``"0"`` = NaN projections.
    classifier:
        Classifier name forwarded to the Orion job.  ``None`` (default) = no
        classification; ``"<name>"`` = use saved classifier from ``models_dir``.
    loop:
        asyncio event loop for scheduling ``submit_streaming`` coroutines.
    """

    def __init__(
        self,
        inputs_node: Any,
        output_root: str,
        backend: Any,
        model_name: str,
        batch_size: int = 8,
        image_key: str = "primary/pil900KW_image",
        thumb_mode: str = "logroi",
        mlflow_version: str = "",
        access_tags: list[str] | None = None,
        param_specs: dict | None = None,
        projector: str | None = None,
        classifier: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.inputs_node = inputs_node
        self.output_root = output_root
        self.backend = backend
        self.model_name = model_name
        self.batch_size = batch_size
        self.image_key = image_key
        self.thumb_mode = thumb_mode
        self.mlflow_version = mlflow_version
        self.access_tags = access_tags
        self.param_specs = param_specs
        self.projector = projector
        self.classifier = classifier
        self.loop = loop
        self._seen_runs: set[str] = set()
        self._seen_runs_lock = threading.Lock()
        self._sub: Any = None

    def _subscribe(self, replay_existing: bool):
        """Create subscription and attach callback; return (sub, start_seq)."""
        start_seq = 1 if replay_existing else None
        logger.info("InputsWatcher watching %s", _tiled_path(self.inputs_node))
        sub = self.inputs_node.subscribe()
        sub.child_created.add_callback(self._on_child_created)
        return sub, start_seq

    def start(self, replay_existing: bool = True) -> None:
        """Start watching. Blocks until ``stop()`` is called."""
        self._sub, start_seq = self._subscribe(replay_existing)
        self._sub.start(start=start_seq)  # blocks

    def start_in_thread(self, replay_existing: bool = True) -> "InputsWatcher":
        """Start in a background thread; returns once WS is connected."""
        self._sub, start_seq = self._subscribe(replay_existing)
        self._sub.start_in_thread(start=start_seq)
        return self

    def stop(self) -> None:
        if self._sub:
            self._sub.disconnect()

    def __enter__(self) -> "InputsWatcher":
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def _on_child_created(self, event: Any) -> None:
        run_key = event.key
        with self._seen_runs_lock:
            if run_key in self._seen_runs:
                logger.debug("Already submitted job for run %s — skipping", run_key)
                return
            self._seen_runs.add(run_key)

        run_path = _tiled_path(self.inputs_node) + f"/{run_key}"
        output_path = self.output_root
        logger.info("New run: %s  →  submitting streaming job", run_key)

        coro = self._submit(run_path, output_path)
        if self.loop is not None and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self.loop)
        else:
            threading.Thread(
                target=asyncio.run,
                args=(coro,),
                daemon=True,
                name=f"emblase-submit-{run_key}",
            ).start()

    async def _submit(self, run_path: str, output_path: str) -> None:
        try:
            job_id = await self.backend.submit_streaming(
                run_path=run_path,
                output=output_path,
                model_name=self.model_name,
                batch_size=self.batch_size,
                image_key=self.image_key,
                thumb_mode=self.thumb_mode,
                mlflow_version=self.mlflow_version,
                access_tags=self.access_tags,
                param_specs=self.param_specs,
                projector=self.projector,
                classifier=self.classifier,
            )
            run_key = run_path.rsplit("/", 1)[-1]
            logger.info("Submitted streaming job %s for run %s", job_id, run_path)
            self._start_job_monitor(job_id, run_key)
        except Exception:
            logger.exception("Failed to submit streaming job for run %s", run_path)

    def _start_job_monitor(self, job_id: str, run_key: str) -> None:
        """If the backend supports job monitoring, start it in a daemon thread."""
        monitor = getattr(self.backend, "monitor_job", None)
        if monitor is None:
            return

        def _run() -> None:
            try:
                monitor(job_id, log_prefix=f"[job {job_id} / {run_key}]")
            except Exception as exc:
                logger.debug("[job %s] monitor stopped: %s", job_id, exc)

        threading.Thread(target=_run, daemon=True, name=f"emblase-monitor-{job_id}").start()
