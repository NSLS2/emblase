"""Streaming inference pipeline CLI.

Watch a Tiled inputs container for new BlueskyRun children, batch their image
frames, submit inference jobs to Orion, and write embeddings incrementally to
a results container.

The watcher opens a dedicated Tiled client (WebSocket) for listening.  Any
process that writes new runs into the inputs container (e.g. the beamline DAQ
or a deepcopy script) must use a **separate** Tiled client so that the two
httpx connections do not interfere.

Example
-------
Watch inputs_copy, write to results/stream, using noop model locally::

    python scripts/stream_pipeline.py \\
        --inputs  smi/sandbox/confab26_demo/inputs_copy \\
        --output  smi/sandbox/confab26_demo/results \\
        --model   noop \\
        --batch-size 4 \\
        --thumb-mode logroi

Press Ctrl+C to stop watching.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emblase.compute.orion import OrionBackend, OrionClient  # noqa: E402
from emblase.pipeline.streaming import InputsWatcher  # noqa: E402
from tiled.client import from_uri  # noqa: E402

from emblase.config import settings  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Watch a Tiled inputs container and stream inference to Orion."
    )
    p.add_argument(
        "--inputs",
        required=True,
        metavar="TILED_PATH",
        help="Tiled path to the inputs container to watch (e.g. smi/sandbox/…/inputs_copy).",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="TILED_PATH",
        help="Tiled path to the results container. Each run's output is placed at OUTPUT/<run_key>.",
    )
    p.add_argument(
        "--model",
        default="vae",
        metavar="MODEL_NAME",
        help="MLflow model name to use for inference (default: vae).",
    )
    p.add_argument(
        "--mlflow-version",
        default="",
        metavar="VERSION",
        help="Optional MLflow model version string.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Number of frames per Orion job (default: 8).",
    )
    p.add_argument(
        "--image-key",
        default="pil900KW_image",
        metavar="KEY",
        help="Name of the image array within the primary stream (default: pil900KW_image).",
    )
    p.add_argument(
        "--thumb-mode",
        default="logroi",
        choices=["default", "logroi"],
        help="Thumbnail generation mode (default: logroi).",
    )
    p.add_argument(
        "--access-tags",
        default="",
        metavar="TAG1,TAG2",
        help="Comma-separated Tiled access tags for output containers.",
    )
    p.add_argument(
        "--no-replay",
        action="store_true",
        help="Do not replay existing runs in the inputs container on startup.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    tiled_uri = settings.tiled_server_uri
    tiled_key = settings.tiled_api_key
    if not tiled_uri:
        sys.exit("EMBLASE_TILED_SERVER_URI is not set.")

    client = from_uri(tiled_uri, api_key=tiled_key)

    # Resolve inputs node
    inputs_segments = [s for s in args.inputs.split("/") if s]
    try:
        inputs_node = client[tuple(inputs_segments)]
    except KeyError:
        sys.exit(f"Inputs container not found: {args.inputs}")

    access_tags = [t.strip() for t in args.access_tags.split(",") if t.strip()]

    # Build Orion backend
    backend = OrionBackend()

    # Run the async event loop in a dedicated thread so Orion coroutines can
    # be scheduled from the websocket callback threads.
    loop = asyncio.new_event_loop()

    def _run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop_thread = threading.Thread(target=_run_loop, daemon=True, name="emblase-asyncio")
    loop_thread.start()

    watcher = InputsWatcher(
        inputs_node=inputs_node,
        output_root=args.output,
        backend=backend,
        model_name=args.model,
        batch_size=args.batch_size,
        image_key=args.image_key,
        thumb_mode=args.thumb_mode,
        mlflow_version=args.mlflow_version,
        access_tags=access_tags,
        loop=loop,
    )

    def _shutdown(signum, frame) -> None:  # noqa: ANN001
        print("\nShutting down…")
        watcher.stop()
        loop.call_soon_threadsafe(loop.stop)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"Watching {args.inputs}  →  output: {args.output}")
    print("Press Ctrl+C to stop.")
    watcher.start(replay_existing=not args.no_replay)  # blocks


if __name__ == "__main__":
    main()
