"""Watch a Tiled inputs container and submit one streaming inference job per new run.

The watcher subscribes to a Tiled container via WebSocket.  Each time a new
BlueskyRun appears it submits a streaming inference job to the chosen compute
backend, which subscribes to the run's image array, encodes frames as they
arrive, and writes embeddings incrementally to Tiled.

Backends
--------
``--backend orion`` (default)
    Submits a Slurm job via the Orion REST API.  Requires ``EMBLASE_ORION_API_KEY``.

``--backend local``
    Runs inference in-process.  Useful for development.

``--backend nersc`` (experimental)
    Submits to NERSC Perlmutter via the IRI API + Shifter.
    Requires ``EMBLASE_NERSC_API_TOKEN``.

Example — Terminal 1 (start the watcher)
-----------------------------------------
::

    pixi run python scripts/start_watcher.py \\
        --inputs     smi/sandbox/confab26_demo/inputs_copy \\
        --output     smi/sandbox/confab26_demo/results \\
        --backend    orion \\
        --model      bnl-nsls2-smi-vit \\
        --image-key  primary/pil900KW_image \\
        --batch-size 1 \\
        --thumb-mode logroi \\
        --param      temperature:primary/LinkamThermal_temperature_current:float:°C \\
        --param      piezo_x:primary/piezo_x:float:μm \\
        --no-replay

Example — Terminal 2 (simulate acquisition)
--------------------------------------------
::

    pixi run python scripts/simulate_acquisition.py \\
        --src         smi/sandbox/confab26_demo/inputs/run_1086139 \\
        --dst         smi/sandbox/confab26_demo/inputs_copy \\
        --rename      run_live_1086139 \\
        --access-tags smi_sandbox \\
        --batch-delay 0.5

--param syntax
--------------
    --param name:source[:dtype[:units]]

- ``source``: ``<stream>/<array_key>`` aligned 1:1 with frames
- ``dtype``: ``float`` (default), ``integer``, ``string``, or ``boolean``
- Repeat for multiple parameters
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

from emblase.compute import build_backend, parse_param_specs  # noqa: E402
from emblase.config import settings  # noqa: E402
from emblase.pipeline.streaming import InputsWatcher  # noqa: E402

try:
    from tiled.client import from_uri
except ImportError:
    sys.exit("tiled is not installed — run: pixi install")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Watch a Tiled inputs container and submit streaming inference jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--inputs",
        required=True,
        metavar="TILED_PATH",
        help="Tiled path of the inputs container to watch.",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="TILED_PATH",
        help="Tiled --output path passed to each streaming job.  If the path is an existing "
        "LatentSpaceEmbedding it is appended to; if it doesn't exist an LSE is created "
        "there; if it exists but is not an LSE a new LSE named "
        "{run_key}_stream_{timestamp} is created inside it.",
    )
    p.add_argument(
        "--backend",
        default="orion",
        choices=["orion", "local", "nersc"],
        help="Compute backend (default: orion).",
    )
    p.add_argument(
        "--model",
        default="bnl-nsls2-smi-vit",
        metavar="MODEL_NAME",
        help="Model name: short architecture or MLflow registry name.",
    )
    p.add_argument(
        "--mlflow-version",
        default="",
        metavar="VERSION",
        help="MLflow model version (default: latest).",
    )
    p.add_argument(
        "--image-key",
        default="primary/pil900KW_image",
        metavar="KEY",
        help="Slash-separated image-array key, e.g. 'primary/pil900KW_image'.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Frames per encode call on the compute node (default: 1).",
    )
    p.add_argument(
        "--thumb-mode",
        default="logroi",
        choices=["default", "logroi"],
        help="Thumbnail mode (default: logroi).",
    )
    p.add_argument(
        "--projector",
        default=None,
        metavar="NAME|false",
        help="Projector: omit=scratch, NAME=saved approx, false=NaN.",
    )
    p.add_argument(
        "--classifier",
        default=None,
        metavar="NAME",
        help="Classifier name (omit = no labels).",
    )
    p.add_argument(
        "--param",
        action="append",
        default=[],
        dest="params",
        metavar="name:source[:dtype[:units]]",
        help="Scalar param to store per embedding. Repeat for multiple.",
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
        help="Skip runs already present in --inputs on startup.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy HTTP-level logs from httpx/httpcore unless DEBUG is requested.
    if args.log_level != "DEBUG":
        for _noisy in ("httpx", "httpcore", "hpack"):
            logging.getLogger(_noisy).setLevel(logging.WARNING)

    if not settings.tiled_server_uri:
        sys.exit("EMBLASE_TILED_SERVER_URI is not set — check your .env file.")

    client = from_uri(settings.tiled_server_uri, api_key=settings.tiled_api_key or None)
    segments = [s for s in args.inputs.split("/") if s]
    try:
        inputs_node = client[tuple(segments)]
    except KeyError:
        sys.exit(f"Inputs container not found in Tiled: {args.inputs}")

    backend = build_backend(args.backend)
    access_tags = [t.strip() for t in args.access_tags.split(",") if t.strip()] or None

    # Run the asyncio event loop in a dedicated thread so backend coroutines
    # can be scheduled from the WebSocket callback threads.
    loop = asyncio.new_event_loop()
    threading.Thread(
        target=lambda: (asyncio.set_event_loop(loop), loop.run_forever()),
        daemon=True,
        name="emblase-asyncio",
    ).start()

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
        param_specs=parse_param_specs(args.params),
        projector=args.projector,
        classifier=args.classifier,
        loop=loop,
    )

    def _shutdown(signum, frame) -> None:  # noqa: ANN001
        print("\nShutting down …")
        watcher.stop()
        loop.call_soon_threadsafe(loop.stop)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    param_specs = parse_param_specs(args.params)
    print(f"Backend  : {args.backend.upper()}")
    print(f"Watching : {args.inputs}")
    print(f"Output   : {args.output}")
    print(f"Model    : {args.model}  image_key={args.image_key}  batch_size={args.batch_size}")
    if param_specs:
        print(f"Params   : {list(param_specs)}")
    if args.projector:
        print(f"Projector: {args.projector}")
    if args.classifier:
        print(f"Classifier: {args.classifier}")
    print("Press Ctrl+C to stop.\n")

    watcher.start(replay_existing=not args.no_replay)


if __name__ == "__main__":
    main()
