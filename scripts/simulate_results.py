"""Simulate incremental arrival of Orion inference results into a local Tiled server.

Reads an existing ``LatentSpaceEmbedding`` container from a remote Tiled server
and re-writes it batch-by-batch into a local Tiled server with an optional delay
between batches.  This lets you develop and test the WebUI against a realistic
live-updating data stream without running Orion.

The core logic lives in ``emblase.pipeline.copy_tiled.copy_embedding`` and can
be called directly from IPython with already-initialised clients::

    from tiled.client import from_uri
    from emblase.pipeline.copy_tiled import copy_embedding

    remote = from_uri("https://tiled.nsls2.bnl.gov", api_key="...")
    local  = from_uri("http://localhost:8000", api_key="secret")

    src = remote["smi", "sandbox", "confab26_demo", "results", "run_live_1086139_1777770235"]
    dst_parent = local["results"]

    dst = copy_embedding(src, dst_parent, rename="run_live_1086139",
                         batch_size=16, batch_delay=2.0)
    print(dst)

CLI example
-----------
::

    python scripts/simulate_results.py \\
        --src  smi/sandbox/confab26_demo/results/run_live_1086139_1777770235 \\
        --dst  results \\
        --rename run_live_1086139 \\
        --batch-size 16 \\
        --batch-delay 2.0 \\
        --local-uri http://localhost:8000 \\
        --local-api-key secret

The script connects to two Tiled servers:
  - Remote (source): ``EMBLASE_TILED_SERVER_URI`` / ``EMBLASE_TILED_API_KEY``
  - Local  (dest):   ``--local-uri`` / ``--local-api-key``

Notes
-----
- ``_index`` rows are sorted by ``indx`` (or ``slice`` for older containers)
  before replay, so batches arrive in acquisition order regardless of SQL row
  ordering.
- ``projections`` are written per-batch (NaN rows are preserved as-is).
- ``notes`` and ``user_labels`` are intentionally skipped — they are user edits
  that would not exist yet during a live acquisition.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import certifi as _certifi
    os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
except ImportError:
    pass

from tiled.client import from_uri

from emblase.config import settings
from emblase.pipeline.copy_tiled import copy_embedding
from emblase.tiled.client import LatentSpaceEmbedding


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stream a LatentSpaceEmbedding from remote Tiled into a local Tiled server."
    )
    p.add_argument(
        "--src",
        required=True,
        metavar="TILED_PATH",
        help="Remote path to the source LatentSpaceEmbedding container.",
    )
    p.add_argument(
        "--dst",
        required=True,
        metavar="TILED_PATH",
        help="Destination parent path on the local Tiled server.",
    )
    p.add_argument(
        "--rename",
        default=None,
        metavar="KEY",
        help=(
            "Base name for the destination container.  A Unix timestamp is appended "
            "automatically (e.g. run_live_1086139 → run_live_1086139_1234567890).  "
            "Defaults to the source container key."
        ),
    )
    p.add_argument(
        "--local-uri",
        default="http://localhost:8000",
        metavar="URI",
        help="URI of the local Tiled server (default: http://localhost:8000).",
    )
    p.add_argument(
        "--local-api-key",
        default=None,
        metavar="KEY",
        help="API key for the local Tiled server (optional).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="Number of embeddings to write per batch (default: 16).",
    )
    p.add_argument(
        "--batch-delay",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Sleep between batches to simulate live arrival (default: 2.0 s).",
    )
    p.add_argument(
        "--access-tags",
        default="",
        metavar="TAG1,TAG2",
        help="Comma-separated Tiled access tags for the destination container.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # --- connect to remote source ---
    remote = from_uri(settings.tiled_server_uri, api_key=settings.tiled_api_key)
    src_segments = [s for s in args.src.split("/") if s]
    try:
        src_node = remote[tuple(src_segments)]
    except KeyError:
        sys.exit(f"Source not found: {args.src}")
    if not isinstance(src_node, LatentSpaceEmbedding):
        sys.exit(
            f"{args.src!r} is not a LatentSpaceEmbedding "
            f"(got {type(src_node).__name__})"
        )

    # --- connect to local destination ---
    local = from_uri(args.local_uri, api_key=args.local_api_key)
    dst_segments = [s for s in args.dst.split("/") if s]
    try:
        dst_parent = local[tuple(dst_segments)] if dst_segments else local
    except KeyError:
        sys.exit(f"Destination parent not found: {args.dst}")

    access_tags = [t.strip() for t in args.access_tags.split(",") if t.strip()] or None

    n_total = src_node.num_embeddings
    print(f"Source: {args.src}  ({n_total} embeddings, dim={src_node.embedding_dim})")
    print(f"Destination parent: {args.dst}  (local: {args.local_uri})")
    print(f"batch_size={args.batch_size}, batch_delay={args.batch_delay} s")

    dst = copy_embedding(
        src_node,
        dst_parent,
        rename=args.rename,
        batch_size=args.batch_size,
        batch_delay=args.batch_delay,
        access_tags=access_tags,
    )
    print(f"Done. {n_total} embeddings written → {dst}")


if __name__ == "__main__":
    main()
