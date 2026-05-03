"""Simulate incremental arrival of Orion inference results into a local Tiled server.

Reads an existing ``LatentSpaceEmbedding`` container from a remote Tiled server
and re-writes it row-by-row into a local Tiled server, with an optional delay
between batches.  This lets you develop and test the WebUI against a realistic
live-updating data stream without running Orion.

The destination container is created fresh on the local server with the same
metadata (model_name, param_specs, embedding_dim, etc.) as the source.

Example
-------
Stream results from the remote sandbox into a local Tiled instance::

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
- ``_index`` rows are sorted by ``slice`` (source frame index) before replay,
  so batches arrive in acquisition order regardless of SQL row ordering.
- ``projections`` are written per-batch if present and non-NaN; NaN rows are
  written as NaN (i.e. not yet computed) to faithfully simulate the pipeline.
- ``notes`` and ``user_labels`` are intentionally skipped — they are user edits
  that would not exist yet during a live acquisition.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import certifi as _certifi
    os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
except ImportError:
    pass

import numpy as np
from tiled.client import from_uri

from emblase.config import settings
from emblase.tiled.client import LatentSpaceEmbedding, create_embedding_container


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
        help="Destination container path on the local Tiled server (parent of the new run node).",
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

    base_name = args.rename or src_segments[-1]
    ts = int(time.time())
    dst_key = f"{base_name}_{ts}"

    access_tags = [t.strip() for t in args.access_tags.split(",") if t.strip()] or None

    # --- read source metadata ---
    src_meta = src_node.metadata
    param_specs = src_meta.get("param_specs") or {}
    embedding_dim = src_meta.get("embedding_dim", 512)
    thumb_shape = tuple(src_meta.get("thumb_shape", [64, 64]))
    model_name = src_meta.get("model_name", "")
    model_version = src_meta.get("model_version", "")

    # --- read and sort source data by slice (frame index) ---
    print(f"Reading source: {args.src}")
    embeddings  = src_node["embeddings"].read()   # (N, D)
    thumbnails  = src_node["thumbnails"].read()   # (N, H, W)
    projections = src_node["projections"].read()  # (N, 2)
    idx_df      = src_node.base["_index"].read()

    # Sort by indx (row offset in embeddings/projections arrays) if present,
    # falling back to slice (source frame index) for older containers.
    sort_col = "indx" if "indx" in idx_df.columns else "slice"
    idx_df = idx_df.sort_values(sort_col, key=lambda s: s.astype(int)).reset_index(drop=True)
    sort_order = idx_df.index.tolist()

    # Re-order arrays to match sorted index
    embeddings  = embeddings[sort_order]
    thumbnails  = thumbnails[sort_order]
    projections = projections[sort_order]

    n_total = len(embeddings)
    print(f"Source has {n_total} embeddings, embedding_dim={embedding_dim}")
    print(f"Destination: {args.dst}/{dst_key}")
    print(f"batch_size={args.batch_size}, batch_delay={args.batch_delay} s")

    # --- create destination container ---
    dst_container = create_embedding_container(
        dst_parent,
        dst_key,
        embedding_dim=embedding_dim,
        thumb_shape=thumb_shape,
        model_name=model_name,
        model_version=model_version,
        params=param_specs,
        access_tags=access_tags,
    )
    print(f"Created destination container: {args.dst}/{dst_key}")

    # --- stream batches ---
    param_names = list(param_specs.keys())

    for start in range(0, n_total, args.batch_size):
        end = min(start + args.batch_size, n_total)
        batch_slice = slice(start, end)
        n = end - start

        emb_batch  = embeddings[batch_slice]
        thumb_batch = thumbnails[batch_slice]
        proj_batch  = projections[batch_slice]
        rows        = idx_df.iloc[batch_slice]

        # source_entries: (path, slice_str) per embedding
        source_entries = [
            (str(rows["path"].iloc[i]), str(rows["slice"].iloc[i]))
            for i in range(n)
        ]

        # params dict
        params = None
        if param_names:
            params = {
                name: rows[f"param_{name}"].tolist()
                for name in param_names
                if f"param_{name}" in rows.columns
            }

        dst_container.append(
            emb_batch.astype(np.float32),
            thumb_batch.astype(np.float32),
            paths=[e[0] for e in source_entries],
            slices=[e[1] for e in source_entries],
            model_version=model_version or None,
            timestamps=rows["timestamp"].tolist(),
            projections=proj_batch.astype(np.float32),
            params=params,
            access_tags=access_tags,
        )
        print(
            f"  [{end:>4}/{n_total}] wrote indx "
            f"{rows[sort_col].iloc[0]}–{rows[sort_col].iloc[-1]}"
        )

        if args.batch_delay > 0 and end < n_total:
            time.sleep(args.batch_delay)

    print(f"Done. {n_total} embeddings written to {args.dst}/{dst_key}")


if __name__ == "__main__":
    main()
