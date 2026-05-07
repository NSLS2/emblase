"""Simulate live acquisition by copying a BlueskyRun into a watched inputs container.

Copies a run from a source Tiled path into a destination container one frame at
a time (default), optionally with a delay between frames to mimic real detector
acquisition speed.  The destination container key is auto-generated as
``<rename>_<unix_timestamp>`` unless ``--rename`` already contains a timestamp.

This script is the standard way to test the streaming pipeline locally.  Run it
*after* ``start_watcher.py`` is already watching the destination container —
the container creation event triggers the compute backend job submission.

Example
-------
Copy run_1086139 into inputs_copy, triggering the streaming pipeline::

    python scripts/simulate_acquisition.py \\
        --src  smi/sandbox/confab26_demo/inputs/run_1086139 \\
        --dst  smi/sandbox/confab26_demo/inputs_copy \\
        --rename run_live_1086139 \\
        --access-tags smi_sandbox \\
        --batch-delay 0.1

Notes
-----
- Always use ``--batch-size 1`` (default).  Larger batch sizes cause Tiled to
  store the array with internal spatial chunking; the WS subscription then
  replays partial tiles instead of full frames, breaking the compute job.
- Start ``start_watcher.py`` first, wait for "Press Ctrl+C to stop", then
  run this script.
- 288 frames x 0.1 s/frame ≈ 30 s total; the compute job starts in ~25 s so
  timing is tight — if the job starts after deepcopy finishes it will hang.
  Use ``--batch-delay 0.5`` for a safer margin (144 s total).
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

from tiled.client import from_uri

from emblase.config import settings
from emblase.pipeline.copy_tiled import deepcopy


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Copy a BlueskyRun into a watched inputs container to simulate live acquisition."
    )
    p.add_argument(
        "--src",
        required=True,
        metavar="TILED_PATH",
        help="Source run path (e.g. smi/sandbox/confab26_demo/inputs/run_1086139).",
    )
    p.add_argument(
        "--dst",
        required=True,
        metavar="TILED_PATH",
        help="Destination container path (e.g. smi/sandbox/confab26_demo/inputs_copy).",
    )
    p.add_argument(
        "--rename",
        default=None,
        metavar="KEY",
        help=(
            "Base name for the copied run.  A Unix timestamp is appended automatically "
            "(e.g. --rename run_live_1086139 → run_live_1086139_1234567890).  "
            "Defaults to the source run key with '_live' inserted."
        ),
    )
    p.add_argument(
        "--access-tags",
        default="",
        metavar="TAG1,TAG2",
        help="Comma-separated Tiled access tags for the copied container.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Rows per Tiled PATCH call (default: 1).  "
            "WARNING: values > 1 or 0/None break the streaming pipeline — "
            "see module docstring."
        ),
    )
    p.add_argument(
        "--batch-delay",
        type=float,
        default=0.1,
        metavar="SECONDS",
        help="Sleep between frame writes to simulate acquisition rate (default: 0.1 s).",
    )
    p.add_argument(
        "--image-key",
        default="primary/pil900KW_image",
        metavar="KEY",
        help=(
            "Slash-separated key identifying the image array to copy incrementally, "
            "in the form '<stream>/<array_key>' (default: primary/pil900KW_image).  "
            "All other arrays are copied in one shot."
        ),
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    tiled_uri = settings.tiled_server_uri
    tiled_key = settings.tiled_api_key
    if not tiled_uri:
        sys.exit("EMBLASE_TILED_SERVER_URI is not set.")

    client = from_uri(tiled_uri, api_key=tiled_key)

    src_segments = [s for s in args.src.split("/") if s]
    dst_segments = [s for s in args.dst.split("/") if s]
    try:
        src_node = client[tuple(src_segments)]
    except KeyError:
        sys.exit(f"Source not found: {args.src}")
    try:
        dst_node = client[tuple(dst_segments)]
    except KeyError:
        sys.exit(f"Destination not found: {args.dst}")

    base = args.rename or (src_segments[-1] + "_live")
    ts = int(time.time())
    rename = f"{base}_{ts}"

    access_tags = [t.strip() for t in args.access_tags.split(",") if t.strip()] or None

    print(f"Source     : {args.src}")
    print(f"Destination: {args.dst}/{rename}")
    print(f"Image key  : {args.image_key}")
    print(f"batch_size : {args.batch_size}  batch_delay: {args.batch_delay} s")
    if access_tags:
        print(f"access_tags: {access_tags}")
    print()

    t_start = time.monotonic()
    _last_print = 0.0

    def _on_progress(written: int, total: int) -> None:
        nonlocal _last_print
        now = time.monotonic()
        elapsed = now - t_start
        pct = written / total * 100 if total else 0
        rate = written / elapsed if elapsed > 0 else 0
        eta = (total - written) / rate if rate > 0 else 0
        if total <= 30 or now - _last_print >= 1.0 or written == total:
            bar_filled = int(pct / 5)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            print(
                f"\r  [{bar}] {written}/{total} frames  "
                f"{pct:5.1f}%  {rate:.1f} fr/s  ETA {eta:.0f}s",
                end="",
                flush=True,
            )
            _last_print = now
        if written == total:
            print()

    deepcopy(
        src_node,
        dst_node,
        rename=rename,
        access_tags=access_tags,
        batch_size=args.batch_size,
        batch_delay=args.batch_delay,
        image_key=args.image_key,
        on_progress=_on_progress,
    )

    elapsed = time.monotonic() - t_start
    print(f"Done in {elapsed:.1f} s — run written to {args.dst}/{rename}")


if __name__ == "__main__":
    main()
