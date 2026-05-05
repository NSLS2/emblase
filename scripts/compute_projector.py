"""Apply a saved projector approximator to a LatentSpaceEmbedding container.

Reads all embeddings, runs them through the saved scaler + MLP, and writes
the ``(N, 2)`` projection coordinates back via ``update_projections()``.

The core logic lives in ``emblase.projector.apply_projector`` and can be called directly
from IPython with a pre-initialised Tiled node::

    from emblase.projector import apply_projector
    node = client["smi/sandbox/confab26_demo/results/run_xyz"]
    apply_projector(node, projector_dir="models/umap_approx")

Usage
-----
    python scripts/compute_projector.py \\
        --dataset smi/sandbox/confab26_demo/results/run_live_1086139_1777749128 \\
        --projector-dir models/umap_approx
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, metavar="TILED_PATH",
                        help="Tiled path to the LatentSpaceEmbedding container.")
    parser.add_argument(
        "--projector-dir",
        default=str(Path(__file__).parent.parent / "models" / "umap_approx"),
        metavar="DIR",
        help="Directory containing the projector model artifacts (default: models/umap_approx).",
    )
    args = parser.parse_args()

    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    from emblase.config import settings
    from tiled.client import from_uri
    from emblase.projector import apply_projector

    tiled_uri = settings.tiled_server_uri
    tiled_key = settings.tiled_api_key
    if not tiled_uri:
        sys.exit("EMBLASE_TILED_SERVER_URI is not set.")

    client = from_uri(tiled_uri, api_key=tiled_key)
    segments = tuple(s for s in args.dataset.split("/") if s)
    node = client[segments]

    apply_projector(node, projector_dir=args.projector_dir)


if __name__ == "__main__":
    main()
