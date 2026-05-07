"""Train the parametric UMAP approximator on embeddings from a LatentSpaceEmbedding dataset.

Steps:
  1. Read embeddings from Tiled.
  2. Fit StandardScaler on the embeddings.
  3. Run pure umap-learn to get reference 2-D coordinates.
  4. Train the MLP (SimpleDimRedApproximator) to regress those coordinates.
  5. Save updated scaler.pkl + umap_approximator.pth back to --umap-dir.
  6. Write final projections back to Tiled via update_projections().

The core logic lives in ``emblase.umap.train_umap`` and can be called directly
from IPython with a pre-initialised Tiled node::

    from emblase.umap import train_umap
    node = client["smi/sandbox/confab26_demo/results/run_xyz"]
    train_umap(node, umap_dir="models/umap_approx")

Usage
-----
    python scripts/train_umap.py \\
        --dataset smi/sandbox/confab26_demo/results/run_live_1086139_1777749128 \\
        --umap-dir models/umap_approx \\
        [--epochs 200] [--lr 1e-3] [--batch-size 64] \\
        [--umap-n-neighbors 15] [--umap-min-dist 0.1]
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        required=True,
        metavar="TILED_PATH",
        help="Tiled path to the LatentSpaceEmbedding container.",
    )
    parser.add_argument(
        "--umap-dir",
        default=str(Path(__file__).parent.parent / "models" / "umap_approx"),
        metavar="DIR",
        help="Directory to save scaler + MLP weights (default: models/umap_approx).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-dims",
        default="128,64",
        help="Comma-separated hidden layer sizes (default: 128,64).",
    )
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    args = parser.parse_args()

    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    from tiled.client import from_uri

    from emblase.config import settings
    from emblase.umap import train_umap

    tiled_uri = settings.tiled_server_uri
    tiled_key = settings.tiled_api_key
    if not tiled_uri:
        sys.exit("EMBLASE_TILED_SERVER_URI is not set.")

    client = from_uri(tiled_uri, api_key=tiled_key)
    segments = tuple(s for s in args.dataset.split("/") if s)
    node = client[segments]

    train_umap(
        node,
        umap_dir=args.umap_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dims=[int(x) for x in args.hidden_dims.split(",")],
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
    )


if __name__ == "__main__":
    main()
