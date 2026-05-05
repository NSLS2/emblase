"""Train an unsupervised cluster-based classifier on embeddings from a LatentSpaceEmbedding dataset.

Steps:
  1. Read embeddings (and projections if available) from Tiled.
  2. Fit StandardScaler on the embeddings.
  3. Cluster scaled embeddings (or 2-D projections) to discover label groups.
  4. Train an MLP classifier to reproduce those cluster assignments.
  5. Save scaler.pkl, label_names.json, classifier.pth to --classifier-dir.
  6. Write predicted labels to the 'label' column of _index in Tiled.

The core logic lives in ``emblase.classifier.train_classifier`` and can be called directly
from IPython with a pre-initialised Tiled node::

    from emblase.classifier import train_classifier
    node = client["smi/sandbox/confab26_demo/results/run_xyz"]
    train_classifier(node, classifier_dir="models/classifier")

Usage
-----
    python scripts/train_classifier.py \\
        --dataset smi/sandbox/confab26_demo/results/run_live_1086139_1777749128 \\
        --classifier-dir models/classifier \\
        [--n-clusters 8] \\
        [--use-projections / --no-use-projections] \\
        [--epochs 100] [--lr 1e-3] [--batch-size 64]
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
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True, metavar="TILED_PATH",
        help="Tiled path to the LatentSpaceEmbedding container.",
    )
    parser.add_argument(
        "--classifier-dir",
        default=str(Path(__file__).parent.parent / "models" / "classifier"),
        metavar="DIR",
        help="Directory to save classifier artefacts (default: models/classifier).",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of clusters for KMeans. "
            "If omitted, HDBSCAN is used for automatic discovery (requires hdbscan package)."
        ),
    )
    parser.add_argument(
        "--use-projections",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cluster on 2-D projector coordinates rather than full embeddings "
            "when valid projections exist (default: True)."
        ),
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        metavar="N",
        help="HDBSCAN min_cluster_size (default: 5, ignored for KMeans).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-dims",
        default="128,64",
        help="Comma-separated hidden layer sizes (default: 128,64).",
    )
    parser.add_argument(
        "--label-prefix",
        default="cluster_",
        help="Prefix for auto-generated label names (default: 'cluster_').",
    )
    args = parser.parse_args()

    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    from emblase.config import settings
    from tiled.client import from_uri
    from emblase.classifier import train_classifier

    tiled_uri = settings.tiled_server_uri
    tiled_key = settings.tiled_api_key
    if not tiled_uri:
        sys.exit("EMBLASE_TILED_SERVER_URI is not set.")

    client = from_uri(tiled_uri, api_key=tiled_key)
    segments = tuple(s for s in args.dataset.split("/") if s)
    node = client[segments]

    train_classifier(
        node,
        classifier_dir=args.classifier_dir,
        n_clusters=args.n_clusters,
        use_projections=args.use_projections,
        min_cluster_size=args.min_cluster_size,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dims=[int(x) for x in args.hidden_dims.split(",")],
        label_prefix=args.label_prefix,
    )


if __name__ == "__main__":
    main()
