"""Compute UMAP projections for a LatentSpaceEmbedding dataset and write them back.

Uses the parametric UMAP approximator (a small MLP trained to approximate UMAP),
loaded directly from the weights + scaler artifacts — no MLflow pyfunc overhead.

Usage
-----
    python scripts/compute_umap.py \\
        --dataset smi/sandbox/confab26_demo/results/run_live_1086139_1777749128 \\
        --umap-dir /path/to/models/umap_approx

The script:
  1. Reads all embeddings from the LatentSpaceEmbedding container.
  2. Applies StandardScaler (if present) then passes through the MLP.
  3. Writes the (N, 2) projection coordinates back via update_projections().
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
log = logging.getLogger("compute_umap")


def _load_approximator(umap_dir: str):
    """Load scaler + MLP from the umap_approx directory.

    Expected layout (flat):
        umap_dir/
            neural_dimred_wrapper.py   — SimpleDimRedApproximator class
            umap_approximator.pth      — MLP state dict
            scaler.pkl                 — sklearn StandardScaler (optional)

    Returns (model, scaler, device).
    """
    import joblib
    import torch

    umap_dir = Path(umap_dir)

    # Make neural_dimred_wrapper importable
    sys.path.insert(0, str(umap_dir))
    from neural_dimred_wrapper import SimpleDimRedApproximator  # noqa: E402

    # Load scaler
    scaler_path = umap_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    if scaler is not None:
        log.info("Scaler loaded from %s", scaler_path)
    else:
        log.warning("No scaler found at %s — using raw embeddings", scaler_path)

    # Load weights
    weights_path = umap_dir / "umap_approximator.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading MLP weights from %s (device=%s)", weights_path, device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # Infer input_dim from first layer weight
    first_weight = state_dict.get("network.0.weight")
    if first_weight is None:
        raise ValueError("Unexpected state_dict format — 'network.0.weight' not found")
    input_dim = first_weight.shape[1]
    log.info("Inferred input_dim=%d from weights", input_dim)

    model = SimpleDimRedApproximator(input_dim=input_dim)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    log.info("MLP loaded: %d → hidden → 2", input_dim)

    return model, scaler, device


def _transform(model, scaler, device, embeddings):
    """Apply scaler then MLP; return (N, 2) numpy array."""
    import numpy as np
    import torch

    X = embeddings.astype("float32")
    if scaler is not None:
        X = scaler.transform(X)
    with torch.no_grad():
        coords = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    return coords.astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True,
        metavar="TILED_PATH",
        help="Tiled path to the LatentSpaceEmbedding container.",
    )
    parser.add_argument(
        "--umap-dir",
        default=str(Path(__file__).parent.parent / "models" / "umap_approx"),
        metavar="DIR",
        help="Directory containing the umap_approx model (artifacts/ and code/).",
    )
    args = parser.parse_args()

    # Set SSL cert for macOS / InCommon CA
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    from emblase.config import settings
    from tiled.client import from_uri
    from emblase.tiled.client import LatentSpaceEmbedding

    tiled_uri = settings.tiled_server_uri
    tiled_key = settings.tiled_api_key
    if not tiled_uri:
        sys.exit("EMBLASE_TILED_SERVER_URI is not set.")

    client = from_uri(tiled_uri, api_key=tiled_key)
    segments = [s for s in args.dataset.split("/") if s]
    node = client[tuple(segments)]

    if not isinstance(node, LatentSpaceEmbedding):
        sys.exit(f"Node at {args.dataset!r} is not a LatentSpaceEmbedding (got {type(node).__name__})")

    n = node.num_embeddings
    log.info("Dataset: %s  (%d embeddings, dim=%d)", args.dataset, n, node.embedding_dim)

    log.info("Reading embeddings...")
    embeddings = node.read_embeddings()
    log.info("Embeddings shape: %s", embeddings.shape)

    model, scaler, device = _load_approximator(args.umap_dir)

    log.info("Computing projections...")
    projections = _transform(model, scaler, device, embeddings)
    log.info("Projections shape: %s  range x=[%.3f, %.3f]  y=[%.3f, %.3f]",
             projections.shape,
             projections[:, 0].min(), projections[:, 0].max(),
             projections[:, 1].min(), projections[:, 1].max())

    log.info("Writing projections back to Tiled...")
    node.update_projections(projections)
    log.info("Done.")


if __name__ == "__main__":
    main()
