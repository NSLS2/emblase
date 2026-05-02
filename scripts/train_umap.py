"""Train the parametric UMAP approximator on embeddings from a LatentSpaceEmbedding dataset.

Steps:
  1. Read embeddings from Tiled.
  2. Fit StandardScaler on the embeddings.
  3. Run pure umap-learn to get reference 2-D coordinates.
  4. Train the MLP (SimpleDimRedApproximator) to regress those coordinates.
  5. Save updated scaler.pkl + umap_approximator.pth back to --umap-dir.
  6. Re-run the approximator over all embeddings and write projections to Tiled.

Usage
-----
    python scripts/train_umap.py \\
        --dataset smi/sandbox/confab26_demo/results/run_live_1086139_1777749128 \\
        --umap-dir models/umap_approx \\
        [--epochs 200] [--lr 1e-3] [--batch-size 64]
        [--umap-n-neighbors 15] [--umap-min-dist 0.1]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_umap")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_embeddings(dataset: str) -> np.ndarray:
    """Read all embeddings from a LatentSpaceEmbedding Tiled node."""
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    from emblase.config import settings
    from tiled.client import from_uri

    tiled_uri = settings.tiled_server_uri
    tiled_key = settings.tiled_api_key
    if not tiled_uri:
        sys.exit("EMBLASE_TILED_SERVER_URI is not set.")

    client = from_uri(tiled_uri, api_key=tiled_key)
    segments = tuple(s for s in dataset.split("/") if s)
    node = client[segments]
    log.info("Reading embeddings from %s (%d × %d)…",
             dataset, node.num_embeddings, node.embedding_dim)
    return node["embeddings"].read(), client, node


def _run_umap(X_scaled: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    import umap
    log.info("Fitting UMAP (n_neighbors=%d, min_dist=%.2f) on %d points…",
             n_neighbors, min_dist, len(X_scaled))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        verbose=False,
    )
    coords = reducer.fit_transform(X_scaled).astype(np.float32)
    log.info("UMAP done. Range x=[%.3f, %.3f]  y=[%.3f, %.3f]",
             coords[:, 0].min(), coords[:, 0].max(),
             coords[:, 1].min(), coords[:, 1].max())
    return coords


def _train_mlp(
    X_scaled: np.ndarray,
    Y: np.ndarray,
    input_dim: int,
    hidden_dims: list[int],
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> nn.Module:
    """Train SimpleDimRedApproximator to regress UMAP coords Y from scaled embeddings X."""
    umap_dir = Path(__file__).parent.parent / "models" / "umap_approx"
    sys.path.insert(0, str(umap_dir))
    from neural_dimred_wrapper import SimpleDimRedApproximator

    model = SimpleDimRedApproximator(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=2)
    model = model.to(device)

    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    log.info("Training MLP: input_dim=%d hidden=%s epochs=%d lr=%.0e batch=%d device=%s",
             input_dim, hidden_dims, epochs, lr, batch_size, device)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            log.info("  epoch %4d/%d  loss=%.6f", epoch, epochs, total_loss / len(X_scaled))

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, metavar="TILED_PATH")
    parser.add_argument(
        "--umap-dir",
        default=str(Path(__file__).parent.parent / "models" / "umap_approx"),
        metavar="DIR",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dims", default="128,64",
                        help="Comma-separated hidden layer sizes (default: 128,64)")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    args = parser.parse_args()

    umap_dir = Path(args.umap_dir)
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # 1. Read embeddings
    embeddings, client, node = _read_embeddings(args.dataset)

    # 2. Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings).astype(np.float32)
    log.info("Scaler fitted (mean~%.4f std~%.4f)", X_scaled.mean(), X_scaled.std())

    # 3. Run pure UMAP
    umap_coords = _run_umap(X_scaled, args.umap_n_neighbors, args.umap_min_dist)

    # 4. Train MLP approximator
    model = _train_mlp(
        X_scaled, umap_coords,
        input_dim=embeddings.shape[1],
        hidden_dims=hidden_dims,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
    )

    # 5. Save scaler + weights
    scaler_path = umap_dir / "scaler.pkl"
    weights_path = umap_dir / "umap_approximator.pth"
    joblib.dump(scaler, scaler_path)
    torch.save(model.state_dict(), weights_path)
    log.info("Saved scaler → %s", scaler_path)
    log.info("Saved weights → %s", weights_path)

    # 6. Compute final projections with trained model and write to Tiled
    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        projections = model(X_t).cpu().numpy().astype(np.float32)

    log.info("Final projections range x=[%.3f, %.3f]  y=[%.3f, %.3f]",
             projections[:, 0].min(), projections[:, 0].max(),
             projections[:, 1].min(), projections[:, 1].max())

    log.info("Writing projections to Tiled…")
    node.update_projections(projections)
    log.info("Done.")


if __name__ == "__main__":
    main()
