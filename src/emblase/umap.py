"""UMAP training and application utilities.

Two public functions usable from IPython with a pre-initialised Tiled node:

    from emblase.umap import train_umap, apply_umap

    node = client["smi/sandbox/.../results/run_xyz"]

    # Fit a new UMAP approximator and write projections back to Tiled:
    train_umap(node, umap_dir="models/umap_approx")

    # Apply an existing approximator and write projections back to Tiled:
    apply_umap(node, umap_dir="models/umap_approx")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from emblase.tiled.client import LatentSpaceEmbedding

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_approximator(umap_dir: str | Path):
    """Load scaler + MLP from the umap_approx directory.

    Expected flat layout::

        umap_dir/
            neural_dimred_wrapper.py   — SimpleDimRedApproximator class
            umap_approximator.pth      — MLP state dict
            scaler.pkl                 — sklearn StandardScaler (optional)

    Returns ``(model, scaler, device)``.
    """
    import joblib
    import torch

    umap_dir = Path(umap_dir)
    umap_dir_str = str(umap_dir)
    if umap_dir_str not in sys.path:
        sys.path.insert(0, umap_dir_str)
    from neural_dimred_wrapper import SimpleDimRedApproximator  # noqa: E402

    scaler_path = umap_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    if scaler is not None:
        log.info("Scaler loaded from %s", scaler_path)
    else:
        log.info("No scaler found at %s — using raw embeddings", scaler_path)

    weights_path = umap_dir / "umap_approximator.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading MLP weights from %s (device=%s)", weights_path, device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    first_weight = state_dict.get("network.0.weight")
    if first_weight is None:
        raise ValueError("Unexpected state_dict format — 'network.0.weight' not found")
    input_dim = first_weight.shape[1]

    model = SimpleDimRedApproximator(input_dim=input_dim)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    log.info("MLP loaded: input_dim=%d → 2", input_dim)

    return model, scaler, device


def _apply_approximator(model, scaler, device, embeddings: np.ndarray) -> np.ndarray:
    """Apply scaler then MLP; return ``(N, 2)`` float32 array."""
    import torch

    X = embeddings.astype("float32")
    if scaler is not None:
        X = scaler.transform(X)
    with torch.no_grad():
        coords = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    return coords.astype("float32")


def _fit_umap(X_scaled: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    """Fit umap-learn and return ``(N, 2)`` float32 coordinates."""
    import umap as umap_lib

    log.info("Fitting UMAP (n_neighbors=%d, min_dist=%.2f) on %d points…",
             n_neighbors, min_dist, len(X_scaled))
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        verbose=False,
    )
    coords = reducer.fit_transform(X_scaled).astype(np.float32)
    log.info("UMAP done. x=[%.3f, %.3f]  y=[%.3f, %.3f]",
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
    device,
    umap_dir: Path,
):
    """Train SimpleDimRedApproximator to regress UMAP coords Y from scaled embeddings X."""
    import torch
    import torch.nn as nn

    umap_dir_str = str(umap_dir)
    if umap_dir_str not in sys.path:
        sys.path.insert(0, umap_dir_str)
    from neural_dimred_wrapper import SimpleDimRedApproximator

    model = SimpleDimRedApproximator(
        input_dim=input_dim, hidden_dims=hidden_dims, output_dim=2
    ).to(device)

    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, Y_t),
        batch_size=batch_size,
        shuffle=True,
    )

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
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            log.info("  epoch %4d/%d  loss=%.6f", epoch, epochs, total_loss / len(X_scaled))

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_umap(
    node: "LatentSpaceEmbedding",
    umap_dir: str | Path = "models/umap_approx",
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    hidden_dims: list[int] | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    write_projections: bool = True,
) -> np.ndarray:
    """Fit a UMAP approximator on embeddings from a ``LatentSpaceEmbedding`` node.

    Steps:

    1. Read all embeddings from the Tiled node.
    2. Fit ``StandardScaler``.
    3. Run ``umap-learn`` to get reference 2-D coordinates.
    4. Train ``SimpleDimRedApproximator`` (MLP) to regress those coordinates.
    5. Save ``scaler.pkl`` and ``umap_approximator.pth`` to ``umap_dir``.
    6. Optionally write the final projections back to Tiled via
       ``node.update_projections()``.

    Parameters
    ----------
    node:
        An initialised ``LatentSpaceEmbedding`` client, e.g.
        ``client["smi/sandbox/.../results/run_xyz"]``.
    umap_dir:
        Directory to save (and load ``neural_dimred_wrapper.py`` from).
        Defaults to ``models/umap_approx`` relative to the current working
        directory.
    epochs, lr, batch_size, hidden_dims:
        MLP training hyper-parameters.
    n_neighbors, min_dist:
        ``umap-learn`` hyper-parameters.
    write_projections:
        If ``True`` (default), call ``node.update_projections()`` with the
        result after training.

    Returns
    -------
    np.ndarray
        ``(N, 2)`` float32 projection coordinates.
    """
    import joblib
    import torch
    from sklearn.preprocessing import StandardScaler

    if hidden_dims is None:
        hidden_dims = [128, 64]

    umap_dir = Path(umap_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Reading embeddings from node (%d × %d)…",
             node.num_embeddings, node.embedding_dim)
    embeddings = node.read_embeddings()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings).astype(np.float32)
    log.info("Scaler fitted (mean~%.4f  std~%.4f)", X_scaled.mean(), X_scaled.std())

    umap_coords = _fit_umap(X_scaled, n_neighbors=n_neighbors, min_dist=min_dist)

    model = _train_mlp(
        X_scaled, umap_coords,
        input_dim=embeddings.shape[1],
        hidden_dims=hidden_dims,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        umap_dir=umap_dir,
    )

    scaler_path = umap_dir / "scaler.pkl"
    weights_path = umap_dir / "umap_approximator.pth"
    joblib.dump(scaler, scaler_path)
    torch.save(model.state_dict(), weights_path)
    log.info("Saved scaler  → %s", scaler_path)
    log.info("Saved weights → %s", weights_path)

    with torch.no_grad():
        projections = model(
            torch.tensor(X_scaled, dtype=torch.float32, device=device)
        ).cpu().numpy().astype(np.float32)

    log.info("Projections  x=[%.3f, %.3f]  y=[%.3f, %.3f]",
             projections[:, 0].min(), projections[:, 0].max(),
             projections[:, 1].min(), projections[:, 1].max())

    if write_projections:
        log.info("Writing projections to Tiled…")
        node.update_projections(projections)
        log.info("Done.")

    return projections


def apply_umap(
    node: "LatentSpaceEmbedding",
    umap_dir: str | Path = "models/umap_approx",
    *,
    write_projections: bool = True,
) -> np.ndarray:
    """Apply a saved UMAP approximator to a ``LatentSpaceEmbedding`` node.

    Reads all embeddings, runs them through the saved scaler + MLP, and
    optionally writes the result back to Tiled.

    Parameters
    ----------
    node:
        An initialised ``LatentSpaceEmbedding`` client.
    umap_dir:
        Directory containing ``neural_dimred_wrapper.py``,
        ``umap_approximator.pth``, and optionally ``scaler.pkl``.
    write_projections:
        If ``True`` (default), call ``node.update_projections()`` with the
        result.

    Returns
    -------
    np.ndarray
        ``(N, 2)`` float32 projection coordinates.
    """
    log.info("Reading embeddings from node (%d × %d)…",
             node.num_embeddings, node.embedding_dim)
    embeddings = node.read_embeddings()

    model, scaler, device = _load_approximator(umap_dir)

    log.info("Computing projections…")
    projections = _apply_approximator(model, scaler, device, embeddings)
    log.info("Projections  x=[%.3f, %.3f]  y=[%.3f, %.3f]",
             projections[:, 0].min(), projections[:, 0].max(),
             projections[:, 1].min(), projections[:, 1].max())

    if write_projections:
        log.info("Writing projections to Tiled…")
        node.update_projections(projections)
        log.info("Done.")

    return projections
