"""Classifier training and application utilities.

Unsupervised cluster-based classifier: discovers label groups from embeddings
via clustering (KMeans or HDBSCAN), then trains an MLP to reproduce those
cluster assignments so that new embeddings can be labelled at inference time.

Labels are strings (e.g. ``"cluster_0"``, ``"cluster_1"``) and are meant to
be passed to ``append(..., labels=[...])`` when writing new embeddings to Tiled
— they are written once alongside the embedding row, never patched afterward.

Typical workflow from IPython:

    from emblase.classifier import train_classifier, apply_classifier

    node = client["smi/sandbox/confab26_demo/results/run_xyz"]

    # Discover clusters, train MLP, save model:
    labels = train_classifier(node, classifier_dir="models/classifier")

    # Apply a saved classifier to get label strings for new embeddings:
    labels = apply_classifier(embeddings, classifier_dir="models/classifier")

Model layout (flat directory)::

    classifier_dir/
        scaler.pkl          — sklearn StandardScaler (optional)
        label_names.json    — list of label strings indexed by class id
        classifier.pth      — MLP state dict
"""

from __future__ import annotations

import json
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

def _build_mlp(input_dim: int, hidden_dims: list[int], n_classes: int, device):
    """Build a classification MLP using torch."""
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers).to(device)


def _hidden_dims_from_state_dict(state_dict: dict) -> list[int]:
    """Infer hidden layer sizes from saved state dict."""
    weight_keys = sorted(k for k in state_dict if k.endswith(".weight"))
    return [state_dict[k].shape[0] for k in weight_keys[:-1]]


def _cluster_embeddings(
    X: np.ndarray,
    n_clusters: int | None,
    use_projections: bool,
    projections: np.ndarray | None,
    min_cluster_size: int,
    random_state: int,
) -> np.ndarray:
    """Run clustering and return integer label array of length N.

    Strategy:
    - If ``use_projections`` is True and valid projections are available,
      cluster on 2-D projector coordinates (faster, visually consistent).
    - Otherwise cluster on full-dim (scaled) embeddings.
    - If ``n_clusters`` is specified use KMeans; otherwise use HDBSCAN for
      automatic cluster count discovery.
    """
    if use_projections and projections is not None:
        valid_mask = ~np.isnan(projections).any(axis=1)
        if valid_mask.sum() < 10:
            log.warning("Too few valid projections — falling back to embedding clustering")
            X_cluster = X
            valid_mask = np.ones(len(X), dtype=bool)
        else:
            X_cluster = projections[valid_mask]
            log.info("Clustering on 2-D projections (%d valid points)", valid_mask.sum())
    else:
        X_cluster = X
        valid_mask = np.ones(len(X), dtype=bool)
        log.info("Clustering on full embeddings (%d dims)", X.shape[1])

    if n_clusters is not None:
        from sklearn.cluster import KMeans
        log.info("KMeans clustering: n_clusters=%d", n_clusters)
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        cluster_ids = km.fit_predict(X_cluster).astype(np.int64)
    else:
        try:
            import hdbscan
            log.info("HDBSCAN clustering: min_cluster_size=%d", min_cluster_size)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            cluster_ids = clusterer.fit_predict(X_cluster).astype(np.int64)
            # HDBSCAN uses -1 for noise; remap to a dedicated "noise" class
            cluster_ids[cluster_ids < 0] = cluster_ids.max() + 1
        except ImportError:
            log.warning("hdbscan not installed — falling back to KMeans(n_clusters=8)")
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=8, random_state=random_state, n_init="auto")
            cluster_ids = km.fit_predict(X_cluster).astype(np.int64)

    n_found = len(np.unique(cluster_ids))
    log.info("Clustering found %d clusters", n_found)

    if valid_mask.all():
        return cluster_ids

    # Map back to full-length array; fill unclustered rows with modal cluster
    full_labels = np.zeros(len(X), dtype=np.int64)
    full_labels[valid_mask] = cluster_ids
    modal = int(np.bincount(cluster_ids).argmax())
    full_labels[~valid_mask] = modal
    return full_labels


def _train_classifier_mlp(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    hidden_dims: list[int],
    epochs: int,
    lr: float,
    batch_size: int,
    device,
):
    """Train a classification MLP on (X, y) with cross-entropy loss."""
    import torch
    import torch.nn as nn

    model = _build_mlp(X.shape[1], hidden_dims, n_classes, device)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    log.info(
        "Training classifier MLP: input_dim=%d hidden=%s n_classes=%d "
        "epochs=%d lr=%.0e batch=%d device=%s",
        X.shape[1], hidden_dims, n_classes, epochs, lr, batch_size, device,
    )

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_correct = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            n_correct += (logits.argmax(1) == yb).sum().item()
        scheduler.step()
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            acc = n_correct / len(X)
            log.info("  epoch %4d/%d  loss=%.4f  acc=%.3f", epoch, epochs,
                     total_loss / len(X), acc)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_classifier(
    node: "LatentSpaceEmbedding",
    classifier_dir: str | Path = "models/classifier",
    *,
    n_clusters: int | None = None,
    use_projections: bool = True,
    min_cluster_size: int = 5,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    hidden_dims: list[int] | None = None,
    label_prefix: str = "cluster_",
    random_state: int = 42,
) -> np.ndarray:
    """Discover cluster-based labels and train an MLP classifier.

    Reads embeddings (and projections if available) from the Tiled node,
    clusters them to discover label groups, trains an MLP to reproduce those
    assignments, and saves the model artefacts to ``classifier_dir``.

    Returns a ``(N,)`` string array of predicted label names for the training
    set — these can be passed to ``append(..., labels=...)`` if you are
    writing the embeddings for the first time, or used for inspection.

    Parameters
    ----------
    node:
        An initialised ``LatentSpaceEmbedding`` client.
    classifier_dir:
        Directory to save model artefacts (``scaler.pkl``, ``label_names.json``,
        ``classifier.pth``).
    n_clusters:
        If set, use KMeans with this many clusters.  If ``None`` (default),
        use HDBSCAN for automatic discovery (requires the ``hdbscan`` package;
        falls back to KMeans(8) if not installed).
    use_projections:
        If ``True`` (default), cluster on 2-D projector coordinates rather than
        full embeddings when valid projections exist.
    min_cluster_size:
        HDBSCAN ``min_cluster_size`` (ignored for KMeans).
    epochs, lr, batch_size, hidden_dims:
        MLP training hyper-parameters.
    label_prefix:
        Prefix for auto-generated label names (e.g. ``"cluster_"`` →
        ``"cluster_0"``, ``"cluster_1"``, ...).
    random_state:
        Random seed for clustering and weight initialisation.

    Returns
    -------
    np.ndarray
        ``(N,)`` string array of predicted label names for all embeddings in
        the training node.
    """
    import joblib
    import torch
    from sklearn.preprocessing import StandardScaler

    if hidden_dims is None:
        hidden_dims = [128, 64]

    classifier_dir = Path(classifier_dir)
    classifier_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Reading embeddings from node (%d × %d)…",
             node.num_embeddings, node.embedding_dim)
    embeddings = node.read_embeddings()

    projections: np.ndarray | None = None
    if use_projections:
        try:
            projections = node.read_projections()
            if np.isnan(projections).all():
                projections = None
                log.info("Projections are all NaN — will cluster on embeddings")
        except Exception:
            projections = None
            log.info("Could not read projections — will cluster on embeddings")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings).astype(np.float32)
    log.info("Scaler fitted (mean~%.4f  std~%.4f)", X_scaled.mean(), X_scaled.std())

    cluster_ids = _cluster_embeddings(
        X_scaled,
        n_clusters=n_clusters,
        use_projections=use_projections,
        projections=projections,
        min_cluster_size=min_cluster_size,
        random_state=random_state,
    )

    unique_ids = sorted(np.unique(cluster_ids))
    n_classes = len(unique_ids)
    id_to_name = {cid: f"{label_prefix}{cid}" for cid in unique_ids}
    label_names = [id_to_name[i] for i in range(max(unique_ids) + 1)]
    log.info("Label names: %s", label_names)

    # Remap cluster_ids to be 0-based contiguous (in case HDBSCAN gaps exist)
    id_remap = {old: new for new, old in enumerate(unique_ids)}
    y = np.array([id_remap[cid] for cid in cluster_ids], dtype=np.int64)

    model = _train_classifier_mlp(
        X_scaled, y,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
    )

    scaler_path = classifier_dir / "scaler.pkl"
    names_path = classifier_dir / "label_names.json"
    weights_path = classifier_dir / "classifier.pth"

    joblib.dump(scaler, scaler_path)
    names_path.write_text(json.dumps(label_names, indent=2))
    torch.save(model.state_dict(), weights_path)
    log.info("Saved scaler      → %s", scaler_path)
    log.info("Saved label_names → %s", names_path)
    log.info("Saved weights     → %s", weights_path)

    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, dtype=torch.float32, device=device))
        pred_ids = logits.argmax(1).cpu().numpy()

    predicted_labels = np.array([label_names[i] for i in pred_ids])
    acc = (pred_ids == y).mean()
    log.info("Training accuracy: %.3f", acc)

    return predicted_labels


def apply_classifier(
    embeddings: np.ndarray,
    classifier_dir: str | Path = "models/classifier",
) -> np.ndarray:
    """Apply a saved classifier to a batch of embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        ``(N, D)`` float32 embedding array.
    classifier_dir:
        Directory containing ``classifier.pth``, ``label_names.json``, and
        optionally ``scaler.pkl``.

    Returns
    -------
    np.ndarray
        ``(N,)`` string array of predicted label names, ready to be passed to
        ``append(..., labels=list(predicted_labels))``.
    """
    import joblib
    import torch

    classifier_dir = Path(classifier_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler_path = classifier_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    if scaler is not None:
        log.info("Scaler loaded from %s", scaler_path)

    names_path = classifier_dir / "label_names.json"
    if not names_path.exists():
        raise FileNotFoundError(f"label_names.json not found in {classifier_dir}")
    label_names: list[str] = json.loads(names_path.read_text())
    n_classes = len(label_names)
    log.info("Labels (%d): %s", n_classes, label_names)

    weights_path = classifier_dir / "classifier.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"classifier.pth not found in {classifier_dir}")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    hidden_dims = _hidden_dims_from_state_dict(state_dict)
    first_weight = state_dict.get("0.weight")
    if first_weight is None:
        raise ValueError("Unexpected state_dict format — '0.weight' not found")
    input_dim = first_weight.shape[1]

    model = _build_mlp(input_dim, hidden_dims, n_classes, device)
    model.load_state_dict(state_dict)
    model.eval()
    log.info("Classifier MLP loaded: input_dim=%d hidden=%s n_classes=%d",
             input_dim, hidden_dims, n_classes)

    X = embeddings.astype("float32")
    if scaler is not None:
        X = scaler.transform(X)

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        pred_ids = logits.argmax(1).cpu().numpy()

    predicted_labels = np.array([label_names[i] for i in pred_ids])

    unique, counts = np.unique(predicted_labels, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        log.info("  %-20s  %d (%.1f%%)", lbl, cnt, 100 * cnt / len(predicted_labels))

    return predicted_labels
