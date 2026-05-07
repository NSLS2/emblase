"""Shared helpers for worker scripts (inference.py.tmpl, streaming_inference.py.tmpl).

These functions are imported by the rendered worker scripts at runtime on the
compute node, so they must only use dependencies available in the pixi env.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("emblase.worker")


def _mlflow_env() -> dict[str, str | None]:
    return {
        "tracking_uri": os.environ.get("EMBLASE_MLFLOW_TRACKING_URI") or None,
        "api_key": os.environ.get("EMBLASE_MLFLOW_API_KEY") or None,
    }


def _cache_root() -> Path:
    return Path(
        os.environ.get("EMBLASE_MODEL_CACHE_DIR", "")
        or Path.home() / ".cache" / "emblase" / "models"
    )


def resolve_model_dir(name: str, models_dir: str) -> Path:
    """Return the local directory for *name*, downloading from MLflow if needed.

    Checks ``<models_dir>/<name>`` first.  On a miss, resolves the latest
    version from MLflow and downloads weights to the model cache dir.
    """
    from emblase import mlflow_registry  # noqa: PLC0415

    local = Path(models_dir) / name
    if local.is_dir():
        return local

    log.info("%r not found locally — pulling from MLflow", name)
    env = _mlflow_env()
    version = mlflow_registry.resolve_version(name, **env)
    dest = _cache_root() / name / f"v{version}" / "model"
    if not any(dest.iterdir()) if dest.exists() else True:
        mlflow_registry.download_model_weights(
            model_name=name, version=version, dest_dir=dest.parent, **env
        )
    return dest


def load_projector(
    projector_name: str,
    models_dir: str,
    device: Any,
) -> tuple[Any, Any]:
    """Load a neural UMAP approximator.

    Returns ``(model, scaler)`` where *scaler* may be ``None``.
    Raises on failure — caller should catch and fall back to NaN projections.
    """
    import joblib
    import torch
    from neural_dimred_wrapper import SimpleDimRedApproximator  # type: ignore[import]

    model_dir = resolve_model_dir(projector_name, models_dir)
    model_dir_str = str(model_dir)
    import sys
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)

    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    sd = torch.load(model_dir / "umap_approximator.pth", map_location=device, weights_only=True)
    weight_keys = sorted(k for k in sd if k.startswith("network.") and k.endswith(".weight"))
    hidden_dims = [sd[k].shape[0] for k in weight_keys[:-1]]
    proj_model = SimpleDimRedApproximator(
        input_dim=sd["network.0.weight"].shape[1], hidden_dims=hidden_dims
    )
    proj_model.load_state_dict(sd)
    proj_model.eval().to(device)
    log.info("Projector loaded from %s", model_dir)
    return proj_model, scaler


def load_classifier_dir(classifier_name: str, models_dir: str) -> Path:
    """Resolve and return the local directory for a classifier model."""
    model_dir = resolve_model_dir(classifier_name, models_dir)
    log.info("Classifier artefacts at %s", model_dir)
    return model_dir
