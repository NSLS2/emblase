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
    path = (
        os.environ.get("EMBLASE_ORION_MODELS_DIR")
        or os.environ.get("EMBLASE_NERSC_MODELS_DIR")
        or ""
    )
    return Path(path) if path else Path.home() / ".cache" / "emblase" / "models"


def resolve_model_dir(name: str, models_dir: str, sentinel: str = "") -> Path:
    """Return the local directory for *name*, downloading from MLflow if needed.

    Local directory layout can be either:
    - Flat:      ``<models_dir>/<name>/<artifacts>``
    - Versioned: ``<models_dir>/<name>/v{N}/model/<artifacts>``  (MLflow download layout)

    When ``sentinel`` is given the directory is only accepted if the sentinel
    file exists directly inside it (flat layout) or inside any ``v*/model/``
    subdirectory (versioned layout).  This guards against stale empty dirs.

    Parameters
    ----------
    name:
        MLflow registered model name.
    models_dir:
        Root directory to search for pre-cached models.
    sentinel:
        A filename that must exist inside the resolved model directory for it to
        be considered valid (e.g. ``"label_names.json"``, ``"umap_approximator.pth"``).
        When empty, any existing directory is accepted.
    """
    from emblase import mlflow_registry  # noqa: PLC0415

    local = Path(models_dir) / name
    if local.is_dir():
        # Case 1: flat layout — artifacts directly inside the directory
        if not sentinel or (local / sentinel).exists():
            return local

        # Case 2: versioned layout — find the highest v*/model/ that has the sentinel
        versioned = sorted(
            (d / "model" for d in local.iterdir() if d.is_dir() and d.name.startswith("v")),
            key=lambda p: int(p.parent.name[1:]) if p.parent.name[1:].isdigit() else 0,
            reverse=True,
        )
        for candidate in versioned:
            if candidate.is_dir() and (not sentinel or (candidate / sentinel).exists()):
                log.info("Using versioned model dir %s", candidate)
                return candidate

        log.info(
            "%r directory exists but sentinel %r is missing — pulling from MLflow",
            name, sentinel,
        )

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
    import sys

    import joblib
    import torch

    model_dir = resolve_model_dir(projector_name, models_dir, sentinel="umap_approximator.pth")
    model_dir_str = str(model_dir)

    # neural_dimred_wrapper ships as an MLflow artifact alongside the model weights.
    # Insert the model directory into sys.path BEFORE importing it.
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)

    from neural_dimred_wrapper import SimpleDimRedApproximator  # type: ignore[import]  # noqa: PLC0415

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
    model_dir = resolve_model_dir(classifier_name, models_dir, sentinel="label_names.json")
    log.info("Classifier artefacts at %s", model_dir)
    return model_dir
