"""Model loading utilities."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

from .config import settings


def _add_models_to_path() -> None:
    p = str(settings.models_dir)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_model(model_name: str, **kwargs) -> torch.nn.Module:
    """Load a model by name.

    If ``settings.models_dir / model_name`` is a directory containing a
    ``loader.py``, the model is loaded locally via that loader.  Otherwise
    weights are pulled from the MLflow registry under ``model_name`` and
    the same local loader is used.

    Accepted kwargs: ``latent_dim``, ``image_size``, ``weights_path``,
    ``mlflow_version``, ``mlflow_tracking_uri``, ``mlflow_api_key``, ``dest_dir``.
    """
    if (settings.models_dir / model_name).is_dir():
        return _load_local(model_name, **kwargs)
    return _load_from_mlflow(model_name, **kwargs)


def _load_local(model_name: str, **kwargs) -> torch.nn.Module:
    _add_models_to_path()
    model_dir = settings.models_dir / model_name
    loader_path = model_dir / "loader.py"
    if not loader_path.exists():
        raise FileNotFoundError(
            f"No loader.py found in {model_dir}. "
            "Each model directory must contain a loader.py with a load(**kwargs) function."
        )
    spec = importlib.util.spec_from_file_location(f"{model_name}.loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load(**kwargs)


def _load_from_mlflow(model_name: str, **kwargs) -> torch.nn.Module:
    from . import mlflow_registry

    tracking_uri = kwargs.get("mlflow_tracking_uri")
    api_key = kwargs.get("mlflow_api_key")

    version = mlflow_registry.resolve_version(
        model_name,
        version=kwargs.get("mlflow_version"),
        tracking_uri=tracking_uri,
        api_key=api_key,
    )

    # Cache dir: <cache_root>/<model_name>/v<version>/
    # Stable across jobs on the same node/filesystem.  Falls back to a
    # system temp dir if no explicit cache_root is provided.
    cache_root = kwargs.get("cache_dir") or Path(
        os.environ.get("EMBLASE_MODEL_CACHE_DIR")
        or Path.home() / ".cache" / "emblase" / "models"
    )
    # weights_dir is always <cache_root>/<model_name>/v<version>/model/
    # matching the artifact subpath MLflow uses on download.
    weights_dir = Path(cache_root) / model_name / f"v{version}" / "model"

    if (weights_dir / "loader.py").exists():
        print(f"Using cached '{model_name}' v{version} from {weights_dir}")
    else:
        weights_dir.mkdir(parents=True, exist_ok=True)
        mlflow_registry.download_model_weights(
            model_name=model_name,
            version=version,
            dest_dir=weights_dir.parent,  # download into v<version>/, MLflow appends "model/"
            tracking_uri=tracking_uri,
            api_key=api_key,
        )

    loader_path = weights_dir / "loader.py"
    if not loader_path.exists():
        raise FileNotFoundError(
            f"No loader.py found in {weights_dir}. "
            "Ensure the model directory was pushed with a loader.py."
        )
    npz_files = sorted(weights_dir.rglob("*.npz"))
    weights_path = npz_files[0] if npz_files else None

    # Add the downloaded dir to sys.path so loader.py can import its siblings.
    p = str(weights_dir)
    if p not in sys.path:
        sys.path.insert(0, p)

    spec = importlib.util.spec_from_file_location(f"{model_name}.loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load(
        weights_path=weights_path,
        latent_dim=kwargs.get("latent_dim", 512),
        image_size=kwargs.get("image_size", (512, 512)),
    )


def encode(model: torch.nn.Module, images: torch.Tensor) -> np.ndarray:
    """Run the encoder and return output as a (B, latent_dim) numpy array."""
    device = next(model.parameters()).device
    images = images.to(device)
    with torch.no_grad():
        if hasattr(model, "encode"):
            mu, _ = model.encode(images)
            return mu.cpu().numpy()
        if hasattr(model, "encoder"):
            latent, _ = model.encoder(images)
            return latent.cpu().numpy()
    raise ValueError(f"Model {type(model).__name__!r} has neither .encode() nor .encoder()")
