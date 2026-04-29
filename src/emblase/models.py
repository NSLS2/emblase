"""Model loading utilities."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch

from .config import settings

log = logging.getLogger(__name__)


def _add_models_to_path() -> None:
    p = str(settings.models_dir)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_model(model_name: str, **kwargs) -> torch.nn.Module:
    """Load a model by name.

    If ``settings.models_dir / model_name`` is a directory containing a
    ``loader.py``, the model is loaded locally via that loader.  Otherwise
    weights are pulled from the MLflow registry under ``model_name``.

    Accepted kwargs: ``mlflow_version``, ``mlflow_tracking_uri``,
    ``mlflow_api_key``, ``cache_dir``.  Architecture parameters (latent_dim,
    image_size) are the responsibility of each model's loader.py.
    """
    if (settings.models_dir / model_name).is_dir():
        log.info("Loading %r from local model directory", model_name)
        return _load_local(model_name, **kwargs)
    log.info("Loading %r from MLflow registry", model_name)
    return _load_from_mlflow(model_name, **kwargs)


def _load_local(model_name: str, **kwargs) -> torch.nn.Module:
    _add_models_to_path()
    model_dir = settings.models_dir / model_name
    loader_path = model_dir / "loader.py"
    if not loader_path.exists():
        raise FileNotFoundError(
            f"No loader.py found in {model_dir}. "
            "Each model directory must contain a loader.py with a load(weights_path) function."
        )
    log.debug("Importing loader from %s", loader_path)
    spec = importlib.util.spec_from_file_location(f"{model_name}.loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.load(weights_path=kwargs.get("weights_path"))
    log.info("Loaded local model %r", model_name)
    return model


def _load_from_mlflow(model_name: str, **kwargs) -> torch.nn.Module:
    from . import mlflow_registry

    tracking_uri = kwargs.get("mlflow_tracking_uri")
    api_key = kwargs.get("mlflow_api_key")
    requested_version = kwargs.get("mlflow_version")

    cache_root = kwargs.get("cache_dir") or Path(
        os.environ.get("EMBLASE_MODEL_CACHE_DIR")
        or Path.home() / ".cache" / "emblase" / "models"
    )

    # Cache-first: if no specific version is requested, scan the local cache
    # for existing versions and use the highest one — no MLflow network call.
    # Only hit MLflow when the cache is empty or a specific version is pinned.
    version: str | None = None
    if requested_version is not None:
        # Explicit version requested — resolve (no-op, just stringifies) then check cache.
        version = mlflow_registry.resolve_version(
            model_name, version=requested_version,
            tracking_uri=tracking_uri, api_key=api_key,
        )
    else:
        # Look for any cached versions: <cache_root>/<model_name>/v<N>/model/loader.py
        model_cache = Path(cache_root) / model_name
        cached_versions = sorted(
            int(p.parent.parent.name[1:])
            for p in model_cache.glob("v*/model/loader.py")
            if p.parent.parent.name[1:].isdigit()
        )
        if cached_versions:
            version = str(max(cached_versions))
            log.info(
                "Cache-first: using cached '%s' v%s (skip MLflow API call)",
                model_name, version,
            )
        else:
            # No cache — must resolve from MLflow.
            version = mlflow_registry.resolve_version(
                model_name, version=None,
                tracking_uri=tracking_uri, api_key=api_key,
            )

    # weights_dir is always <cache_root>/<model_name>/v<version>/model/
    # matching the artifact subpath MLflow uses on download.
    weights_dir = Path(cache_root) / model_name / f"v{version}" / "model"

    if (weights_dir / "loader.py").exists():
        log.info("Cache hit: '%s' v%s at %s", model_name, version, weights_dir)
    else:
        log.info("Cache miss: downloading '%s' v%s → %s", model_name, version, weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        mlflow_registry.download_model_weights(
            model_name=model_name,
            version=version,
            dest_dir=weights_dir.parent,
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
    model = module.load(weights_path=weights_path)
    log.info("Loaded MLflow model %r v%s", model_name, version)
    return model


def encode(
    model: torch.nn.Module,
    images: Union[torch.Tensor, list[np.ndarray]],
    batch_size: int = 1,
) -> np.ndarray:
    """Run the encoder and return output as a ``(B, latent_dim)`` numpy array.

    ``images`` may be:

    - A ``(B, C, H, W)`` tensor — uniform shape, processed in chunks of
      ``batch_size``.
    - A list of ``(H, W)`` numpy arrays — varying shapes allowed, always
      processed one frame at a time regardless of ``batch_size``.

    Default ``batch_size=1`` is safe for any image size; increase for
    throughput when images are small and uniform.
    """
    device = next(model.parameters()).device
    results = []
    input_size: tuple[int, int] | None = getattr(model, "input_size", None)
    if input_size:
        log.info("Model expects input size %s — images will be resized if needed", input_size)

    def _resize(t: torch.Tensor) -> torch.Tensor:
        """Resize-with-aspect-ratio then pad to model.input_size.

        Scale so the longer side fits target_H / target_W (whichever is the
        binding constraint), then zero-pad the shorter dimension to reach the
        exact target size.  If the image already matches, return it unchanged.
        """
        if input_size is None:
            return t
        import torch.nn.functional as F
        target_H, target_W = input_size
        _, _, h, w = t.shape
        if h == target_H and w == target_W:
            return t

        # Scale so the image fits inside target_H × target_W
        scale = min(target_H / h, target_W / w)
        new_h = round(h * scale)
        new_w = round(w * scale)
        resized = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)

        # Zero-pad to reach exact target size (pad right / bottom)
        pad_bottom = target_H - new_h
        pad_right = target_W - new_w
        padded = F.pad(resized, (0, pad_right, 0, pad_bottom))

        log.debug(
            "Resized %s → %s then padded to %s",
            (h, w), (new_h, new_w), (target_H, target_W),
        )
        return padded

    def _forward(chunk: torch.Tensor) -> torch.Tensor:
        chunk = _resize(chunk)
        if hasattr(model, "encode"):
            mu, _ = model.encode(chunk)
            return mu.cpu()
        if hasattr(model, "encoder"):
            latent, _ = model.encoder(chunk)
            return latent.cpu()
        raise ValueError(f"Model {type(model).__name__!r} has neither .encode() nor .encoder()")

    with torch.no_grad():
        if isinstance(images, list):
            # Ragged batch: each (H, W) frame → (1, 1, H, W) tensor
            log.info("Encoding %d ragged frame(s) one at a time", len(images))
            for i, frame in enumerate(images):
                t = torch.from_numpy(np.asarray(frame, dtype=np.float32))[None, None].to(device)
                results.append(_forward(t))
                if (i + 1) % 10 == 0 or (i + 1) == len(images):
                    log.info("  encoded %d / %d", i + 1, len(images))
        else:
            n_batches = (len(images) + batch_size - 1) // batch_size
            log.info("Encoding tensor %s in %d batch(es) of %d", tuple(images.shape), n_batches, batch_size)
            for i in range(0, len(images), batch_size):
                results.append(_forward(images[i : i + batch_size].to(device)))

    return torch.cat(results).numpy()
