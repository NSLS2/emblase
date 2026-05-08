"""Model loading utilities."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError as _torch_err:  # noqa: F841
    torch = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


def _add_models_to_path(models_dir: Path) -> None:
    p = str(models_dir)
    if p not in sys.path:
        sys.path.insert(0, p)


def _local_models_dir() -> Path:
    """Return the local models directory for the current process.

    On compute nodes the caller always passes ``models_dir`` explicitly — this
    fallback is only used when running locally (tests, CLI without --models-dir).
    Prefers EMBLASE_ORION_MODELS_DIR / EMBLASE_NERSC_MODELS_DIR env vars (set
    by the job environment) then falls back to the repo ``models/`` directory.
    """
    import os as _os

    path = _os.environ.get("EMBLASE_ORION_MODELS_DIR") or _os.environ.get(
        "EMBLASE_NERSC_MODELS_DIR"
    )
    if path:
        return Path(path)
    return Path(__file__).resolve().parent.parent.parent / "models"


def load_model(model_name: str, **kwargs) -> torch.nn.Module:
    """Load a model by name.

    If ``models_dir / model_name`` is a directory containing a ``loader.py``,
    the model is loaded locally via that loader.  Otherwise weights are pulled
    from the MLflow registry under ``model_name``.

    Accepted kwargs: ``mlflow_version``, ``mlflow_tracking_uri``,
    ``mlflow_api_key``, ``cache_dir``, ``models_dir``.  Architecture parameters
    (latent_dim, image_size) are the responsibility of each model's loader.py.
    """
    models_dir = Path(kwargs.pop("models_dir", None) or _local_models_dir())
    if (models_dir / model_name).is_dir():
        log.info("Loading %r from local model directory", model_name)
        return _load_local(model_name, models_dir=models_dir, **kwargs)
    log.info("Loading %r from MLflow registry", model_name)
    return _load_from_mlflow(model_name, models_dir=models_dir, **kwargs)


def _load_local(model_name: str, models_dir: Path, **kwargs) -> torch.nn.Module:
    _add_models_to_path(models_dir)
    model_dir = models_dir / model_name
    # Ensure the model's own directory is on sys.path so that loader.py can do
    # relative imports like `from vit import Autoencoder` without conflicts with
    # any installed package sharing the same name.
    model_dir_str = str(model_dir)
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)
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


def _load_from_mlflow(model_name: str, models_dir: Path, **kwargs) -> torch.nn.Module:
    from . import mlflow_registry

    tracking_uri = kwargs.get("mlflow_tracking_uri")
    api_key = kwargs.get("mlflow_api_key")
    requested_version = kwargs.get("mlflow_version")

    cache_root = kwargs.get("cache_dir") or models_dir

    # Cache-first: if no specific version is requested, scan the local cache
    # for existing versions and use the highest one — no MLflow network call.
    # Only hit MLflow when the cache is empty or a specific version is pinned.
    version: str | None = None
    if requested_version is not None:
        # Explicit version requested — resolve (no-op, just stringifies) then check cache.
        version = mlflow_registry.resolve_version(
            model_name,
            version=requested_version,
            tracking_uri=tracking_uri,
            api_key=api_key,
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
                model_name,
                version,
            )
        else:
            # No cache — must resolve from MLflow.
            version = mlflow_registry.resolve_version(
                model_name,
                version=None,
                tracking_uri=tracking_uri,
                api_key=api_key,
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
    images: torch.Tensor | list[np.ndarray],
    batch_size: int = 1,
) -> np.ndarray:
    """Run the encoder and return output as a ``(B, latent_dim)`` numpy array.

    ``images`` may be:

    - A ``(B, C, H, W)`` tensor — processed in chunks of ``batch_size``.
    - A list of ``(H, W)`` numpy arrays:
      - If all frames share the same shape they are stacked into a tensor
        and processed in batches of ``batch_size`` (fast path).
      - If shapes differ they are processed one at a time (ragged path).
    """
    device = next(model.parameters()).device
    results = []
    input_size: tuple[int, int] | None = getattr(model, "input_size", None)

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
            (h, w),
            (new_h, new_w),
            (target_H, target_W),
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
            shapes = {f.shape for f in images}
            if len(shapes) == 1:
                # All frames have the same shape — stack into a tensor and batch.
                arr = np.stack([np.asarray(f, dtype=np.float32) for f in images])
                # (N, H, W) → (N, 1, H, W)
                tensor = torch.from_numpy(arr[:, None]).to(device)
                n_batches = (len(images) + batch_size - 1) // batch_size
                log.info(
                    "Encoding %d frames %s in %d batch(es) of %d",
                    len(images),
                    tuple(shapes)[0],
                    n_batches,
                    batch_size,
                )
                for i in range(0, len(tensor), batch_size):
                    results.append(_forward(tensor[i : i + batch_size]))
            else:
                # Truly ragged — must process one at a time.
                log.info(
                    "Encoding %d ragged frames (%d distinct shapes) one at a time",
                    len(images),
                    len(shapes),
                )
                for i, frame in enumerate(images):
                    t = torch.from_numpy(np.asarray(frame, dtype=np.float32))[None, None].to(device)
                    results.append(_forward(t))
        else:
            n_batches = (len(images) + batch_size - 1) // batch_size
            log.info(
                "Encoding tensor %s in %d batch(es) of %d",
                tuple(images.shape),
                n_batches,
                batch_size,
            )
            for i in range(0, len(images), batch_size):
                results.append(_forward(images[i : i + batch_size].to(device)))

    return torch.cat(results).numpy()
