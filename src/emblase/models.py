"""Model loading utilities for VAE and ViT autoencoders."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from .config import settings


def _add_models_to_path() -> None:
    p = str(settings.models_dir)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_weights(model: torch.nn.Module, weights_path: Path) -> None:
    """Load weights from a .npz file into a model in-place (strict=False)."""
    if not weights_path.exists():
        warnings.warn(
            f"Weights not found at {weights_path} — using random initialisation"
        )
        return
    data = np.load(str(weights_path), allow_pickle=True)
    state_dict = {k: torch.from_numpy(data[k]) for k in data.files}
    model.load_state_dict(state_dict, strict=False)


def load_vae(
    latent_dim: int = 512,
    image_size: tuple[int, int] = (512, 512),
    weights_path: Path | None = None,
) -> torch.nn.Module:
    _add_models_to_path()
    from vae.vae import ConvVAE  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvVAE(latent_dim=latent_dim, image_size=image_size).to(device)
    _load_weights(
        model, weights_path or settings.models_dir / "vae" / "vae_model_512_weights.npz"
    )
    return model.eval()


def load_vit(
    latent_dim: int = 512,
    weights_path: Path | None = None,
) -> torch.nn.Module:
    _add_models_to_path()
    from vit.vit import Autoencoder  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(latent_dim=latent_dim).to(device)
    _load_weights(
        model, weights_path or settings.models_dir / "vit" / "vit_model_weights.npz"
    )
    return model.eval()


def load_model(model_name: str, **kwargs) -> torch.nn.Module:
    """Load a model by name ('vae' or 'vit'). image_size is ignored for ViT."""
    if model_name == "vae":
        return load_vae(**kwargs)
    if model_name == "vit":
        kwargs.pop("image_size", None)
        return load_vit(**kwargs)
    raise ValueError(f"Unknown model: {model_name!r}. Available: ['vae', 'vit']")


def encode(model: torch.nn.Module, images: torch.Tensor, model_name: str) -> np.ndarray:
    """Run the encoder and return output as (B, latent_dim) numpy array."""
    device = next(model.parameters()).device
    with torch.no_grad():
        if model_name == "vae":
            mu, _ = model.encode(images.to(device))
            return mu.cpu().numpy()
        if model_name == "vit":
            latent, _ = model.encoder(images.to(device))
            return latent.cpu().numpy()
    raise ValueError(f"Unknown model_name: {model_name!r}")
