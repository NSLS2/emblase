"""Model loading utilities for VAE and ViT autoencoders."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .config import settings


def _add_models_to_path():
    """Ensure the models directory is on sys.path so we can import them."""
    models_dir = str(settings.models_dir)
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)


def load_vae(
    latent_dim: int = 512,
    image_size: tuple[int, int] = (512, 512),
    weights_path: Path | None = None,
) -> torch.nn.Module:
    """Load the ConvVAE model with pre-trained weights."""
    _add_models_to_path()
    from vae.vae import ConvVAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvVAE(latent_dim=latent_dim, image_size=image_size).to(device)

    if weights_path is None:
        weights_path = settings.models_dir / "vae" / "vae_model_512_weights.npz"

    if weights_path.exists():
        data = np.load(str(weights_path), allow_pickle=True)
        state_dict = {}
        for key in data.files:
            state_dict[key] = torch.from_numpy(data[key])
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


def load_vit(
    latent_dim: int = 512,
    weights_path: Path | None = None,
) -> torch.nn.Module:
    """Load the ViT Autoencoder with pre-trained weights."""
    _add_models_to_path()
    from vit.vit import Autoencoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(latent_dim=latent_dim).to(device)

    if weights_path is None:
        weights_path = settings.models_dir / "vit" / "vit_model_weights.npz"

    if weights_path.exists():
        data = np.load(str(weights_path), allow_pickle=True)
        state_dict = {}
        for key in data.files:
            state_dict[key] = torch.from_numpy(data[key])
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


def load_model(model_name: str, **kwargs) -> torch.nn.Module:
    """Load a model by name."""
    loaders = {
        "vae": load_vae,
        "vit": load_vit,
    }
    if model_name not in loaders:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(loaders)}")
    return loaders[model_name](**kwargs)


def encode(model: torch.nn.Module, images: torch.Tensor, model_name: str) -> np.ndarray:
    """Run encoder forward pass, return latent vectors as numpy array.

    Args:
        model: Loaded model (VAE or ViT autoencoder)
        images: Tensor of shape (B, 1, H, W)
        model_name: "vae" or "vit" — needed because encode API differs

    Returns:
        Latent vectors as numpy array of shape (B, latent_dim)
    """
    device = next(model.parameters()).device
    images = images.to(device)

    with torch.no_grad():
        if model_name == "vae":
            mu, logvar = model.encode(images)
            return mu.cpu().numpy()
        elif model_name == "vit":
            latent, _skip = model.encoder(images)
            return latent.cpu().numpy()
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
