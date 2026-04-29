"""Noop model loader — for development and pipeline testing.

Returns deterministic (seeded) random embeddings without loading any weights
or touching the network.  Drop-in replacement for real models:

    pixi run orion infer --model noop --inputs ...

The seed is derived from the input tensor's shape so that:
  - Different input sizes give different (but reproducible) embeddings.
  - Re-running with the same inputs gives the same embeddings.

``model.latent_dim`` defaults to 512 and can be overridden via the
``NOOP_LATENT_DIM`` environment variable.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

_DEFAULT_LATENT_DIM = int(os.environ.get("NOOP_LATENT_DIM", 512))


class NoopEncoder(nn.Module):
    """Returns seeded random embeddings; no real computation performed."""

    def __init__(self, latent_dim: int = _DEFAULT_LATENT_DIM) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # Dummy parameter so encode() can discover the device via model.parameters()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode(self, x: torch.Tensor):
        """Return (mu, log_var) — log_var is always zeros."""
        batch = x.shape[0]
        seed = int(x.shape[-1]) * 1000 + int(x.shape[-2])
        rng = np.random.default_rng(seed)
        mu = torch.from_numpy(
            rng.standard_normal((batch, self.latent_dim)).astype(np.float32)
        ).to(x.device)
        log_var = torch.zeros_like(mu)
        return mu, log_var


def load(weights_path=None, **kwargs) -> NoopEncoder:
    latent_dim = int(os.environ.get("NOOP_LATENT_DIM", _DEFAULT_LATENT_DIM))
    model = NoopEncoder(latent_dim=latent_dim)
    model.eval()
    return model
