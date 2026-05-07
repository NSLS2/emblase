"""Legacy shim — use ``emblase.projector`` instead.

.. deprecated::
    ``emblase.umap`` is kept for backward compatibility only.
    All functionality has moved to :mod:`emblase.projector`.

    Replace::

        from emblase.umap import train_umap, apply_umap

    with::

        from emblase.projector import train_projector as train_umap, apply_projector as apply_umap
"""

from __future__ import annotations

import warnings

from emblase.projector import apply_projector as apply_umap
from emblase.projector import train_projector as train_umap

__all__ = ["train_umap", "apply_umap"]

warnings.warn(
    "emblase.umap is deprecated; use emblase.projector (train_projector / apply_projector) instead.",
    DeprecationWarning,
    stacklevel=2,
)
