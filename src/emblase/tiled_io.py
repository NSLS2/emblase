"""Tiled I/O utilities for reading input images and writing output arrays."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from tiled.ndslice import NDSlice

# Type alias: a tiled entry is either a bare path string or a (path, slice) tuple.
TiledEntry = str | tuple[str, str]


def read_images(
    client,
    entries: Sequence[TiledEntry],
) -> list[np.ndarray]:
    """Read tiled nodes and return a flat list of 2-D float32 numpy arrays.

    Parameters
    ----------
    client:
        An already-initialised tiled client pointing at the catalog root
        (e.g. ``tiled.client.from_uri(url, api_key=key)``).
    entries:
        Each item is either:
        - a ``str`` tiled path  →  ``client[path].read()``
        - a ``(path, slice)`` tuple  →  ``client[path].read(slice)``

        The tiled client resolves slash-separated paths natively (e.g.
        ``"proposal/scan"`` maps to ``client["proposal/scan"]``).

    Each node may contain:
    - A 2-D array ``(H, W)``     → one frame
    - An N-D array ``(..., H, W)`` → all frames (product of leading dims)

    Image dimensions may vary across entries.
    """
    frames: list[np.ndarray] = []
    for entry in entries:
        if isinstance(entry, str):
            path, slc = entry, None
        else:
            path, slc = entry

        slc = NDSlice.from_numpy_str(slc) if slc is not None else None
        arr = client[path].read(slc)

        if arr.ndim < 2:
            raise ValueError(
                f"Array at {path!r} has fewer than 2 dimensions: {arr.shape}"
            )
        for frame in arr.reshape(-1, arr.shape[-2], arr.shape[-1]):
            frames.append(frame)
    return frames


def write_output(
    client,
    path: str,
    array: np.ndarray,
    key: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Write ``array`` into the tiled container at ``path``.

    Parameters
    ----------
    client:
        An already-initialised tiled client pointing at the catalog root.
    path:
        Slash-separated path to the writable container node.
    """
    container = client[path]
    container.write_array(array, key=key, metadata=metadata or {}, access_tags=['smi_sandbox'])
