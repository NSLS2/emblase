"""Tiled I/O utilities for reading input images and writing output arrays."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np


def _client(server_url: str | None = None):
    from tiled.client import from_uri
    url = server_url or os.environ["EMBLASE_TILED_SERVER_URL"]
    key = os.environ.get("EMBLASE_TILED_API_KEY") or None
    return from_uri(url, api_key=key)


def _resolve(client, path: str):
    """Walk a slash-separated path from a tiled root client."""
    node = client
    for part in path.strip("/").split("/"):
        node = node[part]
    return node


def read_images(uris: Sequence[str], server_url: str | None = None) -> list[np.ndarray]:
    """Read tiled node paths and return a flat list of 2-D float32 numpy arrays.

    Each URI may point to:
    - A 2-D array  ``(H, W)``     → one frame
    - An N-D array ``(..., H, W)`` → all frames (product of leading dims)

    Image dimensions may vary across URIs.
    """
    client = _client(server_url)
    frames: list[np.ndarray] = []
    for uri in uris:
        arr = np.asarray(_resolve(client, uri).read(), dtype=np.float32)
        if arr.ndim < 2:
            raise ValueError(f"Array at {uri!r} has fewer than 2 dimensions: {arr.shape}")
        for frame in arr.reshape(-1, arr.shape[-2], arr.shape[-1]):
            frames.append(frame)
    return frames


def write_output(
    container_uri: str,
    array: np.ndarray,
    key: str | None = None,
    metadata: dict | None = None,
    server_url: str | None = None,
) -> None:
    """Write ``array`` into the tiled container at ``container_uri``."""
    client = _client(server_url)
    container = _resolve(client, container_uri)
    container.write_array(array, key=key, metadata=metadata or {})
