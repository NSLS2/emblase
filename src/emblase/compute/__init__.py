"""Compute backends and shared CLI utilities.

Top-level imports are kept minimal so that ``import emblase`` (or importing
the Tiled router) does not drag in torch / httpx / etc.  The concrete backend
classes are imported lazily — either via :func:`build_backend` or by importing
them directly from their submodules (e.g.
``from emblase.compute.orion import OrionBackend``).
"""

from __future__ import annotations

import sys

from .base import ComputeBackend, JobResult, JobStatus

__all__ = [
    "ComputeBackend",
    "JobStatus",
    "JobResult",
    "LocalBackend",
    "NERSCBackend",
    "NERSCClient",
    "NERSCJob",
    "OrionClient",
    "OrionJob",
    "OrionBackend",
    "parse_param_specs",
    "build_backend",
]


def __getattr__(name: str):
    """Lazily import backend classes on first attribute access."""
    if name in ("LocalBackend",):
        from .local import LocalBackend

        return LocalBackend
    if name in ("OrionBackend", "OrionClient", "OrionJob"):
        from . import orion as _orion

        return getattr(_orion, name)
    if name in ("NERSCBackend", "NERSCClient", "NERSCJob"):
        from . import nersc as _nersc

        return getattr(_nersc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def parse_param_specs(param_strs: list[str]) -> dict | None:
    """Parse ``--param name:source[:dtype[:units]]`` CLI strings into a ParamSpec dict.

    Returns ``None`` when *param_strs* is empty.  Exits with an error message
    on malformed input (intended for use in CLI entry-points).
    """
    if not param_strs:
        return None
    specs: dict = {}
    for s in param_strs:
        parts = s.split(":")
        if len(parts) < 2:
            sys.exit(f"Invalid --param spec {s!r}: expected name:source[:dtype[:units]]")
        specs[parts[0]] = {
            "source": parts[1],
            "dtype": parts[2] if len(parts) > 2 else "float",
            "units": parts[3] if len(parts) > 3 else "",
        }
    return specs


def build_backend(name: str) -> ComputeBackend:
    """Instantiate a compute backend by name (``"orion"``, ``"local"``, ``"nersc"``)."""
    if name == "orion":
        from .orion import OrionBackend

        return OrionBackend()
    if name == "local":
        from .local import LocalBackend

        return LocalBackend()
    if name == "nersc":
        from .nersc import NERSCBackend

        return NERSCBackend()
    sys.exit(f"Unknown backend: {name!r}  (choices: orion, local, nersc)")
