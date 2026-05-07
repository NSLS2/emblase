"""Compute backends and shared CLI utilities."""

from __future__ import annotations

import sys

from .base import ComputeBackend, JobResult, JobStatus
from .local import LocalBackend
from .orion import OrionBackend, OrionClient, OrionJob

__all__ = [
    "ComputeBackend",
    "JobStatus",
    "JobResult",
    "LocalBackend",
    "OrionClient",
    "OrionJob",
    "OrionBackend",
    "parse_param_specs",
    "build_backend",
]


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
        return OrionBackend()
    if name == "local":
        return LocalBackend()
    if name == "nersc":
        from .nersc import NERSCBackend  # optional dependency

        return NERSCBackend()
    sys.exit(f"Unknown backend: {name!r}  (choices: orion, local, nersc)")
