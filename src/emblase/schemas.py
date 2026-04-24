"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from .compute.base import JobStatus  # single source of truth

__all__ = [
    "ModelName",
    "OutputMode",
    "JobStatus",
    "EvaluateRequest",
    "EvaluateResponse",
]


class ModelName(str, Enum):
    vae = "vae"
    vit = "vit"


class OutputMode(str, Enum):
    """How the compute node should deliver results.

    - ``none``  — save output.npy on the node only (retrieve manually via the
                  Orion filesystem).
    - ``tiled`` — write the output array into a Tiled collection at
                  EMBLASE_TILED_URI / tiled_result_path on the compute node.
    """

    none = "none"
    tiled = "tiled"


class EvaluateRequest(BaseModel):
    """Request to run dimensionality reduction on images."""

    model: ModelName = ModelName.vae
    images: list[list[list[float]]] | None = None
    dummy_images: int | None = 4
    image_size: tuple[int, int] = (512, 512)
    latent_dim: int = 512
    output_mode: OutputMode = OutputMode.none
    # Path within the Tiled server where results are written.
    # Only used when output_mode == "tiled".
    tiled_result_path: str | None = None
    # Registered MLflow model name.  When set, weights are pulled from the
    # registry on the compute node instead of loaded from local models_dir.
    mlflow_model: str | None = None
    # MLflow model version to use.  None → latest.
    mlflow_version: str | None = None


class EvaluateResponse(BaseModel):
    job_id: str
    status: JobStatus
    output_data: list[list[float]] | None = None
    error: str | None = None
