"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from emblase.compute.base import JobStatus
from emblase.schemas import EvaluateRequest, EvaluateResponse, ModelName


def test_evaluate_request_defaults():
    req = EvaluateRequest()
    assert req.model == ModelName.vae
    assert req.dummy_images == 4
    assert req.image_size == (512, 512)
    assert req.latent_dim == 512
    assert req.images is None


def test_evaluate_request_with_images():
    images = [[[0.1, 0.2], [0.3, 0.4]]]  # 1 x 2 x 2
    req = EvaluateRequest(model="vit", images=images, dummy_images=None)
    assert req.model == ModelName.vit
    assert req.images is not None


def test_evaluate_request_invalid_model():
    with pytest.raises(ValidationError):
        EvaluateRequest(model="unknown")


def test_evaluate_response_jobstatus_is_shared_enum():
    """schemas.JobStatus must be the same object as compute.base.JobStatus."""
    from emblase.schemas import JobStatus as SchemasJobStatus

    assert SchemasJobStatus is JobStatus


def test_evaluate_response_roundtrip():
    resp = EvaluateResponse(
        job_id="abc123",
        status=JobStatus.completed,
        output_data=[[0.1, 0.2, 0.3]],
        error=None,
    )
    data = resp.model_dump()
    assert data["status"] == "completed"
    assert data["output_data"] == [[0.1, 0.2, 0.3]]
