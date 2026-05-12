"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from emblase.compute.base import JobStatus
from emblase.schemas import EvaluateRequest, EvaluateResponse, ModelName


def test_evaluate_request_with_images():
    images = [[[0.1, 0.2], [0.3, 0.4]]]  # 1 x 2 x 2
    req = EvaluateRequest(model="vit", images=images, dummy_images=None)
    assert req.model == ModelName.vit
    assert req.images is not None


def test_evaluate_request_with_output():
    req = EvaluateRequest(output="results/scan1")
    assert req.output == "results/scan1"


def test_evaluate_request_invalid_model():
    with pytest.raises(ValidationError):
        EvaluateRequest(model="unknown")


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
