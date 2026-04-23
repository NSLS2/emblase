"""Tests for LocalBackend with mocked model loading."""

import numpy as np
import pytest

from emblase.compute.base import JobStatus
from emblase.compute.local import LocalBackend

DUMMY_IMAGES = np.random.rand(2, 64, 64).astype(np.float32)
DUMMY_LATENT = np.random.rand(2, 8).astype(np.float32)


@pytest.fixture
def backend_with_mock_models(monkeypatch):
    """Patch load_model and encode so no real model weights are needed."""

    class FakeModel:
        pass

    def fake_load_model(model_name, **kwargs):
        return FakeModel()

    def fake_encode(model, tensor, model_name):
        # Return a fixed latent array matching batch size
        return DUMMY_LATENT[: tensor.shape[0]]

    monkeypatch.setattr("emblase.models.load_model", fake_load_model)
    monkeypatch.setattr("emblase.models.encode", fake_encode)
    return LocalBackend()


@pytest.mark.asyncio
async def test_submit_returns_job_id(backend_with_mock_models):
    job_id = await backend_with_mock_models.submit("vae", DUMMY_IMAGES, latent_dim=8)
    assert isinstance(job_id, str)
    assert len(job_id) > 0


@pytest.mark.asyncio
async def test_submit_completed_with_output_data(backend_with_mock_models):
    job_id = await backend_with_mock_models.submit("vae", DUMMY_IMAGES, latent_dim=8)
    result = await backend_with_mock_models.result(job_id)
    assert result.status == JobStatus.completed
    assert result.output_data is not None
    assert result.output_data.shape[0] == DUMMY_IMAGES.shape[0]


@pytest.mark.asyncio
async def test_status_after_submit(backend_with_mock_models):
    job_id = await backend_with_mock_models.submit("vae", DUMMY_IMAGES, latent_dim=8)
    status = await backend_with_mock_models.status(job_id)
    assert status == JobStatus.completed


@pytest.mark.asyncio
async def test_cancel_sets_failed(backend_with_mock_models):
    job_id = await backend_with_mock_models.submit("vae", DUMMY_IMAGES, latent_dim=8)
    await backend_with_mock_models.cancel(job_id)
    result = await backend_with_mock_models.result(job_id)
    assert result.status == JobStatus.failed
    assert result.error == "Cancelled"


@pytest.mark.asyncio
async def test_submit_records_error_on_failure(monkeypatch):
    """If model loading raises, the job should record a failed status."""

    def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr("emblase.models.load_model", boom)

    backend = LocalBackend()
    job_id = await backend.submit("vae", DUMMY_IMAGES, latent_dim=8)
    result = await backend.result(job_id)
    assert result.status == JobStatus.failed
    assert "model exploded" in result.error


@pytest.mark.asyncio
async def test_result_unknown_job_raises():
    backend = LocalBackend()
    with pytest.raises(KeyError):
        await backend.result("nonexistent")
