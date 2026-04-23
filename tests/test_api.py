"""Tests for the FastAPI app using TestClient with a mocked backend."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from emblase.compute.base import JobResult, JobStatus


@pytest.fixture
def app_with_mock_backend(monkeypatch):
    """Patch the global backend in app.py with a controllable fake."""
    import emblase.app as app_module

    class FakeBackend:
        async def submit(self, model_name, images, latent_dim=512, **kw):
            return "job-001"

        async def result(self, job_id):
            if job_id == "job-001":
                return JobResult(
                    job_id="job-001",
                    status=JobStatus.completed,
                    latent_vectors=np.array([[0.1, 0.2, 0.3]]),
                )
            raise KeyError(job_id)

        async def status(self, job_id):
            return JobStatus.completed

        async def cancel(self, job_id):
            if job_id != "job-001":
                raise KeyError(job_id)

    monkeypatch.setattr(app_module, "backend", FakeBackend())
    return TestClient(app_module.app)


def test_health(app_with_mock_backend):
    resp = app_with_mock_backend.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_evaluate_dummy_images(app_with_mock_backend):
    resp = app_with_mock_backend.post(
        "/evaluate", json={"dummy_images": 2, "image_size": [64, 64]}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "job-001"
    assert data["status"] == "completed"
    assert data["latent_vectors"] == [[0.1, 0.2, 0.3]]


def test_evaluate_no_input_returns_400(app_with_mock_backend):
    resp = app_with_mock_backend.post(
        "/evaluate", json={"images": None, "dummy_images": None}
    )
    assert resp.status_code == 400


def test_get_job_found(app_with_mock_backend):
    resp = app_with_mock_backend.get("/jobs/job-001")
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


def test_get_job_not_found(app_with_mock_backend):
    resp = app_with_mock_backend.get("/jobs/does-not-exist")
    assert resp.status_code == 404


def test_cancel_job(app_with_mock_backend):
    resp = app_with_mock_backend.delete("/jobs/job-001")
    assert resp.status_code == 200
    assert resp.json()["job_id"] == "job-001"


def test_cancel_job_not_found(app_with_mock_backend):
    resp = app_with_mock_backend.delete("/jobs/ghost")
    assert resp.status_code == 404
