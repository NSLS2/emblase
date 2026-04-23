"""Tests for OrionBackend — script rendering and sbatch construction."""

import base64
import io

import numpy as np
import pytest

from emblase.compute.orion import (
    OrionBackend,
    _build_sbatch_script,
    _render_inference_script,
)

DUMMY_IMAGES = np.zeros((2, 512, 512), dtype=np.float32)


def test_render_inference_script_substitutes_all_placeholders():
    script = _render_inference_script(
        model_name="vae",
        latent_dim=512,
        image_size=(512, 512),
        models_dir="/models",
    )
    assert "/models" in script
    assert 'model_name = "vae"' in script
    assert "latent_dim = 512" in script
    assert "image_size = (512, 512)" in script
    assert 'models_dir = "/models"' in script
    # job_dir is now read from JOB_DIR env var at runtime, not substituted
    assert 'os.environ["JOB_DIR"]' in script


def test_render_inference_script_vit():
    script = _render_inference_script(
        model_name="vit",
        latent_dim=256,
        image_size=(224, 224),
        models_dir="/models",
    )
    assert "vit" in script
    assert "256" in script


def test_build_sbatch_script_structure():
    script = _build_sbatch_script(
        working_dir="/jobs",
        python_script="print('hello')",
        images_b64="AABBCC==",
        project_dir="/code/emblase",
        job_name="emblase-vae",
        time_limit="0-00:10:00",
    )
    assert script.startswith("#!/bin/bash")
    assert "#SBATCH --job-name=emblase-vae" in script
    assert "#SBATCH --time=0-00:10:00" in script
    assert "#SBATCH --output=/jobs/slurm-%j.out" in script
    assert "job_$" in script  # $SLURM_JOB_ID used for dir name
    assert "AABBCC==" in script
    assert "EMBLASE_INFERENCE_EOF" in script
    assert "print('hello')" in script
    assert "pixi run python" in script


def test_images_b64_roundtrip():
    """Base64-encoded images must survive the encode/decode cycle."""
    buf = io.BytesIO()
    np.save(buf, DUMMY_IMAGES)
    b64 = base64.b64encode(buf.getvalue()).decode()

    recovered = np.load(io.BytesIO(base64.b64decode(b64)))
    np.testing.assert_array_equal(recovered, DUMMY_IMAGES)


@pytest.mark.asyncio
async def test_orion_backend_submit_calls_client(monkeypatch):
    """submit() should call OrionClient.submit_job and store job metadata."""
    submitted = {}

    class FakeClient:
        async def submit_job(self, script, working_dir, overrides=None, environment=None):
            submitted["script"] = script
            submitted["working_dir"] = working_dir
            submitted["overrides"] = overrides
            submitted["environment"] = environment
            return 99

    backend = OrionBackend(
        client=FakeClient(),
        working_dir="/jobs",
        models_dir="/models",
        account="staff",
    )
    job_id = await backend.submit("vae", DUMMY_IMAGES, latent_dim=512)

    assert job_id == "99"
    assert 99 in backend._jobs
    assert backend._jobs[99]["model_name"] == "vae"
    assert "emblase-vae" in submitted["script"]
    assert submitted["overrides"]["account"] == "staff"
