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
        models_dir="/models",
        output_mode="none",
        tiled_result_path="",
    )
    assert "/models" in script
    assert 'model_name = "vae"' in script
    assert 'models_dir = "/models"' in script
    assert 'output_mode = "none"' in script
    assert 'os.environ["JOB_DIR"]' in script


def test_render_inference_script_vit():
    script = _render_inference_script(
        model_name="vit",
        models_dir="/models",
        output_mode="tiled",
        tiled_result_path="results/job1",
    )
    assert "vit" in script
    assert "results/job1" in script


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
    assert "job_$" in script
    assert "AABBCC==" in script
    assert "EMBLASE_INFERENCE_EOF" in script
    assert "print('hello')" in script
    assert "pixi run python" in script


def test_build_sbatch_script_from_path():
    script = _build_sbatch_script(
        working_dir="/jobs",
        python_script="print('hello')",
        image_path="/data/images.npy",
        project_dir="/code/emblase",
        job_name="emblase-vae",
    )
    assert "ln -sf /data/images.npy" in script
    assert "EMBLASE_B64_EOF" not in script


def test_build_sbatch_script_from_tiled():
    script = _build_sbatch_script(
        working_dir="/jobs",
        python_script="print('hello')",
        tiled_uris=["path/to/scan1", "path/to/scan2"],
        project_dir="/code/emblase",
        job_name="emblase-vae",
    )
    assert "path/to/scan1" in script
    assert "path/to/scan2" in script
    assert "EMBLASE_TILED_URI" in script
    assert "EMBLASE_B64_EOF" not in script
    assert "ln -sf" not in script


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
        async def submit_job(
            self, script, working_dir, overrides=None, environment=None
        ):
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
    job_id = await backend.submit(
        model_name="vae",
        image_data=DUMMY_IMAGES,
        image_size=(512, 512),
        latent_dim=512,
    )

    assert job_id == "99"
    assert 99 in backend._jobs
    assert backend._jobs[99]["model_name"] == "vae"
    assert "emblase-vae" in submitted["script"]
    assert submitted["overrides"]["account"] == "staff"


@pytest.mark.asyncio
async def test_orion_backend_submit_image_path():
    """image_path source should produce a symlink line in the script."""
    submitted = {}

    class FakeClient:
        async def submit_job(
            self, script, working_dir, overrides=None, environment=None
        ):
            submitted["script"] = script
            return 7

    backend = OrionBackend(
        client=FakeClient(), working_dir="/jobs", models_dir="/models", account="staff"
    )
    await backend.submit(
        model_name="vae",
        image_path="/data/remote.npy",
        image_size=(512, 512),
    )
    assert "ln -sf /data/remote.npy" in submitted["script"]


@pytest.mark.asyncio
async def test_orion_backend_tiled_output_injects_env(monkeypatch):
    """output_mode='tiled' should add EMBLASE_TILED_URI to the job environment."""
    submitted = {}

    class FakeClient:
        async def submit_job(
            self, script, working_dir, overrides=None, environment=None
        ):
            submitted["environment"] = environment
            return 8

    import emblase.compute.orion as orion_module

    monkeypatch.setattr(orion_module.settings, "tiled_uri", "http://tiled")
    monkeypatch.setattr(orion_module.settings, "tiled_api_key", "key123")

    backend = OrionBackend(
        client=FakeClient(), working_dir="/jobs", models_dir="/models", account="staff"
    )
    await backend.submit(
        model_name="vae",
        image_data=DUMMY_IMAGES,
        image_size=(512, 512),
        output_mode="tiled",
        tiled_result_path="results/scan1",
    )

    env = submitted["environment"]
    assert any("EMBLASE_TILED_URI=http://tiled" in e for e in env)
    assert any("EMBLASE_TILED_API_KEY=key123" in e for e in env)


def test_build_sbatch_script_raises_without_source():
    with pytest.raises(ValueError, match="exactly one"):
        _build_sbatch_script(
            working_dir="/jobs",
            python_script="pass",
            project_dir="/code",
            job_name="test",
        )


def test_build_sbatch_script_raises_with_two_sources():
    with pytest.raises(ValueError, match="exactly one"):
        _build_sbatch_script(
            working_dir="/jobs",
            python_script="pass",
            project_dir="/code",
            job_name="test",
            images_b64="abc",
            image_path="/data/f.npy",
        )
