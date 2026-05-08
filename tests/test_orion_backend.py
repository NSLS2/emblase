"""Tests for OrionBackend — script rendering and sbatch construction."""

import base64
import io

import numpy as np
import pytest

from emblase.compute.orion import (
    OrionBackend,
    _build_sbatch_script,
    _render_inference_script,
    _render_streaming_inference_script,
)

DUMMY_IMAGES = np.zeros((2, 512, 512), dtype=np.float32)


def test_render_inference_script_substitutes_all_placeholders():
    script = _render_inference_script(
        model_name="vae",
        models_dir="/models",
        inputs=[],
        output="",
    )
    assert "/models" in script
    assert '"vae"' in script
    assert "inputs          = []" in script
    assert 'output          = ""' in script
    assert 'os.environ.get("JOB_DIR"' in script
    assert "param_specs     = " in script
    assert "projector_mode  = " in script
    assert "projector_name  = " in script


def test_render_inference_script_with_output():
    script = _render_inference_script(
        model_name="vit",
        models_dir="/models",
        inputs=[],
        output="results/job1",
    )
    assert "vit" in script
    assert "results/job1" in script


def test_build_sbatch_script_structure():
    script = _build_sbatch_script(
        working_dir="/jobs",
        python_script="print('hello')",
        payload_b64="AABBCC==",
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


def test_build_sbatch_script_from_npy_path():
    script = _build_sbatch_script(
        working_dir="/jobs",
        python_script="print('hello')",
        npy_path="/data/images.npy",
        project_dir="/code/emblase",
        job_name="emblase-vae",
    )
    assert "ln -sf /data/images.npy" in script
    assert "EMBLASE_B64_EOF" not in script


def test_build_sbatch_script_from_tiled():
    # tiled_input=True means no input pre-fetch section in the sbatch preamble.
    script = _build_sbatch_script(
        working_dir="/jobs",
        python_script="print('hello')",
        tiled_input=True,
        project_dir="/code/emblase",
        job_name="emblase-vae",
    )
    assert "EMBLASE_B64_EOF" not in script
    assert "ln -sf" not in script
    assert "print('hello')" in script


def test_payload_b64_roundtrip():
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
    job_id = await backend.submit(
        model_name="vae",
        images=DUMMY_IMAGES,
    )

    assert job_id == "99"
    assert 99 in backend._jobs
    assert backend._jobs[99]["model_name"] == "vae"
    assert "emblase-vae" in submitted["script"]
    assert submitted["overrides"]["account"] == "staff"


@pytest.mark.asyncio
async def test_orion_backend_submit_npy_path():
    """npy_path source should produce a symlink line in the script."""
    submitted = {}

    class FakeClient:
        async def submit_job(self, script, working_dir, overrides=None, environment=None):
            submitted["script"] = script
            return 7

    backend = OrionBackend(
        client=FakeClient(), working_dir="/jobs", models_dir="/models", account="staff"
    )
    await backend.submit(
        model_name="vae",
        npy_path="/data/remote.npy",
    )
    assert "ln -sf /data/remote.npy" in submitted["script"]


@pytest.mark.asyncio
async def test_orion_backend_output_injects_tiled_env(monkeypatch):
    """output being set should add EMBLASE_TILED_SERVER_URI to the job environment."""
    submitted = {}

    class FakeClient:
        async def submit_job(self, script, working_dir, overrides=None, environment=None):
            submitted["environment"] = environment
            return 8

    import emblase.compute.orion as orion_module

    monkeypatch.setattr(orion_module.settings, "tiled_server_uri", "http://tiled")
    monkeypatch.setattr(orion_module.settings, "tiled_api_key", "key123")

    backend = OrionBackend(
        client=FakeClient(), working_dir="/jobs", models_dir="/models", account="staff"
    )
    await backend.submit(
        model_name="vae",
        images=DUMMY_IMAGES,
        output="results/scan1",
    )

    env = submitted["environment"]
    assert any("EMBLASE_TILED_SERVER_URI=http://tiled" in e for e in env)
    assert any("EMBLASE_TILED_API_KEY=key123" in e for e in env)


@pytest.mark.asyncio
async def test_orion_backend_inputs_injects_tiled_env(monkeypatch):
    """inputs alone should still inject Tiled env vars."""
    submitted = {}

    class FakeClient:
        async def submit_job(self, script, working_dir, overrides=None, environment=None):
            submitted["environment"] = environment
            return 9

    import emblase.compute.orion as orion_module

    monkeypatch.setattr(orion_module.settings, "tiled_server_uri", "http://tiled")
    monkeypatch.setattr(orion_module.settings, "tiled_api_key", "key123")

    backend = OrionBackend(
        client=FakeClient(), working_dir="/jobs", models_dir="/models", account="staff"
    )
    await backend.submit(
        model_name="vae",
        inputs=["proposal/scan"],
    )

    env = submitted["environment"]
    assert any("EMBLASE_TILED_SERVER_URI=http://tiled" in e for e in env)
    assert any("EMBLASE_TILED_API_KEY=key123" in e for e in env)


@pytest.mark.asyncio
async def test_orion_backend_tiled_missing_url_raises(monkeypatch):
    """submit() with inputs but no tiled_server_uri configured must raise."""

    class FakeClient:
        async def submit_job(self, **kwargs):
            return 10

    import emblase.compute.orion as orion_module

    monkeypatch.setattr(orion_module.settings, "tiled_server_uri", "")

    backend = OrionBackend(
        client=FakeClient(), working_dir="/jobs", models_dir="/models", account="staff"
    )
    with pytest.raises(ValueError, match="EMBLASE_TILED_SERVER_URI is not set"):
        await backend.submit(
            model_name="vae",
            inputs=["proposal/scan"],
        )


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
            payload_b64="abc",
            npy_path="/data/f.npy",
        )


# ---------------------------------------------------------------------------
# Streaming script rendering
# ---------------------------------------------------------------------------


def test_render_streaming_inference_script_substitutes_all_placeholders():
    script = _render_streaming_inference_script(
        model_name="noop",
        models_dir="/models",
        run_path="smi/sandbox/run_xyz",
        output="smi/sandbox/results/run_xyz",
        batch_size=4,
        thumb_mode="logroi",
        image_key="pil900KW_image",
        ws_max_size=8 * 1024 * 1024,
    )
    assert 'model_name      = "noop"' in script
    assert 'models_dir      = "/models"' in script
    assert 'run_path        = "smi/sandbox/run_xyz"' in script
    assert 'output          = "smi/sandbox/results/run_xyz"' in script
    assert "batch_size      = 4" in script
    assert 'thumb_mode      = "logroi"' in script
    assert 'image_key       = "pil900KW_image"' in script
    assert f"ws_max_size     = {8 * 1024 * 1024}" in script


def test_render_streaming_inference_script_contains_valid_python():
    """The rendered script must compile without syntax errors."""
    script = _render_streaming_inference_script(
        model_name="noop",
        models_dir="/models",
        run_path="smi/sandbox/run_xyz",
        output="smi/sandbox/results/run_xyz",
    )
    compile(script, "<streaming_inference>", "exec")  # raises SyntaxError on failure


# ---------------------------------------------------------------------------
# submit_streaming — environment injection
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self):
        self.last_environment = None

    async def submit_job(self, script, working_dir, overrides=None, environment=None):
        self.last_environment = environment
        return 42


@pytest.mark.asyncio
async def test_submit_streaming_injects_tiled_env(monkeypatch):
    """submit_streaming must forward Tiled URI and API key to the job environment."""
    import emblase.compute.orion as orion_module

    monkeypatch.setattr(orion_module.settings, "tiled_server_uri", "https://tiled.example.com")
    monkeypatch.setattr(orion_module.settings, "tiled_api_key", "secret")
    monkeypatch.setattr(orion_module.settings, "tiled_access_tags", "")

    client = _FakeClient()
    backend = OrionBackend(
        client=client, working_dir="/jobs", models_dir="/models", account="staff"
    )
    await backend.submit_streaming(
        run_path="smi/sandbox/run_xyz",
        output="smi/sandbox/results/run_xyz",
        model_name="noop",
    )

    env = client.last_environment
    assert any("EMBLASE_TILED_SERVER_URI=https://tiled.example.com" in e for e in env)
    assert any("EMBLASE_TILED_API_KEY=secret" in e for e in env)


@pytest.mark.asyncio
async def test_submit_streaming_missing_tiled_uri_raises(monkeypatch):
    """submit_streaming must raise if EMBLASE_TILED_SERVER_URI is not configured."""
    import emblase.compute.orion as orion_module

    monkeypatch.setattr(orion_module.settings, "tiled_server_uri", "")

    backend = OrionBackend(
        client=_FakeClient(), working_dir="/jobs", models_dir="/models", account="staff"
    )
    with pytest.raises(ValueError, match="EMBLASE_TILED_SERVER_URI is not set"):
        await backend.submit_streaming(
            run_path="smi/sandbox/run_xyz",
            output="smi/sandbox/results/run_xyz",
            model_name="noop",
        )


@pytest.mark.asyncio
async def test_submit_streaming_injects_access_tags(monkeypatch):
    """Access tags must appear in the environment when configured."""
    import emblase.compute.orion as orion_module

    monkeypatch.setattr(orion_module.settings, "tiled_server_uri", "https://tiled.example.com")
    monkeypatch.setattr(orion_module.settings, "tiled_api_key", "")
    monkeypatch.setattr(orion_module.settings, "tiled_access_tags", "smi_sandbox,staff")

    client = _FakeClient()
    backend = OrionBackend(
        client=client, working_dir="/jobs", models_dir="/models", account="staff"
    )
    await backend.submit_streaming(
        run_path="smi/sandbox/run_xyz",
        output="smi/sandbox/results/run_xyz",
        model_name="noop",
    )

    env = client.last_environment
    assert any("EMBLASE_TILED_ACCESS_TAGS=smi_sandbox,staff" in e for e in env)


# ---------------------------------------------------------------------------
# OrionBackend.wait and result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orion_backend_wait_returns_completed():
    class FakeClient:
        async def wait_for_job(self, job_id, poll_interval, timeout):
            from emblase.compute.orion import OrionJob

            return OrionJob(job_id=job_id, state="COMPLETED")

    backend = OrionBackend(
        client=FakeClient(), working_dir="/jobs", models_dir="/models", account="staff"
    )
    status = await backend.wait("5", poll_interval=0, timeout=10)
    from emblase.compute.base import JobStatus

    assert status == JobStatus.completed


@pytest.mark.asyncio
async def test_orion_backend_wait_returns_failed():
    class FakeClient:
        async def wait_for_job(self, job_id, poll_interval, timeout):
            from emblase.compute.orion import OrionJob

            return OrionJob(job_id=job_id, state="FAILED")

    backend = OrionBackend(
        client=FakeClient(), working_dir="/jobs", models_dir="/models", account="staff"
    )
    status = await backend.wait("5", poll_interval=0, timeout=10)
    from emblase.compute.base import JobStatus

    assert status == JobStatus.failed


# ---------------------------------------------------------------------------
# _projector_mode_and_name
# ---------------------------------------------------------------------------


def test_projector_mode_none_gives_scratch():
    from emblase.compute.orion import _projector_mode_and_name

    mode, name = _projector_mode_and_name(None)
    assert mode == "scratch"
    assert name == ""


def test_projector_mode_false_variants():
    from emblase.compute.orion import _projector_mode_and_name

    for val in ("false", "False", "FALSE", "0", "no", "none", "No"):
        mode, name = _projector_mode_and_name(val)
        assert mode == "false", f"failed for {val!r}"
        assert name == ""


def test_projector_mode_name():
    from emblase.compute.orion import _projector_mode_and_name

    mode, name = _projector_mode_and_name("umap_approx")
    assert mode == "name"
    assert name == "umap_approx"


def test_render_inference_script_contains_classifier_name():
    script = _render_inference_script(
        model_name="vit",
        models_dir="/models",
        classifier="my_cls",
    )
    assert 'classifier_name = "my_cls"' in script


def test_render_inference_script_classifier_empty_by_default():
    script = _render_inference_script(
        model_name="vit",
        models_dir="/models",
    )
    assert 'classifier_name = ""' in script


def test_render_streaming_inference_script_contains_classifier_name():
    script = _render_streaming_inference_script(
        model_name="vit",
        models_dir="/models",
        run_path="smi/run_xyz",
        output="smi/results/run_xyz",
        classifier="my_cls",
    )
    assert 'classifier_name = "my_cls"' in script


def test_render_inference_script_compiles():
    """The rendered batch inference script must compile without syntax errors."""
    script = _render_inference_script(
        model_name="noop",
        models_dir="/models",
        inputs=[],
        output="",
    )
    compile(script, "<inference>", "exec")
