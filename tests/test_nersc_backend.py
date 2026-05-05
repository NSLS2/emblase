"""Tests for NERSCBackend — script rendering, sbatch construction, and environment injection.

Mirrors the structure of test_orion_backend.py.  All tests use fake NERSCClient
implementations so no real API calls are made.
"""

from __future__ import annotations

import pytest

from emblase.compute.base import JobStatus
from emblase.compute.nersc import (
    NERSCBackend,
    NERSCJob,
    _NERSC_STATE_MAP,
    _build_nersc_script,
)
from emblase.compute.orion import (
    _render_inference_script,
    _render_streaming_inference_script,
)


# ---------------------------------------------------------------------------
# Sbatch script structure
# ---------------------------------------------------------------------------


def test_build_nersc_script_structure():
    script = _build_nersc_script(
        working_dir="/pscratch/jobs",
        script_path="/pscratch/jobs/scripts/123/inference.py",
        container_image="ghcr.io/nsls2/emblase:latest",
        job_name="emblase-vit",
        time_limit="00:30:00",
        constraint="gpu",
        account="nslsii",
    )
    assert script.startswith("#!/bin/bash")
    assert "#SBATCH --job-name=emblase-vit" in script
    assert "#SBATCH --time=00:30:00" in script
    assert "#SBATCH --constraint=gpu" in script
    assert "#SBATCH --account=nslsii" in script
    assert "#SBATCH --output=/pscratch/jobs/slurm-%j.out" in script
    assert "shifter --image=ghcr.io/nsls2/emblase:latest" in script
    assert "inference.py" in script
    assert "module load shifter" in script


def test_build_nersc_script_no_account():
    script = _build_nersc_script(
        working_dir="/pscratch/jobs",
        script_path="/pscratch/jobs/scripts/123/inference.py",
        container_image="ghcr.io/nsls2/emblase:latest",
    )
    assert "#SBATCH --account" not in script


def test_build_nersc_script_creates_job_dir():
    script = _build_nersc_script(
        working_dir="/pscratch/jobs",
        script_path="/pscratch/jobs/scripts/123/inference.py",
        container_image="ghcr.io/nsls2/emblase:latest",
    )
    assert "mkdir -p" in script
    assert "JOB_DIR" in script


# ---------------------------------------------------------------------------
# State mapping
# ---------------------------------------------------------------------------


def test_nersc_state_map_covers_slurm_states():
    for state in ("PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
        assert state in _NERSC_STATE_MAP, f"Missing state: {state}"


def test_nersc_state_map_completed():
    assert _NERSC_STATE_MAP["COMPLETED"] == JobStatus.completed
    assert _NERSC_STATE_MAP["completed"] == JobStatus.completed


def test_nersc_state_map_failed_variants():
    for state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"):
        assert _NERSC_STATE_MAP[state] == JobStatus.failed


def test_nersc_state_map_pending_variants():
    for state in ("PENDING", "CONFIGURING", "queued"):
        assert _NERSC_STATE_MAP[state] == JobStatus.pending


# ---------------------------------------------------------------------------
# Fake client helpers
# ---------------------------------------------------------------------------


class _FakeClient:
    """Minimal fake NERSCClient for unit tests — no real HTTP calls."""

    def __init__(self, task_id="42", state="COMPLETED"):
        self._task_id = task_id
        self._state = state
        self.submitted: dict = {}
        self.uploaded: list[tuple[str, str]] = []
        self.mkdirs: list[str] = []
        self.resource_id = "perlmutter"

    async def mkdir(self, remote_path: str) -> None:
        self.mkdirs.append(remote_path)

    async def upload(self, remote_path: str, content: str) -> None:
        self.uploaded.append((remote_path, content))

    async def submit_job(self, script, working_dir, constraint="gpu", account="",
                         time_limit="00:30:00", nodes=1, tasks_per_node=1,
                         environment=None) -> str:
        self.submitted = {
            "script": script,
            "working_dir": working_dir,
            "environment": environment,
            "account": account,
        }
        return self._task_id

    async def get_job(self, task_id: str) -> NERSCJob:
        return NERSCJob(job_id=task_id, state=self._state)

    async def wait_for_job(self, task_id: str, poll_interval=10.0, timeout=1800.0) -> NERSCJob:
        return NERSCJob(job_id=task_id, state=self._state)

    async def cancel_job(self, task_id: str) -> None:
        self.submitted["cancelled"] = task_id


# ---------------------------------------------------------------------------
# NERSCBackend.submit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nersc_backend_submit_calls_client():
    client = _FakeClient(task_id="99")
    backend = NERSCBackend(
        client=client,
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
        account="nslsii",
        container_image="ghcr.io/nsls2/emblase:latest",
    )
    task_id = await backend.submit(model_name="vit")

    assert task_id == "99"
    assert "99" in backend._jobs
    assert backend._jobs["99"]["model_name"] == "vit"
    # Script uploaded before submission
    assert len(client.uploaded) == 1
    _, uploaded_content = client.uploaded[0]
    assert "vit" in uploaded_content
    # Sbatch script submitted to NERSC
    assert "shifter" in client.submitted["script"]
    assert client.submitted["account"] == "nslsii"


@pytest.mark.asyncio
async def test_nersc_backend_submit_injects_tiled_env(monkeypatch):
    import emblase.compute.nersc as nersc_module

    monkeypatch.setattr(nersc_module.settings, "tiled_server_uri", "https://tiled.example.com")
    monkeypatch.setattr(nersc_module.settings, "tiled_api_key", "mykey")
    monkeypatch.setattr(nersc_module.settings, "tiled_access_tags", "")

    client = _FakeClient()
    backend = NERSCBackend(
        client=client,
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
    )
    await backend.submit(model_name="vit", output="results/scan1")

    env = client.submitted["environment"]
    assert env.get("EMBLASE_TILED_SERVER_URI") == "https://tiled.example.com"
    assert env.get("EMBLASE_TILED_API_KEY") == "mykey"


@pytest.mark.asyncio
async def test_nersc_backend_submit_injects_access_tags(monkeypatch):
    import emblase.compute.nersc as nersc_module

    monkeypatch.setattr(nersc_module.settings, "tiled_server_uri", "https://tiled.example.com")
    monkeypatch.setattr(nersc_module.settings, "tiled_api_key", "")
    monkeypatch.setattr(nersc_module.settings, "tiled_access_tags", "smi_sandbox,staff")

    client = _FakeClient()
    backend = NERSCBackend(
        client=client,
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
    )
    await backend.submit(model_name="vit", output="results/scan1")

    env = client.submitted["environment"]
    assert env.get("EMBLASE_TILED_ACCESS_TAGS") == "smi_sandbox,staff"


@pytest.mark.asyncio
async def test_nersc_backend_submit_tiled_required_raises(monkeypatch):
    import emblase.compute.nersc as nersc_module

    monkeypatch.setattr(nersc_module.settings, "tiled_server_uri", "")

    client = _FakeClient()
    backend = NERSCBackend(
        client=client,
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
    )
    with pytest.raises(ValueError, match="EMBLASE_TILED_SERVER_URI is not set"):
        await backend.submit(model_name="vit", run_path="smi/sandbox/run_xyz")


@pytest.mark.asyncio
async def test_nersc_backend_submit_injects_mlflow_env(monkeypatch):
    import emblase.compute.nersc as nersc_module

    monkeypatch.setattr(nersc_module.settings, "tiled_server_uri", "")
    monkeypatch.setattr(nersc_module.settings, "mlflow_tracking_uri", "https://mlflow.example.com")
    monkeypatch.setattr(nersc_module.settings, "mlflow_api_key", "mlf-key")
    monkeypatch.setattr(nersc_module.settings, "model_cache_dir", "/pscratch/cache")

    client = _FakeClient()
    backend = NERSCBackend(
        client=client,
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
    )
    await backend.submit(model_name="vit")

    env = client.submitted["environment"]
    assert env.get("EMBLASE_MLFLOW_TRACKING_URI") == "https://mlflow.example.com"
    assert env.get("EMBLASE_MLFLOW_API_KEY") == "mlf-key"
    assert env.get("EMBLASE_MODEL_CACHE_DIR") == "/pscratch/cache"


# ---------------------------------------------------------------------------
# NERSCBackend.submit_streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nersc_backend_submit_streaming_requires_tiled(monkeypatch):
    import emblase.compute.nersc as nersc_module

    monkeypatch.setattr(nersc_module.settings, "tiled_server_uri", "")

    client = _FakeClient()
    backend = NERSCBackend(
        client=client,
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
    )
    with pytest.raises(ValueError, match="EMBLASE_TILED_SERVER_URI is not set"):
        await backend.submit_streaming(
            run_path="smi/sandbox/run_xyz",
            output="smi/sandbox/results/run_xyz",
            model_name="vit",
        )


@pytest.mark.asyncio
async def test_nersc_backend_submit_streaming_injects_tiled_env(monkeypatch):
    import emblase.compute.nersc as nersc_module

    monkeypatch.setattr(nersc_module.settings, "tiled_server_uri", "https://tiled.example.com")
    monkeypatch.setattr(nersc_module.settings, "tiled_api_key", "secret")
    monkeypatch.setattr(nersc_module.settings, "tiled_access_tags", "")

    client = _FakeClient(task_id="77")
    backend = NERSCBackend(
        client=client,
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
    )
    task_id = await backend.submit_streaming(
        run_path="smi/sandbox/run_xyz",
        output="smi/sandbox/results/run_xyz",
        model_name="vit",
    )

    assert task_id == "77"
    env = client.submitted["environment"]
    assert env.get("EMBLASE_TILED_SERVER_URI") == "https://tiled.example.com"
    assert env.get("EMBLASE_TILED_API_KEY") == "secret"
    # Uploaded script must contain streaming-specific markers
    _, uploaded_content = client.uploaded[0]
    assert "streaming_inference" in uploaded_content or "_on_new_image_data" in uploaded_content


@pytest.mark.asyncio
async def test_nersc_backend_submit_streaming_uses_extended_time_limit(monkeypatch):
    import emblase.compute.nersc as nersc_module

    monkeypatch.setattr(nersc_module.settings, "tiled_server_uri", "https://tiled.example.com")
    monkeypatch.setattr(nersc_module.settings, "tiled_api_key", "")
    monkeypatch.setattr(nersc_module.settings, "tiled_access_tags", "")

    client = _FakeClient()
    backend = NERSCBackend(
        client=client,
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
        time_limit="00:30:00",  # default batch limit; streaming should override
    )
    await backend.submit_streaming(
        run_path="smi/sandbox/run_xyz",
        output="smi/sandbox/results/run_xyz",
        model_name="vit",
    )
    # Streaming hard-codes 2-hour limit in the sbatch script
    assert "02:00:00" in client.submitted["script"]


# ---------------------------------------------------------------------------
# NERSCBackend.status / wait / result / cancel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nersc_backend_status_completed():
    client = _FakeClient(task_id="1", state="COMPLETED")
    backend = NERSCBackend(client=client, working_dir="/j", models_dir="/m")
    st = await backend.status("1")
    assert st == JobStatus.completed


@pytest.mark.asyncio
async def test_nersc_backend_status_failed():
    client = _FakeClient(task_id="2", state="FAILED")
    backend = NERSCBackend(client=client, working_dir="/j", models_dir="/m")
    st = await backend.status("2")
    assert st == JobStatus.failed


@pytest.mark.asyncio
async def test_nersc_backend_status_pending():
    client = _FakeClient(task_id="3", state="PENDING")
    backend = NERSCBackend(client=client, working_dir="/j", models_dir="/m")
    st = await backend.status("3")
    assert st == JobStatus.pending


@pytest.mark.asyncio
async def test_nersc_backend_wait_returns_completed():
    client = _FakeClient(task_id="4", state="COMPLETED")
    backend = NERSCBackend(client=client, working_dir="/j", models_dir="/m")
    st = await backend.wait("4", poll_interval=0, timeout=10)
    assert st == JobStatus.completed


@pytest.mark.asyncio
async def test_nersc_backend_wait_returns_failed():
    client = _FakeClient(task_id="5", state="FAILED")
    backend = NERSCBackend(client=client, working_dir="/j", models_dir="/m")
    st = await backend.wait("5", poll_interval=0, timeout=10)
    assert st == JobStatus.failed


@pytest.mark.asyncio
async def test_nersc_backend_result_no_metadata():
    client = _FakeClient(task_id="6", state="COMPLETED")
    backend = NERSCBackend(client=client, working_dir="/j", models_dir="/m")
    result = await backend.result("6")
    assert result.status == JobStatus.failed
    assert "metadata lost" in (result.error or "")


@pytest.mark.asyncio
async def test_nersc_backend_result_with_output(monkeypatch):
    import emblase.compute.nersc as nersc_module

    monkeypatch.setattr(nersc_module.settings, "tiled_server_uri", "https://tiled")
    monkeypatch.setattr(nersc_module.settings, "tiled_api_key", "")
    monkeypatch.setattr(nersc_module.settings, "tiled_access_tags", "")

    client = _FakeClient(task_id="7", state="COMPLETED")
    backend = NERSCBackend(client=client, working_dir="/j", models_dir="/m")
    await backend.submit(model_name="vit", output="results/scan1")

    result = await backend.result("7")
    assert result.status == JobStatus.completed
    assert result.output_data is None  # Tiled output — nothing to return locally


@pytest.mark.asyncio
async def test_nersc_backend_cancel():
    client = _FakeClient(task_id="8")
    backend = NERSCBackend(client=client, working_dir="/j", models_dir="/m")
    await backend.cancel("8")
    assert client.submitted.get("cancelled") == "8"


# ---------------------------------------------------------------------------
# Inference script rendering (shared with Orion — just validate NERSC paths)
# ---------------------------------------------------------------------------


def test_render_inference_script_with_nersc_paths():
    script = _render_inference_script(
        model_name="bnl-nsls2-smi-vit",
        models_dir="/pscratch/sd/y/ymatviych/emblase/models",
        batch_size=1,
        run_path="smi/sandbox/run_1086139",
        output="smi/sandbox/results/run_1086139",
        thumb_mode="logroi",
    )
    assert "/pscratch/sd/y/ymatviych/emblase/models" in script
    assert '"bnl-nsls2-smi-vit"' in script
    assert '"smi/sandbox/run_1086139"' in script
    compile(script, "<inference_nersc>", "exec")


def test_render_streaming_script_with_nersc_paths():
    script = _render_streaming_inference_script(
        model_name="bnl-nsls2-smi-vit",
        models_dir="/pscratch/sd/y/ymatviych/emblase/models",
        run_path="smi/sandbox/inputs_copy/run_xyz",
        output="smi/sandbox/results/run_xyz",
        batch_size=1,
        thumb_mode="logroi",
    )
    assert "/pscratch/sd/y/ymatviych/emblase/models" in script
    assert '"smi/sandbox/inputs_copy/run_xyz"' in script
    compile(script, "<streaming_nersc>", "exec")


def test_render_inference_script_classifier_and_projector():
    script = _render_inference_script(
        model_name="vit",
        models_dir="/pscratch/models",
        classifier="my_cls",
        projector="umap_approx",
    )
    assert 'classifier_name = "my_cls"' in script
    assert 'projector_mode = "name"' in script
    assert 'projector_name = "umap_approx"' in script


# ---------------------------------------------------------------------------
# NERSCClient — token validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nersc_client_missing_token_raises(monkeypatch):
    import emblase.compute.nersc as nersc_module

    monkeypatch.setattr(nersc_module.settings, "nersc_api_token", "")

    from emblase.compute.nersc import NERSCClient

    client = NERSCClient(api_token="")
    with pytest.raises(ValueError, match="NERSC API token is not set"):
        await client._ensure_client()
