"""Tests for NERSCBackend — script rendering, job submission, and environment injection.

Mirrors the structure of test_orion_backend.py.  All tests use fake NERSCClient
implementations so no real API calls are made.
"""

from __future__ import annotations

from urllib.parse import urlparse

import pytest

from emblase.compute.base import JobStatus
from emblase.compute.nersc import (
    _NERSC_STATE_MAP,
    NERSCBackend,
    NERSCJob,
)
from emblase.compute.orion import (
    _render_inference_script,
    _render_streaming_inference_script,
)

# ---------------------------------------------------------------------------
# State mapping
# ---------------------------------------------------------------------------


def test_nersc_state_map_covers_iri_states():
    for state in ("new", "queued", "held", "active", "completed", "failed", "canceled"):
        assert state in _NERSC_STATE_MAP, f"Missing state: {state}"


def test_nersc_state_map_completed():
    assert _NERSC_STATE_MAP["completed"] == JobStatus.completed


def test_nersc_state_map_failed_variants():
    for state in ("failed", "canceled"):
        assert _NERSC_STATE_MAP[state] == JobStatus.failed


def test_nersc_state_map_pending_variants():
    for state in ("new", "queued", "held"):
        assert _NERSC_STATE_MAP[state] == JobStatus.pending


def test_nersc_state_map_running():
    assert _NERSC_STATE_MAP["active"] == JobStatus.running


# ---------------------------------------------------------------------------
# _parse_time_limit
# ---------------------------------------------------------------------------


def test_parse_time_limit_hhmmss():
    assert NERSCBackend._parse_time_limit("01:30:00") == 5400


def test_parse_time_limit_mmss():
    assert NERSCBackend._parse_time_limit("30:00") == 1800


def test_parse_time_limit_seconds_only():
    assert NERSCBackend._parse_time_limit("3600") == 3600


# ---------------------------------------------------------------------------
# Fake client helpers
# ---------------------------------------------------------------------------


class _FakeClient:
    """Minimal fake NERSCClient for unit tests — no real HTTP calls."""

    def __init__(self, job_id="42", state="completed"):
        self._job_id = job_id
        self._state = state
        self.submitted: dict = {}
        self.uploaded: dict = {}  # remote_path -> content
        self.resource_id = "perlmutter"

    async def upload_script(
        self,
        content: str,
        remote_path: str,
        filesystem_resource_id: str = "scratch",
        mode: str = "400",
    ) -> None:
        self.uploaded[remote_path] = content

    async def submit_job(
        self,
        executable: str,
        arguments: list[str],
        working_dir: str,
        container_image: str,
        name: str = "emblase",
        account: str = "",
        time_limit_s: int = 1800,
        nodes: int = 1,
        gpus_per_process: int = 1,
        constraint: str = "",
        queue_name: str = "",
        pre_launch: str = "",
        stdout_path: str = "",
        stderr_path: str = "",
        volume_mounts: list | None = None,
    ) -> str:
        self.submitted = {
            "executable": executable,
            "arguments": arguments,
            "working_dir": working_dir,
            "container_image": container_image,
            "name": name,
            "account": account,
            "time_limit_s": time_limit_s,
            "constraint": constraint,
            "queue_name": queue_name,
            "pre_launch": pre_launch,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "volume_mounts": volume_mounts,
        }
        return self._job_id

    async def get_job(self, job_id: str) -> NERSCJob:
        return NERSCJob(job_id=job_id, state=self._state)

    async def wait_for_job(self, job_id: str, poll_interval=15.0, timeout=1800.0) -> NERSCJob:
        return NERSCJob(job_id=job_id, state=self._state)

    async def cancel_job(self, job_id: str) -> None:
        self.submitted["cancelled"] = job_id


# ---------------------------------------------------------------------------
# NERSCBackend.submit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nersc_backend_submit_calls_client():
    client = _FakeClient(job_id="99")
    backend = NERSCBackend(
        client=client,
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
        account="nslsii",
        container_image="ghcr.io/nsls2/emblase:latest",
    )
    job_id = await backend.submit(model_name="vit")

    assert job_id == "99"
    assert "99" in backend._jobs
    assert backend._jobs["99"]["model_name"] == "vit"
    # Script uploaded to scratch before job submission
    script_path = backend._jobs["99"]["script_path"]
    assert script_path in client.uploaded
    assert "vit" in client.uploaded[script_path]
    # Job submitted with plain python executable pointing at the uploaded script
    assert client.submitted["executable"] == "python"
    assert client.submitted["arguments"] == [script_path]
    assert client.submitted["container_image"] == "ghcr.io/nsls2/emblase:latest"
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

    # Secrets are baked directly into the preamble — verify key vars are present
    script_path = backend._jobs[client._job_id]["script_path"]
    script_src = client.uploaded[script_path]
    assert "EMBLASE_TILED_API_KEY" in script_src
    # No per-job .env file should be uploaded to /pscratch
    assert not any(k.endswith("/.env") for k in client.uploaded)


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

    # No per-job .env should appear on /pscratch
    assert not any(k.endswith("/.env") for k in client.uploaded)


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

    client = _FakeClient()
    backend = NERSCBackend(
        client=client,
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
    )
    await backend.submit(model_name="vit")

    # No secrets written to /pscratch as a separate .env file
    assert not any(k.endswith("/.env") for k in client.uploaded)
    # Preamble should bake in the MLflow credentials directly
    script_path = backend._jobs[client._job_id]["script_path"]
    assert "EMBLASE_MLFLOW_TRACKING_URI" in client.uploaded[script_path]
    assert "EMBLASE_MLFLOW_API_KEY" in client.uploaded[script_path]


@pytest.mark.asyncio
async def test_nersc_backend_submit_time_limit_converted():
    """time_limit HH:MM:SS is converted to seconds in the job spec."""
    client = _FakeClient()
    backend = NERSCBackend(
        client=client,
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
        time_limit="00:45:00",
    )
    await backend.submit(model_name="vit")
    assert client.submitted["time_limit_s"] == 2700


@pytest.mark.asyncio
async def test_nersc_backend_submit_queue_propagated():
    """queue_name is forwarded to submit_job."""
    client = _FakeClient()
    backend = NERSCBackend(
        client=client,
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
        queue="gpu_shared",
    )
    await backend.submit(model_name="vit")
    assert client.submitted["queue_name"] == "gpu_shared"


@pytest.mark.asyncio
async def test_nersc_client_submit_job_default_volume_mounts():
    """NERSCClient.submit_job adds /pscratch volume mount by default."""
    captured = {}

    class _CapturingClient:
        resource_id = "perlmutter"
        base_url = "https://api.iri.nersc.gov/api/v1"

        async def _ensure_client(self):
            return self

        async def post(self, url, json=None):
            captured["payload"] = json

            class _Resp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"id": "42"}

            return _Resp()

    from emblase.compute.nersc import NERSCClient

    real_client = NERSCClient.__new__(NERSCClient)
    real_client.resource_id = "perlmutter"
    real_client.base_url = "https://api.iri.nersc.gov/api/v1"
    real_client._ensure_client = _CapturingClient()._ensure_client
    real_client._http = _CapturingClient()

    # Patch _ensure_client to return our capturing mock
    async def _fake_ensure():
        return _CapturingClient()

    real_client._ensure_client = _fake_ensure

    await real_client.submit_job(
        executable="python",
        arguments=["/pscratch/jobs/inference.py"],
        working_dir="/pscratch/jobs",
        container_image="ghcr.io/test/emblase:latest",
    )

    mounts = captured["payload"]["container"]["volume_mounts"]
    assert any(m["source"] == "/pscratch" for m in mounts)
    assert not any(m["source"] == "/global/cfs" for m in mounts)


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

    client = _FakeClient(job_id="77")
    backend = NERSCBackend(
        client=client,
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
    )
    job_id = await backend.submit_streaming(
        run_path="smi/sandbox/run_xyz",
        output="smi/sandbox/results/run_xyz",
        model_name="vit",
    )

    assert job_id == "77"
    # No per-job .env file should be uploaded to /pscratch
    assert not any(k.endswith("/.env") for k in client.uploaded)
    # Preamble in the uploaded script must bake in credentials directly
    script_path = backend._jobs["77"]["script_path"]
    uploaded_src = client.uploaded[script_path]
    assert "EMBLASE_TILED_SERVER_URI" in uploaded_src
    assert "_on_new_image_data" in uploaded_src or "streaming" in uploaded_src


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
        time_limit="00:30:00",  # default batch limit; streaming should use 2h
    )
    await backend.submit_streaming(
        run_path="smi/sandbox/run_xyz",
        output="smi/sandbox/results/run_xyz",
        model_name="vit",
    )
    # Streaming defaults to 2-hour limit → 7200 seconds
    assert client.submitted["time_limit_s"] == 7200


# ---------------------------------------------------------------------------
# NERSCBackend.status / wait / result / cancel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nersc_backend_status_completed():
    client = _FakeClient(job_id="1", state="completed")
    backend = NERSCBackend(client=client, working_dir="/j", models_dir="/m")
    st = await backend.status("1")
    assert st == JobStatus.completed


@pytest.mark.asyncio
async def test_nersc_backend_status_failed():
    client = _FakeClient(job_id="2", state="failed")
    backend = NERSCBackend(client=client, working_dir="/j", models_dir="/m")
    st = await backend.status("2")
    assert st == JobStatus.failed


@pytest.mark.asyncio
async def test_nersc_backend_status_pending():
    client = _FakeClient(job_id="3", state="queued")
    backend = NERSCBackend(client=client, working_dir="/j", models_dir="/m")
    st = await backend.status("3")
    assert st == JobStatus.pending


@pytest.mark.asyncio
async def test_nersc_backend_wait_returns_completed():
    client = _FakeClient(job_id="4", state="completed")
    backend = NERSCBackend(client=client, working_dir="/j", models_dir="/m")
    st = await backend.wait("4", poll_interval=0, timeout=10)
    assert st == JobStatus.completed


@pytest.mark.asyncio
async def test_nersc_backend_wait_returns_failed():
    client = _FakeClient(job_id="5", state="failed")
    backend = NERSCBackend(client=client, working_dir="/j", models_dir="/m")
    st = await backend.wait("5", poll_interval=0, timeout=10)
    assert st == JobStatus.failed


@pytest.mark.asyncio
async def test_nersc_backend_result_no_metadata():
    client = _FakeClient(job_id="6", state="completed")
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

    client = _FakeClient(job_id="7", state="completed")
    backend = NERSCBackend(client=client, working_dir="/j", models_dir="/m")
    await backend.submit(model_name="vit", output="results/scan1")

    result = await backend.result("7")
    assert result.status == JobStatus.completed
    assert result.output_data is None  # Tiled output — nothing to return locally


@pytest.mark.asyncio
async def test_nersc_backend_cancel():
    client = _FakeClient(job_id="8")
    backend = NERSCBackend(client=client, working_dir="/j", models_dir="/m")
    await backend.cancel("8")
    assert client.submitted.get("cancelled") == "8"


# ---------------------------------------------------------------------------
# Inference script rendering (shared with Orion — just validate NERSC paths)
# ---------------------------------------------------------------------------


def test_render_inference_script_with_nersc_paths():
    script = _render_inference_script(
        model_name="bnl-nsls2-smi-vit",
        models_dir="/pscratch/sd/j/jdoe/emblase/models",
        batch_size=1,
        run_path="smi/sandbox/run_1086139",
        output="smi/sandbox/results/run_1086139",
        thumb_mode="logroi",
    )
    assert "/pscratch/sd/j/jdoe/emblase/models" in script
    assert '"bnl-nsls2-smi-vit"' in script
    assert '"smi/sandbox/run_1086139"' in script
    compile(script, "<inference_nersc>", "exec")


def test_render_streaming_script_with_nersc_paths():
    script = _render_streaming_inference_script(
        model_name="bnl-nsls2-smi-vit",
        models_dir="/pscratch/sd/j/jdoe/emblase/models",
        run_path="smi/sandbox/inputs_copy/run_xyz",
        output="smi/sandbox/results/run_xyz",
        batch_size=1,
        thumb_mode="logroi",
    )
    assert "/pscratch/sd/j/jdoe/emblase/models" in script
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
    assert 'projector_mode  = "name"' in script
    assert 'projector_name  = "umap_approx"' in script


def test_render_inference_script_image_key_default():
    """NERSCBackend.submit() uses 'primary/pil900KW_image' as the image_key default."""
    script = _render_inference_script(
        model_name="vit",
        models_dir="/pscratch/models",
        image_key="primary/pil900KW_image",
    )
    assert '"primary/pil900KW_image"' in script


# ---------------------------------------------------------------------------
# monitor_job
# ---------------------------------------------------------------------------


def test_monitor_job_polls_until_terminal():
    """monitor_job returns once the job reaches a terminal state."""
    states = ["queued", "active", "completed"]
    calls: list[str] = []

    class _SequentialClient(_FakeClient):
        async def get_job(self, job_id: str) -> NERSCJob:
            state = states.pop(0) if states else "completed"
            calls.append(state)
            return NERSCJob(job_id=job_id, state=state)

    backend = NERSCBackend(
        client=_SequentialClient(),
        working_dir="/j",
        models_dir="/m",
    )
    backend.monitor_job("42", poll_interval=0.0)

    assert calls[-1] == "completed"
    assert len(calls) == 3


def test_monitor_job_calls_on_status():
    """on_status callback is invoked after each poll."""
    status_log: list[tuple] = []

    class _TerminalClient(_FakeClient):
        async def get_job(self, job_id: str) -> NERSCJob:
            return NERSCJob(job_id=job_id, state="completed")

    backend = NERSCBackend(
        client=_TerminalClient(),
        working_dir="/j",
        models_dir="/m",
    )

    def _on_status(state, node, elapsed):
        status_log.append((state, node, elapsed))

    backend.monitor_job("42", poll_interval=0.0, on_status=_on_status)
    assert len(status_log) == 1
    assert status_log[0][0] == "completed"
    assert status_log[0][1] is None  # NERSC has no node field


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


@pytest.mark.asyncio
async def test_nersc_backend_submit_uploads_script():
    """submit() must upload the script to scratch before submitting the job."""
    client = _FakeClient(job_id="55")
    backend = NERSCBackend(
        client=client,
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
        account="proj_g",
        container_image="img:latest",
    )
    await backend.submit(model_name="vit")

    script_path = backend._jobs["55"]["script_path"]
    assert script_path in client.uploaded, "script not uploaded before job submission"
    assert "vit" in client.uploaded[script_path]
    assert client.submitted["executable"] == "python"
    assert client.submitted["arguments"] == [script_path]


# ---------------------------------------------------------------------------
# _secrets_preamble
# ---------------------------------------------------------------------------


def test_secrets_preamble_bakes_in_credentials(monkeypatch):
    """The preamble must contain os.environ.setdefault calls for all secret vars."""
    import emblase.compute.nersc as nersc_module

    monkeypatch.setattr(nersc_module.settings, "mlflow_tracking_uri", "https://mlflow.example.com")
    monkeypatch.setattr(nersc_module.settings, "mlflow_api_key", "mlf-key")
    monkeypatch.setattr(nersc_module.settings, "tiled_server_uri", "https://tiled.example.com")
    monkeypatch.setattr(nersc_module.settings, "tiled_api_key", "tiled-key")
    monkeypatch.setattr(nersc_module.settings, "tiled_access_tags", "smi_sandbox")
    monkeypatch.setattr(nersc_module.settings, "nersc_models_dir", "/pscratch/models")

    backend = NERSCBackend(
        client=_FakeClient(),
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
    )
    preamble = backend._secrets_preamble(extra_env={"JOB_DIR": "/pscratch/jobs/123"})

    # Must be valid Python
    compile(preamble, "<preamble>", "exec")

    assert "EMBLASE_MLFLOW_TRACKING_URI" in preamble
    assert "https://mlflow.example.com" in preamble
    assert "EMBLASE_MLFLOW_API_KEY" in preamble
    assert "mlf-key" in preamble
    assert "EMBLASE_TILED_SERVER_URI" in preamble
    assert "EMBLASE_TILED_API_KEY" in preamble
    assert "EMBLASE_TILED_ACCESS_TAGS" in preamble
    assert "smi_sandbox" in preamble
    assert "JOB_DIR" in preamble
    assert "/pscratch/jobs/123" in preamble


def test_secrets_preamble_omits_empty_values(monkeypatch):
    """Variables with empty values must not appear in the preamble."""
    import emblase.compute.nersc as nersc_module

    monkeypatch.setattr(nersc_module.settings, "mlflow_tracking_uri", "https://mlflow.example.com")
    monkeypatch.setattr(nersc_module.settings, "mlflow_api_key", "")
    monkeypatch.setattr(nersc_module.settings, "tiled_server_uri", "")
    monkeypatch.setattr(nersc_module.settings, "tiled_api_key", "")
    monkeypatch.setattr(nersc_module.settings, "tiled_access_tags", "")
    monkeypatch.setattr(nersc_module.settings, "nersc_models_dir", "/pscratch/models")

    backend = NERSCBackend(
        client=_FakeClient(),
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
    )
    preamble = backend._secrets_preamble()
    compile(preamble, "<preamble>", "exec")

    assert "EMBLASE_MLFLOW_TRACKING_URI" in preamble
    assert "EMBLASE_MLFLOW_API_KEY" not in preamble
    assert "EMBLASE_TILED_SERVER_URI" not in preamble


def test_secrets_preamble_is_executable(monkeypatch):
    """Executing the preamble must set the expected env vars via setdefault."""
    import os

    import emblase.compute.nersc as nersc_module

    monkeypatch.setattr(nersc_module.settings, "mlflow_tracking_uri", "https://mlflow.example.com")
    monkeypatch.setattr(nersc_module.settings, "mlflow_api_key", "secret-key")
    monkeypatch.setattr(nersc_module.settings, "tiled_server_uri", "")
    monkeypatch.setattr(nersc_module.settings, "tiled_api_key", "")
    monkeypatch.setattr(nersc_module.settings, "tiled_access_tags", "")
    monkeypatch.setattr(nersc_module.settings, "nersc_models_dir", "/pscratch/models")

    backend = NERSCBackend(
        client=_FakeClient(),
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
    )
    preamble = backend._secrets_preamble(extra_env={"JOB_DIR": "/pscratch/jobs/abc"})

    for key in ("EMBLASE_MLFLOW_TRACKING_URI", "EMBLASE_MLFLOW_API_KEY", "JOB_DIR"):
        os.environ.pop(key, None)
    try:
        exec(preamble, {})  # noqa: S102
        assert os.environ.get("EMBLASE_MLFLOW_TRACKING_URI") == "https://mlflow.example.com"
        assert os.environ.get("EMBLASE_MLFLOW_API_KEY") == "secret-key"
        assert os.environ.get("JOB_DIR") == "/pscratch/jobs/abc"
    finally:
        for key in ("EMBLASE_MLFLOW_TRACKING_URI", "EMBLASE_MLFLOW_API_KEY", "JOB_DIR"):
            os.environ.pop(key, None)


@pytest.mark.asyncio
async def test_nersc_backend_submit_script_has_secrets_preamble(monkeypatch):
    """Uploaded inference script must start with the secrets preamble with baked-in creds."""
    import emblase.compute.nersc as nersc_module

    monkeypatch.setattr(nersc_module.settings, "tiled_server_uri", "https://t.test")
    monkeypatch.setattr(nersc_module.settings, "tiled_api_key", "k")
    monkeypatch.setattr(nersc_module.settings, "tiled_access_tags", "")
    monkeypatch.setattr(nersc_module.settings, "mlflow_tracking_uri", "")
    monkeypatch.setattr(nersc_module.settings, "mlflow_api_key", "")
    monkeypatch.setattr(nersc_module.settings, "nersc_models_dir", "/pscratch/models")

    client = _FakeClient(job_id="77")
    backend = NERSCBackend(
        client=client,
        working_dir="/pscratch/jobs",
        models_dir="/pscratch/models",
    )
    await backend.submit(model_name="vit", output="results/scan1")

    script_path = backend._jobs["77"]["script_path"]
    script_src = client.uploaded[script_path]
    assert "emblase job environment" in script_src, "secrets preamble header not found"
    parsed = urlparse("https://t.test")
    assert parsed.scheme == "https" and parsed.hostname == "t.test"
    assert f"'EMBLASE_TILED_SERVER_URI', '{parsed.geturl()}'" in script_src
    # No file-reading logic — credentials are baked in directly
    assert "open(" not in script_src.split("# --- end")[0]
    assert "os.remove" not in script_src


# ---------------------------------------------------------------------------
# NERSCClient.upload_script chmod
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_script_chmods_400():
    """upload_script() must POST upload then PUT chmod 400."""
    calls: list[dict] = []

    class _MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"task_id": "task-1"}

    class _MockHttpClient:
        async def post(self, url, **kwargs):
            calls.append({"method": "POST", "url": url, **kwargs})
            return _MockResponse()

        async def put(self, url, **kwargs):
            calls.append({"method": "PUT", "url": url, **kwargs})
            return _MockResponse()

        async def get(self, url, **kwargs):
            calls.append({"method": "GET", "url": url})
            return type(
                "R",
                (),
                {
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {"status": "completed", "result": {}},
                },
            )()

    from emblase.compute.nersc import NERSCClient

    real_client = NERSCClient.__new__(NERSCClient)
    real_client.base_url = "https://api.iri.nersc.gov/api/v1"
    real_client.resource_id = "perlmutter"
    mock_http = _MockHttpClient()
    real_client._client = mock_http

    async def _fake_ensure():
        return mock_http

    real_client._ensure_client = _fake_ensure

    await real_client.upload_script(
        content="print('hello')",
        remote_path="/pscratch/sd/d/dallan/jobs/scripts/123/inference.py",
    )

    post_calls = [c for c in calls if c["method"] == "POST"]
    put_calls = [c for c in calls if c["method"] == "PUT"]

    # mkdir + upload
    assert any("upload" in c["url"] for c in post_calls)
    # chmod 400
    assert len(put_calls) >= 1
    assert "filesystem/chmod/scratch" in put_calls[0]["url"]
    assert put_calls[0]["json"]["mode"] == "400"


def test_model_cache_uses_orion_models_dir(monkeypatch, tmp_path):
    """load_model() must pass orion_models_dir as the MLflow cache root on Orion."""
    import emblase.models as _models

    captured = {}

    def _fake_load_from_mlflow(model_name, models_dir, **kwargs):
        captured["cache_root"] = str(models_dir)
        raise RuntimeError("stop")

    monkeypatch.setenv("EMBLASE_ORION_MODELS_DIR", str(tmp_path / "orion_models"))
    monkeypatch.delenv("EMBLASE_NERSC_MODELS_DIR", raising=False)
    monkeypatch.setattr(_models, "_load_from_mlflow", _fake_load_from_mlflow)

    with pytest.raises(RuntimeError, match="stop"):
        _models.load_model("bnl-nsls2-smi-vit")

    assert captured["cache_root"] == str(tmp_path / "orion_models")


def test_model_cache_uses_nersc_models_dir(monkeypatch, tmp_path):
    """load_model() must pass nersc_models_dir as the MLflow cache root on NERSC."""
    import emblase.models as _models

    captured = {}

    def _fake_load_from_mlflow(model_name, models_dir, **kwargs):
        captured["cache_root"] = str(models_dir)
        raise RuntimeError("stop")

    monkeypatch.delenv("EMBLASE_ORION_MODELS_DIR", raising=False)
    monkeypatch.setenv("EMBLASE_NERSC_MODELS_DIR", str(tmp_path / "nersc_models"))
    monkeypatch.setattr(_models, "_load_from_mlflow", _fake_load_from_mlflow)

    with pytest.raises(RuntimeError, match="stop"):
        _models.load_model("bnl-nsls2-smi-vit")

    assert captured["cache_root"] == str(tmp_path / "nersc_models")
