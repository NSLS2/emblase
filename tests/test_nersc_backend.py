"""Tests for NERSCBackend — script rendering, job submission, and environment injection.

Mirrors the structure of test_orion_backend.py.  All tests use fake NERSCClient
implementations so no real API calls are made.
"""

from __future__ import annotations

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
        self.resource_id = "perlmutter"

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
        environment: dict | None = None,
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
            "environment": environment,
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
    # Script embedded in pre_launch (no filesystem upload needed)
    assert "vit" in client.submitted.get("pre_launch", "")
    assert "inference.py" in client.submitted.get("pre_launch", "")
    # Structured job spec passed to submit_job
    assert client.submitted["executable"] == "python"
    assert any("inference.py" in a for a in client.submitted["arguments"])
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
    env = client.submitted["environment"]
    assert env.get("EMBLASE_TILED_SERVER_URI") == "https://tiled.example.com"
    assert env.get("EMBLASE_TILED_API_KEY") == "secret"
    # Script embedded in pre_launch — must contain streaming-specific markers
    pre_launch = client.submitted.get("pre_launch", "")
    assert "_on_new_image_data" in pre_launch or "streaming" in pre_launch


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
        models_dir="/pscratch/sd/d/dallan/emblase/models",
        batch_size=1,
        run_path="smi/sandbox/run_1086139",
        output="smi/sandbox/results/run_1086139",
        thumb_mode="logroi",
    )
    assert "/pscratch/sd/d/dallan/emblase/models" in script
    assert '"bnl-nsls2-smi-vit"' in script
    assert '"smi/sandbox/run_1086139"' in script
    compile(script, "<inference_nersc>", "exec")


def test_render_streaming_script_with_nersc_paths():
    script = _render_streaming_inference_script(
        model_name="bnl-nsls2-smi-vit",
        models_dir="/pscratch/sd/d/dallan/emblase/models",
        run_path="smi/sandbox/inputs_copy/run_xyz",
        output="smi/sandbox/results/run_xyz",
        batch_size=1,
        thumb_mode="logroi",
    )
    assert "/pscratch/sd/d/dallan/emblase/models" in script
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
