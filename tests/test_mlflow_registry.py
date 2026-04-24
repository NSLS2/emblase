"""Tests for mlflow_registry.py — all external MLflow calls are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import emblase.mlflow_registry as reg

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_mv(name="my-model", version="1", run_id="run123"):
    mv = MagicMock()
    mv.name = name
    mv.version = version
    mv.run_id = run_id
    return mv


def _make_rm(name="my-model", latest_versions=None, description="desc"):
    rm = MagicMock()
    rm.name = name
    rm.latest_versions = latest_versions or []
    rm.description = description
    return rm


# ── _get_client ───────────────────────────────────────────────────────────────


def test_get_client_exits_without_uri(monkeypatch):
    monkeypatch.setattr(reg.settings, "mlflow_tracking_uri", None)
    with pytest.raises(ValueError, match="No MLflow tracking URI"):
        reg._get_client(tracking_uri=None)


# ── push ──────────────────────────────────────────────────────────────────────


def test_push_file(tmp_path, monkeypatch):
    weights = tmp_path / "model.npz"
    weights.write_bytes(b"fake-weights")

    monkeypatch.setattr(reg.settings, "mlflow_tracking_uri", "http://localhost:5000")
    monkeypatch.setattr(reg.settings, "mlflow_experiment", "test-exp")

    mock_client = MagicMock()
    mock_client.get_registered_model.side_effect = Exception("not found")
    mock_client.create_model_version.return_value = _make_mv(version="2")

    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "run123"
    mock_run.info.artifact_uri = "azureml://artifacts/run123"

    with (
        patch("emblase.mlflow_registry._get_client", return_value=mock_client),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=mock_run),
        patch("mlflow.log_artifact"),
    ):
        version = reg.push(path=weights, name="model")

    assert version == "2"
    mock_client.create_registered_model.assert_called_once()
    mock_client.create_model_version.assert_called_once()


def test_push_existing_model(tmp_path, monkeypatch):
    """Should skip create_registered_model when model already exists."""
    weights = tmp_path / "model.npz"
    weights.write_bytes(b"fake-weights")

    monkeypatch.setattr(reg.settings, "mlflow_tracking_uri", "http://localhost:5000")
    monkeypatch.setattr(reg.settings, "mlflow_experiment", "test-exp")

    mock_client = MagicMock()
    mock_client.get_registered_model.return_value = MagicMock()
    mock_client.create_model_version.return_value = _make_mv(version="3")

    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "run123"
    mock_run.info.artifact_uri = "azureml://artifacts/run123"

    with (
        patch("emblase.mlflow_registry._get_client", return_value=mock_client),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=mock_run),
        patch("mlflow.log_artifact"),
    ):
        reg.push(path=weights, name="model")

    mock_client.create_registered_model.assert_not_called()


def test_push_missing_path_exits(tmp_path):
    with pytest.raises(FileNotFoundError):
        reg.push(path=tmp_path / "nonexistent.npz")


# ── pull ──────────────────────────────────────────────────────────────────────


def test_pull_specific_version(tmp_path, monkeypatch):
    monkeypatch.setattr(reg.settings, "mlflow_tracking_uri", "http://localhost:5000")

    mock_client = MagicMock()
    mock_client.get_model_version.return_value = _make_mv()
    mock_client.download_artifacts.return_value = str(tmp_path / "model")

    with patch("emblase.mlflow_registry._get_client", return_value=mock_client):
        result = reg.pull("my-model", version="1", output_dir=tmp_path)

    assert result == tmp_path / "model"
    mock_client.download_artifacts.assert_called_once_with(
        "run123", "model", str(tmp_path)
    )


def test_pull_latest_version(tmp_path, monkeypatch):
    monkeypatch.setattr(reg.settings, "mlflow_tracking_uri", "http://localhost:5000")

    mv1 = _make_mv(version="1")
    mv3 = _make_mv(version="3")

    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [mv1, mv3]
    mock_client.get_model_version.return_value = mv3
    mock_client.download_artifacts.return_value = str(tmp_path / "model")

    with patch("emblase.mlflow_registry._get_client", return_value=mock_client):
        reg.pull("my-model", version=None, output_dir=tmp_path)

    mock_client.get_model_version.assert_called_once_with("my-model", "3")


def test_pull_no_versions_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(reg.settings, "mlflow_tracking_uri", "http://localhost:5000")

    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = []

    with (
        patch("emblase.mlflow_registry._get_client", return_value=mock_client),
        pytest.raises(ValueError, match="No versions found"),
    ):
        reg.pull("missing-model", output_dir=tmp_path)


# ── list_models ───────────────────────────────────────────────────────────────


def test_list_models(monkeypatch):
    monkeypatch.setattr(reg.settings, "mlflow_tracking_uri", "http://localhost:5000")

    lv = MagicMock()
    lv.version = "2"
    rm = _make_rm(name="vae-512", latest_versions=[lv], description="VAE model")

    mock_client = MagicMock()
    mock_client.search_registered_models.return_value = [rm]

    with patch("emblase.mlflow_registry._get_client", return_value=mock_client):
        models = reg.list_models()

    assert len(models) == 1
    assert models[0].name == "vae-512"
    assert models[0].latest_version == 2
    assert models[0].description == "VAE model"


def test_list_models_empty(monkeypatch):
    monkeypatch.setattr(reg.settings, "mlflow_tracking_uri", "http://localhost:5000")

    mock_client = MagicMock()
    mock_client.search_registered_models.return_value = []

    with patch("emblase.mlflow_registry._get_client", return_value=mock_client):
        assert reg.list_models() == []


# ── delete ────────────────────────────────────────────────────────────────────


def test_delete(monkeypatch):
    monkeypatch.setattr(reg.settings, "mlflow_tracking_uri", "http://localhost:5000")

    mock_client = MagicMock()
    mock_client.get_registered_model.return_value = MagicMock()

    with patch("emblase.mlflow_registry._get_client", return_value=mock_client):
        reg.delete("my-model")

    mock_client.delete_registered_model.assert_called_once_with("my-model")


def test_delete_missing_exits(monkeypatch):
    monkeypatch.setattr(reg.settings, "mlflow_tracking_uri", "http://localhost:5000")

    mock_client = MagicMock()
    mock_client.get_registered_model.side_effect = Exception("not found")

    with (
        patch("emblase.mlflow_registry._get_client", return_value=mock_client),
        pytest.raises(ValueError, match="not found"),
    ):
        reg.delete("ghost-model")


# ── chunked upload / download ─────────────────────────────────────────────────


def test_log_artifact_chunked_small_file(tmp_path):
    """Files below chunk_size are uploaded as-is via mlflow.log_artifact."""
    f = tmp_path / "small.npz"
    f.write_bytes(b"x" * 100)

    with patch("mlflow.log_artifact") as mock_log:
        reg._log_artifact_chunked(f, artifact_path="model", chunk_size=1024)

    mock_log.assert_called_once_with(str(f), artifact_path="model")


def test_log_artifact_chunked_large_file(tmp_path):
    """Files above chunk_size are split into parts + a manifest."""
    f = tmp_path / "big.npz"
    data = b"A" * 250  # 250 bytes
    f.write_bytes(data)
    chunk_size = 100  # → 3 chunks: 100, 100, 50

    logged = []
    with patch(
        "mlflow.log_artifact",
        side_effect=lambda p, artifact_path: logged.append(Path(p).name),
    ):
        reg._log_artifact_chunked(f, artifact_path="model", chunk_size=chunk_size)

    assert "big.npz.part000" in logged
    assert "big.npz.part001" in logged
    assert "big.npz.part002" in logged
    assert "big.npz.chunks" in logged
    assert len(logged) == 4


def test_reassemble_chunks(tmp_path):
    """_reassemble_chunks joins parts back into the original file."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    original = b"HELLO" * 40  # 200 bytes split into 2 × 100
    chunk_size = 100
    (model_dir / "weights.npz.part000").write_bytes(original[:chunk_size])
    (model_dir / "weights.npz.part001").write_bytes(original[chunk_size:])
    (model_dir / "weights.npz.chunks").write_text(
        '{"original": "weights.npz", "n_chunks": 2}'
    )

    reg._reassemble_chunks(model_dir)

    reassembled = (model_dir / "weights.npz").read_bytes()
    assert reassembled == original
    assert not (model_dir / "weights.npz.part000").exists()
    assert not (model_dir / "weights.npz.part001").exists()
    assert not (model_dir / "weights.npz.chunks").exists()


def test_reassemble_chunks_noop_when_no_manifests(tmp_path):
    """_reassemble_chunks does nothing when there are no .chunks files."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.npz").write_bytes(b"intact")

    reg._reassemble_chunks(model_dir)  # must not raise

    assert (model_dir / "weights.npz").read_bytes() == b"intact"


def test_push_uses_chunked_upload(tmp_path, monkeypatch):
    """push() calls _log_artifact_chunked, not mlflow.log_artifact directly."""
    weights = tmp_path / "model.npz"
    weights.write_bytes(b"weights")

    monkeypatch.setattr(reg.settings, "mlflow_tracking_uri", "http://localhost:5000")
    monkeypatch.setattr(reg.settings, "mlflow_experiment", "test-exp")

    mock_client = MagicMock()
    mock_client.get_registered_model.return_value = MagicMock()
    mock_client.create_model_version.return_value = _make_mv(version="1")

    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "run1"
    mock_run.info.artifact_uri = "http://server/artifacts/run1"

    with (
        patch("emblase.mlflow_registry._get_client", return_value=mock_client),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=mock_run),
        patch("emblase.mlflow_registry._log_artifact_chunked") as mock_chunked,
    ):
        reg.push(path=weights, name="model")

    mock_chunked.assert_called_once_with(weights, artifact_path="model", chunk_size=0)


# ── download_model_weights ────────────────────────────────────────────────────


def test_download_model_weights_delegates_to_pull(tmp_path, monkeypatch):
    monkeypatch.setattr(reg.settings, "mlflow_tracking_uri", "http://localhost:5000")
    expected = tmp_path / "model"

    with patch("emblase.mlflow_registry.pull", return_value=expected) as mock_pull:
        result = reg.download_model_weights("my-model", version="2", dest_dir=tmp_path)

    mock_pull.assert_called_once_with(
        "my-model", version="2", output_dir=tmp_path, tracking_uri=None, api_key=None
    )
    assert result == expected


# ── load_model_from_mlflow ────────────────────────────────────────────────────


def test_load_model_from_mlflow(tmp_path):
    weights_dir = tmp_path / "model"
    weights_dir.mkdir()
    fake_npz = weights_dir / "model.npz"
    fake_npz.write_bytes(b"fake")

    with (
        patch("emblase.mlflow_registry.pull", return_value=weights_dir),
        patch("emblase.models.load_model") as mock_load,
    ):
        mock_load.return_value = MagicMock()
        from emblase.models import load_model_from_mlflow

        load_model_from_mlflow(
            model_name="vae",
            mlflow_model="vae-512",
            mlflow_version="1",
            dest_dir=tmp_path,
        )

    mock_load.assert_called_once()
    assert mock_load.call_args[0][0] == "vae"
    assert mock_load.call_args[1]["weights_path"] == fake_npz


def test_load_model_from_mlflow_no_npz_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with (
        patch("emblase.mlflow_registry.pull", return_value=empty_dir),
        pytest.raises(FileNotFoundError, match="No .npz weight files"),
    ):
        from emblase.models import load_model_from_mlflow

        load_model_from_mlflow(
            model_name="vae",
            mlflow_model="vae-512",
            dest_dir=tmp_path,
        )
