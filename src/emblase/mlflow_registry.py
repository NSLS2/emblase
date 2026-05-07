"""MLflow model registry integration for Emblase.

Uses the standard MLflow tracking API, which works with any MLflow-compatible
server: American Science Cloud (ASC), Azure ML (via azureml-mlflow tracking URI),
Databricks, or a self-hosted open-source MLflow server.

Set EMBLASE_MLFLOW_TRACKING_URI to point at your server.  For Azure ML the URI
looks like:
  azureml://<region>.api.azureml.ms/mlflow/v1.0/subscriptions/<sub>/
  resourceGroups/<rg>/providers/Microsoft.MachineLearningServices/workspaces/<ws>

You can obtain it via:
  az ml workspace show --query mlflow_tracking_uri -o tsv

Large file handling
-------------------
Some servers (e.g. AmSC) impose a per-request upload size limit.  When a file
exceeds ``CHUNK_SIZE`` (default 100 MB) it is automatically split into numbered
chunk files (``<name>.part000``, ``<name>.part001``, …) plus a small JSON
manifest (``<name>.chunks``) before upload.  ``pull`` detects the manifest and
reassembles the original file transparently.

Public API
----------
push(path, name, description, tracking_uri, api_key, chunk_size)
pull(model_name, version, output_dir, tracking_uri, api_key) -> Path
list_models(tracking_uri, api_key) -> list[ModelInfo]
delete(model_name, tracking_uri, api_key)
download_model_weights(model_name, version, dest_dir, tracking_uri, api_key) -> Path
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import settings

# Files larger than this are split into chunks before upload (bytes).
# AmSC MLflow (Kong gateway) enforces a ~16 MB per-request limit.
# 10 MB gives comfortable headroom. Chunking is disabled by default (0 = off).
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB

# Directories and file patterns to skip when uploading a model directory.
_SKIP_DIRS = {"__pycache__", ".git", ".pixi", ".venv", "node_modules"}
_SKIP_SUFFIXES = {".pyc", ".pyo", ".DS_Store", ".ipynb_checkpoints"}
_SKIP_NAMES = {".DS_Store", ".gitignore", ".gitkeep"}


def _should_upload(path: Path) -> bool:
    """Return True if *path* should be included in a model upload.

    Excludes:
    - Any file inside a directory listed in ``_SKIP_DIRS`` (at any depth)
    - Files whose suffix is in ``_SKIP_SUFFIXES``
    - Files whose name is in ``_SKIP_NAMES``
    - Hidden files (name starts with ``.``)
    """
    for part in path.parts:
        if part in _SKIP_DIRS:
            return False
    if path.suffix in _SKIP_SUFFIXES:
        return False
    if path.name in _SKIP_NAMES:
        return False
    if path.name.startswith("."):
        return False
    return True


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class ModelInfo:
    name: str
    latest_version: str | int | None
    description: str | None = None


# ── Chunked upload / download helpers ─────────────────────────────────────────


def _log_artifact_chunked(
    local_file: Path,
    artifact_path: str,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """Upload ``local_file`` as a run artifact, splitting into chunks if needed.

    For files smaller than ``chunk_size`` this is identical to
    ``mlflow.log_artifact``.  For larger files the file is split into
    ``<name>.part000``, ``<name>.part001``, … plus a ``<name>.chunks``
    manifest, all uploaded under ``artifact_path``.  ``_reassemble_chunks``
    reverses the process on download.
    """
    import tempfile

    import mlflow

    size = local_file.stat().st_size
    if not chunk_size or size <= chunk_size:
        mlflow.log_artifact(str(local_file), artifact_path=artifact_path)
        return

    n_chunks = (size + chunk_size - 1) // chunk_size
    print(
        f"  {local_file.name} is {size / 1024**2:.0f} MB — "
        f"splitting into {n_chunks} × {chunk_size // 1024**2} MB chunks"
    )

    with tempfile.TemporaryDirectory(prefix="emblase_chunks_") as tmp:
        tmp_path = Path(tmp)
        manifest = {"original": local_file.name, "n_chunks": n_chunks}

        with open(local_file, "rb") as fh:
            for i in range(n_chunks):
                chunk_name = f"{local_file.name}.part{i:03d}"
                chunk_path = tmp_path / chunk_name
                chunk_path.write_bytes(fh.read(chunk_size))
                mlflow.log_artifact(str(chunk_path), artifact_path=artifact_path)
                print(f"    uploaded chunk {i + 1}/{n_chunks}: {chunk_name}")

        manifest_path = tmp_path / f"{local_file.name}.chunks"
        manifest_path.write_text(json.dumps(manifest))
        mlflow.log_artifact(str(manifest_path), artifact_path=artifact_path)


def _reassemble_chunks(model_dir: Path) -> None:
    """Reassemble any chunked files found under ``model_dir`` in-place.

    Looks for ``*.chunks`` manifest files, reads them, concatenates the
    corresponding ``.partNNN`` files into the original filename, then removes
    the chunk files and manifest.
    """
    for manifest_path in sorted(model_dir.rglob("*.chunks")):
        manifest = json.loads(manifest_path.read_text())
        original_name = manifest["original"]
        n_chunks = manifest["n_chunks"]
        out_path = manifest_path.parent / original_name

        print(f"  reassembling {original_name} from {n_chunks} chunks …")
        with open(out_path, "wb") as fh:
            for i in range(n_chunks):
                chunk_path = manifest_path.parent / f"{original_name}.part{i:03d}"
                fh.write(chunk_path.read_bytes())
                chunk_path.unlink()

        manifest_path.unlink()
        print(f"  → {out_path} ({out_path.stat().st_size / 1024**2:.0f} MB)")


# ── Client ────────────────────────────────────────────────────────────────────


def _get_client(tracking_uri: str | None = None, api_key: str | None = None):
    """Return a configured MlflowClient.

    Applies up to three runtime patches:

    1. X-Api-Key header injection — when ``api_key`` is provided (or
       ``EMBLASE_MLFLOW_API_KEY`` is set), every MLflow REST call receives an
       ``X-Api-Key`` header.  Required for AmSC MLflow; harmless elsewhere.

    2. Prompt-exclusion filter — mlflow 3.x injects a
       ``tag.mlflow.prompt.is_prompt != 'true'`` filter into
       ``search_registered_models`` that Azure ML's endpoint doesn't support.

    3. Artifact builder signature — mlflow 3.x passes ``tracking_uri`` /
       ``registry_uri`` kwargs to the ``azureml_artifacts_builder`` entry point,
       which only accepts ``artifact_uri``.  We wrap it to absorb the extras.

    Patches 2 and 3 are no-ops against a standard MLflow server (ASC,
    open-source).  Patch 3 is skipped when ``azureml-mlflow`` is not installed.
    """
    import mlflow
    import mlflow.tracking._model_registry.client as _rc
    from mlflow.tracking import MlflowClient

    uri = tracking_uri or settings.mlflow_tracking_uri
    if not uri:
        raise ValueError(
            "No MLflow tracking URI set. Set EMBLASE_MLFLOW_TRACKING_URI or pass --tracking-uri."
        )
    mlflow.set_tracking_uri(uri)

    # Patch 1: X-Api-Key header injection (AmSC and similar servers).
    #
    # In mlflow 3.x the call chain is:
    #   RestStore._call_endpoint → call_endpoint → http_request
    # _call_endpoint never passes extra_headers, so patching http_request alone
    # is not enough — the key never reaches the request.  We patch call_endpoint
    # instead, which sits between the two and does accept extra_headers.
    key = api_key or settings.mlflow_api_key or None
    if key:
        import mlflow.utils.rest_utils as _ru

        if not getattr(_ru.call_endpoint, "_emblase_api_key_patched", False):
            _orig_call = _ru.call_endpoint

            def _patched_call(*args, extra_headers=None, **kwargs):
                h = dict(extra_headers or {})
                h["X-Api-Key"] = key
                return _orig_call(*args, extra_headers=h, **kwargs)

            _patched_call._emblase_api_key_patched = True
            _ru.call_endpoint = _patched_call
            # rest_store imports call_endpoint at module load; patch its reference too
            import mlflow.store.tracking.rest_store as _rs
            _rs.call_endpoint = _patched_call
    key = api_key or settings.mlflow_api_key or None
    if key:
        import mlflow.utils.rest_utils as _ru

        if not getattr(_ru.call_endpoint, "_emblase_api_key_patched", False):
            _orig_call = _ru.call_endpoint

            def _patched_call(*args, extra_headers=None, **kwargs):
                h = dict(extra_headers or {})
                h["X-Api-Key"] = key
                return _orig_call(*args, extra_headers=h, **kwargs)

            _patched_call._emblase_api_key_patched = True
            _ru.call_endpoint = _patched_call
            # rest_store imports call_endpoint at module load; patch its reference too
            import mlflow.store.tracking.rest_store as _rs
            _rs.call_endpoint = _patched_call

    # Patch 2: suppress prompt-exclusion filter (Azure ML)
    _rc.is_prompt_supported_registry = lambda *_a, **_kw: False

    # Patch 3: fix artifact builder signature (azureml-mlflow + mlflow 3.x)
    try:
        from azureml.mlflow import entry_point_loaders as _epl

        _orig = _epl.azureml_artifacts_builder
        if not getattr(_orig, "_emblase_patched", False):

            def _patched(artifact_uri=None, **_kw):
                return _orig(artifact_uri=artifact_uri)

            _patched._emblase_patched = True
            _epl.azureml_artifacts_builder = _patched
            from mlflow.store.artifact.artifact_repository_registry import (
                _artifact_repository_registry,
            )

            _artifact_repository_registry._registry["azureml"] = _patched
    except (ImportError, AttributeError, KeyError):
        pass

    return MlflowClient()


# ── push ──────────────────────────────────────────────────────────────────────


def push(
    path: str | Path,
    name: str | None = None,
    description: str | None = None,
    experiment: str | None = None,
    tracking_uri: str | None = None,
    api_key: str | None = None,
    chunk_size: int = 0,
) -> str:
    """Upload a local file or directory to the model registry.

    Returns the registered model version string.

    Parameters
    ----------
    path:
        Local file (e.g. ``vae_weights.npz``) or directory (e.g. ``models/vae/``).
        All files inside a directory are uploaded under the ``model/`` artifact path.
    name:
        Registry name.  Defaults to the stem of the file or the directory name.
    description:
        Optional human-readable description stored in the registry.
    experiment:
        MLflow experiment name to log the upload run under.  Created automatically
        if it does not exist.  Defaults to ``EMBLASE_MLFLOW_EXPERIMENT`` (``emblase-models``).
    tracking_uri:
        Override ``EMBLASE_MLFLOW_TRACKING_URI`` for this call only.
    api_key:
        Override ``EMBLASE_MLFLOW_API_KEY`` for this call only.
    chunk_size:
        Split files larger than this many bytes into chunks before upload.
        ``0`` (default) disables chunking — files are uploaded as-is.
        Use ``CHUNK_SIZE`` (10 MB) or a custom value for servers with strict
        per-request size limits (e.g. AmSC MLflow, ~16 MB limit).
    """
    import mlflow

    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Path not found: {file_path}")

    model_name = name or (file_path.stem if file_path.is_file() else file_path.name)
    desc = description or f"Uploaded from {file_path.name}"
    client = _get_client(tracking_uri, api_key)

    mlflow.set_experiment(experiment or settings.mlflow_experiment)
    with mlflow.start_run(run_name=f"push-{model_name}") as run:
        if file_path.is_dir():
            files = sorted(f for f in file_path.rglob("*") if f.is_file() and _should_upload(f))
            if not files:
                raise ValueError(f"No uploadable files found under {file_path}")
            for child in files:
                # Preserve sub-directory structure relative to the model root
                rel = child.relative_to(file_path).parent
                artifact_path = f"model/{rel}" if str(rel) != "." else "model"
                _log_artifact_chunked(child, artifact_path=artifact_path, chunk_size=chunk_size)
        else:
            if not _should_upload(file_path):
                raise ValueError(f"File {file_path.name} is excluded from upload (matches skip rules)")
            _log_artifact_chunked(
                file_path, artifact_path="model", chunk_size=chunk_size
            )
        artifact_uri = f"{run.info.artifact_uri}/model"

    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name, description=desc)

    mv = client.create_model_version(
        name=model_name,
        source=artifact_uri,
        run_id=run.info.run_id,
        description=desc,
    )
    print(f"Registered '{mv.name}' version {mv.version}")
    return str(mv.version)


# ── pull ──────────────────────────────────────────────────────────────────────


def resolve_version(
    model_name: str,
    version: str | int | None = None,
    tracking_uri: str | None = None,
    api_key: str | None = None,
) -> str:
    """Return the concrete version string for ``model_name``.

    If ``version`` is already specified it is returned as-is (stringified).
    Otherwise the highest registered version number is returned.  This is a
    cheap metadata-only call — no artifacts are downloaded.
    """
    if version is not None:
        return str(version)
    client = _get_client(tracking_uri, api_key)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'")
    latest = str(max(int(v.version) for v in versions))
    print(f"No version specified for '{model_name}' — using latest: {latest}")
    return latest


def pull(
    model_name: str,
    version: str | int | None = None,
    output_dir: str | Path = ".",
    tracking_uri: str | None = None,
    api_key: str | None = None,
) -> Path:
    """Download a registered model to ``output_dir``.

    Automatically reassembles any files that were split into chunks on upload.
    Returns the local path to the downloaded artefacts.
    """
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    client = _get_client(tracking_uri, api_key)

    version = resolve_version(
        model_name, version=version, tracking_uri=tracking_uri, api_key=api_key
    )

    mv = client.get_model_version(model_name, str(version))
    print(f"Pulling '{model_name}' v{version} → {out}")
    local = client.download_artifacts(mv.run_id, "model", str(out))
    model_dir = Path(local)
    _reassemble_chunks(model_dir)
    print(f"Downloaded to {local}")
    return model_dir


# ── list ──────────────────────────────────────────────────────────────────────


def list_models(tracking_uri: str | None = None, api_key: str | None = None) -> list[ModelInfo]:
    """Return a list of :class:`ModelInfo` for every registered model."""
    client = _get_client(tracking_uri, api_key)
    models = client.search_registered_models(filter_string="")
    return sorted(
        [
            ModelInfo(
                name=m.name,
                latest_version=max((int(v.version) for v in m.latest_versions), default=None),
                description=m.description,
            )
            for m in models
        ],
        key=lambda m: m.name,
    )


# ── delete ────────────────────────────────────────────────────────────────────


def delete(model_name: str, tracking_uri: str | None = None, api_key: str | None = None) -> None:
    """Delete a registered model and all its versions."""
    client = _get_client(tracking_uri, api_key)
    try:
        client.get_registered_model(model_name)
    except Exception:
        raise ValueError(f"Model '{model_name}' not found")
    client.delete_registered_model(model_name)
    print(f"Deleted '{model_name}'")


# ── download_model_weights ─────────────────────────────────────────────────────


def download_model_weights(
    model_name: str,
    version: str | int | None = None,
    dest_dir: str | Path = ".",
    tracking_uri: str | None = None,
    api_key: str | None = None,
) -> Path:
    """Pull model artefacts and return the local directory path.

    Thin alias for :func:`pull` used by the inference template on the compute node.
    """
    return pull(
        model_name,
        version=version,
        output_dir=dest_dir,
        tracking_uri=tracking_uri,
        api_key=api_key,
    )
