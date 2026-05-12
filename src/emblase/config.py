from pathlib import Path

from pydantic_settings import BaseSettings

_REPO_MODELS = Path(__file__).resolve().parent.parent.parent / "models"


def resolve_tiled_path(flag: str | None, env_default: str) -> str:
    """Combine a CLI flag value with an env-var default, pathlib-style.

    Rules (mirrors pathlib.Path / behaviour):
    - flag is None or ""  → return env_default as-is
    - flag starts with '/' → treat as absolute Tiled path; return flag (ignores env_default)
    - flag starts with './'  → strip the './' prefix and append to env_default
    - anything else (letter/digit/…) → append to env_default with '/'
    """
    if not flag:
        return env_default
    if flag.startswith("/"):
        return flag.lstrip("/")  # strip leading slash — Tiled paths have no leading slash
    base = env_default.rstrip("/")
    relative = flag[2:] if flag.startswith("./") else flag
    return f"{base}/{relative}" if base else relative


class Settings(BaseSettings):
    """Application settings, loaded from environment / .env file."""

    # Per-backend models directories — used for manually uploaded local models
    # and the MLflow artifact cache.  Set to absolute writable paths.
    # On Orion: persistent NFS, e.g. /nsls2/users/<user>/code/emblase/models
    # On NERSC: fast Lustre scratch, e.g. /pscratch/sd/<i>/<user>/emblase/models
    #           (30-day purge policy — re-push weights if purged)
    # Local default for both: <repo>/models
    orion_models_dir: Path = _REPO_MODELS
    nersc_models_dir: Path = _REPO_MODELS

    # Tiled
    tiled_server_uri: str = ""
    tiled_api_key: str = ""
    # Default Tiled container paths — used as base when --inputs/--output CLI flags
    # are relative (or omitted).  Override per-deployment in .env.
    tiled_input_container: str = ""  # e.g. smi/sandbox/confab26_demo/_inputs
    tiled_output_container: str = ""  # e.g. smi/sandbox/confab26_demo/output
    # Comma-separated Tiled access tags applied to every node written by write_output.
    # Example: "nsls2", "staff,nsls2"
    # If empty, no access_tags argument is passed and server defaults apply.
    tiled_access_tags: str = ""

    # MLflow — set EMBLASE_MLFLOW_TRACKING_URI to point at any compatible server.
    # For American Science Cloud (ASC): the tracking URI provided by the ASC portal.
    # For Azure ML: azureml://<region>.api.azureml.ms/mlflow/v1.0/subscriptions/...
    # For a local server: http://localhost:5000
    mlflow_tracking_uri: str = ""
    # API key for servers that require X-Api-Key header (e.g. AmSC MLflow).
    # Leave empty for Azure ML (uses Azure CLI credential instead).
    mlflow_api_key: str = ""
    # MLflow experiment name used when logging runs during push
    mlflow_experiment: str = "emblase-models"

    # Orion compute
    orion_api_url: str = "https://orion-api-staging.nsls2.bnl.gov"
    orion_api_key: str = ""
    orion_cluster: str = "orion"
    orion_project_dir: str = ""  # e.g. /nsls2/users/<user>/code/emblase
    orion_working_dir: str = ""  # e.g. /nsls2/users/<user>/code/emblase/jobs
    orion_home: str = ""  # e.g. /nsls2/users/<user>
    orion_account: str = "staff"
    orion_path: str = (
        "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/local/cuda/bin:/nsls2/software/bin"
    )
    # SSH access to the Orion login node for streaming job logs.
    # Host is derived from orion_api_url by default (same hostname, port 22).
    # Override with EMBLASE_ORION_SSH_HOST if the login node differs from the API host.
    orion_ssh_host: str = ""  # e.g. "orion-staging.nsls2.bnl.gov"
    orion_ssh_user: str = ""  # e.g. "jdoe"; defaults to local $USER

    # NERSC compute (IRI API + Shifter)
    nersc_api_uri: str = "https://api.iri.nersc.gov"
    nersc_api_token: str = ""
    nersc_resource_id: str = "perlmutter"
    nersc_working_dir: str = ""
    nersc_account: str = ""
    nersc_container_image: str = "ghcr.io/genematx/emblase:latest"
    nersc_time_limit: str = "00:30:00"
    nersc_constraint: str = ""  # empty = no constraint, let Perlmutter pick any GPU node
    # Queue/partition for job submission.
    # "shared"         → shared_gpu_ss11 / gpu_shared QOS — best for single-GPU jobs.
    # "debug"          → gpu_ss11 / gpu_debug QOS — fast dispatch, ≤ 30 min cap.
    # ""               → let the scheduler pick (lands on gpu_debug by default).
    # "premium"        → gpu_ss11 / gpu_premium QOS — faster turn-around, higher cost;
    #                    requires EMBLASE_NERSC_ACCOUNT=amsc006_g.
    # "express_amsc_g" → real-time GPU access via 32 reserved AMSC nodes; requires
    #                    EMBLASE_NERSC_ACCOUNT=amsc006_g.  Avoid >4 nodes or >2 h/job.
    # "express_amsc"   → same pool, CPU nodes; requires EMBLASE_NERSC_ACCOUNT=amsc006.
    nersc_queue: str = "shared"

    # Compute backend selection: "local", "orion", or "nersc"
    compute_backend: str = "local"

    # AmSC OpenMetadata catalog
    amsc_openmetadata_catalog_url: str = "https://api.american-science-cloud.org/api/current"
    amsc_openmetadata_token: str = ""
    amsc_openmetadata_catalog_name: str = ""  # e.g. "bnl-lse-demo-storage"
    amsc_openmetadata_parent_fqn: str = ""  # FQN of the parent ScientificWork to publish under

    # Chat app (AmSC LLM assistant)
    chatapp_url: str = "https://chat-amsc-dev.nsls2.bnl.gov"
    chatapp_token: str = ""
    chatapp_model: str = "openai/gpt-oss-20b"

    # Dashboard web server
    # Port the dashboard FastAPI server listens on (default 8765).
    # Override with EMBLASE_DASHBOARD_PORT in your .env file.
    dashboard_port: int = 8765
    # If true, the dashboard server auto-opens a browser tab on startup.
    dashboard_open_browser: bool = True
    # MLflow model name prefix shown in the Models section.
    mlflow_model_prefix: str = "bnl-nsls2-"

    model_config = {
        "env_file": Path(__file__).resolve().parent.parent.parent / ".env",
        "env_prefix": "EMBLASE_",
        "extra": "ignore",
    }


settings = Settings()
