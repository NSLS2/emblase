from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from environment / .env file."""

    # Paths
    models_dir: Path = Path(__file__).resolve().parent.parent.parent / "models"

    # Tiled
    tiled_server_uri: str = ""
    tiled_api_key: str = ""
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
    # Local directory where MLflow model artifacts are cached, keyed by name+version.
    # On Orion this should be a persistent path (e.g. /nsls2/users/<user>/.cache/emblase/models).
    # Falls back to ~/.cache/emblase/models if unset.
    model_cache_dir: str = ""

    # Orion compute
    orion_api_url: str = "https://orion-api-staging.nsls2.bnl.gov"
    orion_api_key: str = ""
    orion_cluster: str = "orion"
    orion_project_dir: str = ""  # e.g. /nsls2/users/<user>/code/emblase
    orion_working_dir: str = ""  # e.g. /nsls2/users/<user>/code/emblase/jobs
    orion_models_dir: str = ""  # e.g. /nsls2/users/<user>/code/emblase/models
    orion_home: str = ""  # e.g. /nsls2/users/<user>
    orion_account: str = "staff"
    orion_path: str = (
        "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/local/cuda/bin:/nsls2/software/bin"
    )
    # SSH access to the Orion login node for streaming job logs.
    # Host is derived from orion_api_url by default (same hostname, port 22).
    # Override with EMBLASE_ORION_SSH_HOST if the login node differs from the API host.
    orion_ssh_host: str = ""  # e.g. "orion-staging.nsls2.bnl.gov"
    orion_ssh_user: str = ""  # e.g. "ymatviych"; defaults to local $USER

    # NERSC compute (IRI API + Shifter)
    nersc_api_uri: str = "https://api.iri.nersc.gov"
    nersc_api_token: str = ""
    nersc_resource_id: str = "perlmutter"
    nersc_filesystem_resource_id: str = "scratch"  # IRI resource for filesystem ops (mkdir/upload)
    nersc_working_dir: str = "/pscratch/sd/y/confab26_demo/emblase/jobs"
    nersc_models_dir: str = "/pscratch/sd/y/confab26_demo/emblase/models"
    nersc_account: str = ""
    nersc_container_image: str = "ghcr.io/genematx/emblase:latest"
    nersc_time_limit: str = "00:30:00"
    nersc_constraint: str = "gpu"

    # Compute backend selection: "local", "orion", or "nersc"
    compute_backend: str = "local"

    model_config = {
        "env_file": Path(__file__).resolve().parent.parent.parent / ".env",
        "env_prefix": "EMBLASE_",
        "extra": "ignore",
    }


settings = Settings()
