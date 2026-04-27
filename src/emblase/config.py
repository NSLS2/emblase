from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from environment / .env file."""

    # Paths
    models_dir: Path = Path(__file__).resolve().parent.parent.parent / "models"

    # Tiled
    tiled_uri: str = ""
    tiled_api_key: str = ""

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
    # On Orion this should be a persistent path (e.g. /nsls2/users/ymatviych/.cache/emblase/models).
    # Falls back to ~/.cache/emblase/models if unset.
    model_cache_dir: str = ""

    # Orion compute
    orion_api_url: str = "https://orion-api-staging.nsls2.bnl.gov"
    orion_api_key: str = ""
    orion_cluster: str = "orion"
    orion_project_dir: str = "/nsls2/users/ymatviych/code/emblase"
    orion_working_dir: str = "/nsls2/users/ymatviych/code/emblase/jobs"
    orion_models_dir: str = "/nsls2/users/ymatviych/code/emblase/models"
    orion_home: str = "/nsls2/users/ymatviych"
    orion_account: str = "staff"
    orion_path: str = "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/local/cuda/bin:/nsls2/software/bin"

    # Compute backend selection: "local" or "orion"
    compute_backend: str = "local"

    model_config = {"env_file": ".env", "env_prefix": "EMBLASE_", "extra": "ignore"}


settings = Settings()
