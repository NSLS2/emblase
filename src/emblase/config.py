from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from environment / .env file."""

    # Paths
    models_dir: Path = Path(__file__).resolve().parent.parent.parent / "models"

    # Orion compute
    orion_api_url: str = "https://orion-api-staging.nsls2.bnl.gov"
    orion_api_key: str = ""
    orion_cluster: str = "orion"
    orion_project_dir: str = "/nsls2/users/ymatviych/code/emblase"
    orion_working_dir: str = "/nsls2/users/ymatviych/code/emblase/jobs"
    orion_models_dir: str = "/nsls2/users/ymatviych/code/emblase/models"
    orion_account: str = "staff"

    # Compute backend selection: "local" or "orion"
    compute_backend: str = "local"

    model_config = {"env_file": ".env", "env_prefix": "EMBLASE_", "extra": "ignore"}


settings = Settings()
