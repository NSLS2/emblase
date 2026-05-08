from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from environment / .env file."""

    # Single models directory — used for both manually uploaded local models and
    # the MLflow artifact cache.  Set EMBLASE_MODELS_DIR to an absolute writable
    # path on the compute node.
    # On Orion: /nsls2/users/<user>/code/emblase/models  (persistent NFS)
    # On NERSC: /pscratch/sd/<i>/<user>/emblase/models   (Lustre, 30-day purge)
    # Local default: <repo>/models
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
    # "shared" → shared_gpu_ss11 / gpu_shared QOS — fastest for single-GPU jobs.
    # "debug"  → gpu_ss11 / gpu_debug QOS — fast, but capped at 30 min.
    # ""       → let the scheduler pick (lands on gpu_debug by default).
    nersc_queue: str = "shared"
    # Absolute path to the secrets file on the NERSC compute node (inside the container).
    # This file holds Tiled/MLflow credentials read by inference scripts at startup.
    # Create it once via:
    #   python scripts/submit_nersc.py secrets --path /global/u2/<i>/<user>/.emblase_secrets
    # $HOME (/global/u2/...) is bind-mounted read-only inside podman-hpc containers.
    # The preamble uses os.path.expanduser(), so both absolute paths and ~ are accepted.
    # Perlmutter home path convention: /global/u2/<first_letter>/<username>
    nersc_secrets_file: str = "~/.emblase_secrets"

    # Compute backend selection: "local", "orion", or "nersc"
    compute_backend: str = "local"

    model_config = {
        "env_file": Path(__file__).resolve().parent.parent.parent / ".env",
        "env_prefix": "EMBLASE_",
        "extra": "ignore",
    }


settings = Settings()
