"""Orion HPC compute backend — submits Slurm jobs via the Orion API."""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from ..config import settings
from .base import ComputeBackend, JobResult, JobStatus

# Map Slurm states to our JobStatus
_SLURM_STATE_MAP = {
    "PENDING": JobStatus.pending,
    "CONFIGURING": JobStatus.pending,
    "RUNNING": JobStatus.running,
    "COMPLETING": JobStatus.running,
    "COMPLETED": JobStatus.completed,
    "FAILED": JobStatus.failed,
    "CANCELLED": JobStatus.failed,
    "TIMEOUT": JobStatus.failed,
    "NODE_FAIL": JobStatus.failed,
    "OUT_OF_MEMORY": JobStatus.failed,
}


class OrionBackend(ComputeBackend):
    """Submit inference jobs to Orion (Slurm) via the Orion REST API.

    Workflow:
    1. Save input images as .npy to Orion's shared filesystem
    2. Generate a Python inference script
    3. Wrap it in an sbatch script
    4. Submit via Orion API
    5. Poll status; read results from .npy output file
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        cluster: str | None = None,
        working_dir: str | None = None,
        account: str | None = None,
    ):
        self.api_url = (api_url or settings.orion_api_url).rstrip("/")
        self.api_key = api_key or settings.orion_api_key
        self.cluster = cluster or settings.orion_cluster
        self.working_dir = working_dir or settings.orion_working_dir
        self.account = account or settings.orion_account
        self._jobs: dict[str, dict[str, Any]] = {}  # job_id -> metadata

    def _build_inference_script(
        self,
        job_dir: str,
        model_name: str,
        latent_dim: int,
        image_shape: tuple[int, ...],
    ) -> str:
        """Generate a Python script that loads the model and runs inference."""
        return f"""\
import sys
import os
import numpy as np
import torch

# Add model directory to path
models_dir = os.path.expanduser("~/code/mlex/models")
sys.path.insert(0, models_dir)

# Load input
images = np.load(os.path.join("{job_dir}", "input.npy"))
if images.ndim == 3:
    images = images[:, np.newaxis, :, :]
tensor = torch.from_numpy(images).float()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = tensor.to(device)

# Load model
model_name = "{model_name}"
latent_dim = {latent_dim}
image_size = {image_shape[-2:]}

if model_name == "vae":
    from vae.vae import ConvVAE
    model = ConvVAE(latent_dim=latent_dim, image_size=image_size).to(device)
    weights_path = os.path.join(models_dir, "vae", "vae_model_512_weights.npz")
elif model_name == "vit":
    from vit.vit import Autoencoder
    model = Autoencoder(latent_dim=latent_dim).to(device)
    weights_path = os.path.join(models_dir, "vit", "vit_model_weights.npz")
else:
    raise ValueError(f"Unknown model: {{model_name}}")

# Load weights
if os.path.exists(weights_path):
    data = np.load(weights_path, allow_pickle=True)
    state_dict = {{k: torch.from_numpy(data[k]) for k in data.files}}
    model.load_state_dict(state_dict, strict=False)

model.eval()

# Run inference
with torch.no_grad():
    if model_name == "vae":
        mu, logvar = model.encode(tensor)
        latent = mu.cpu().numpy()
    elif model_name == "vit":
        latent, _skip = model.encoder(tensor)
        latent = latent.cpu().numpy()

# Save output
np.save(os.path.join("{job_dir}", "output.npy"), latent)
print(f"Done. Output shape: {{latent.shape}}")
"""

    def _build_sbatch_script(
        self, job_dir: str, model_name: str, latent_dim: int, image_shape: tuple
    ) -> str:
        """Build the full sbatch script that runs the inference Python script."""
        python_code = self._build_inference_script(
            job_dir, model_name, latent_dim, image_shape
        )
        # Escape for embedding in bash heredoc
        return f"""\
#!/bin/bash
#SBATCH --job-name=mlex-{model_name}
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out

# Activate conda/environment if needed
# source activate mlex

cd {job_dir}

python3 -c '
{python_code}
'
"""

    async def submit(
        self,
        model_name: str,
        images: np.ndarray,
        latent_dim: int = 512,
        **kwargs: Any,
    ) -> str:
        job_tag = str(uuid.uuid4())[:8]
        job_dir = f"{self.working_dir}/job_{job_tag}"

        script = self._build_sbatch_script(
            job_dir, model_name, latent_dim, images.shape
        )

        # We need to create the job directory and save input on the remote filesystem.
        # For now, we submit a two-part job: first mkdir + save, then inference.
        # In practice the input data could be on shared storage already.
        # For simplicity, we embed the input data as base64 in a setup script.
        import base64
        import io

        buf = io.BytesIO()
        np.save(buf, images)
        b64_data = base64.b64encode(buf.getvalue()).decode()

        setup_and_run = f"""\
#!/bin/bash
#SBATCH --job-name=mlex-{model_name}
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out

mkdir -p {job_dir}
cd {job_dir}

# Decode input data
python3 -c "
import base64, numpy as np, io
data = base64.b64decode('{b64_data}')
arr = np.load(io.BytesIO(data))
np.save('input.npy', arr)
print(f'Input saved: {{arr.shape}}')
"

# Run inference
python3 << 'INFERENCE_EOF'
{self._build_inference_script(job_dir, model_name, latent_dim, images.shape)}
INFERENCE_EOF
"""

        payload = {
            "script": setup_and_run,
            "working_dir_path": job_dir,
            "overrides": {
                "tres_per_job": "gres/gpu:1",
                "account": self.account,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.api_url}/api/v1/compute/{self.cluster}/jobs",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                },
            )
            resp.raise_for_status()
            resp_data = resp.json()

        # Orion API returns the Slurm job ID
        slurm_job_id = str(resp_data.get("job_id", job_tag))

        self._jobs[slurm_job_id] = {
            "job_dir": job_dir,
            "model_name": model_name,
            "tag": job_tag,
        }

        return slurm_job_id

    async def status(self, job_id: str) -> JobStatus:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self.api_url}/api/v1/compute/{self.cluster}/jobs/{job_id}",
                headers={"x-api-key": self.api_key},
            )
            resp.raise_for_status()
            data = resp.json()

        slurm_state = data.get("job_state", ["UNKNOWN"])
        if isinstance(slurm_state, list):
            slurm_state = slurm_state[0] if slurm_state else "UNKNOWN"
        return _SLURM_STATE_MAP.get(slurm_state, JobStatus.pending)

    async def result(self, job_id: str) -> JobResult:
        st = await self.status(job_id)
        if st != JobStatus.completed:
            return JobResult(job_id=job_id, status=st)

        meta = self._jobs.get(job_id, {})
        job_dir = meta.get("job_dir")
        if not job_dir:
            return JobResult(
                job_id=job_id,
                status=JobStatus.failed,
                error="Job metadata lost (server restart?)",
            )

        # TODO: Read output.npy from remote filesystem.
        # For now this assumes shared filesystem access.
        output_path = Path(job_dir).expanduser() / "output.npy"
        if output_path.exists():
            latent = np.load(str(output_path))
            return JobResult(
                job_id=job_id, status=JobStatus.completed, latent_vectors=latent
            )
        else:
            return JobResult(
                job_id=job_id,
                status=JobStatus.completed,
                error=f"Output file not found at {output_path}. Job may have failed — check logs.",
            )

    async def cancel(self, job_id: str) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.delete(
                f"{self.api_url}/api/v1/compute/{self.cluster}/jobs/{job_id}",
                headers={"x-api-key": self.api_key},
            )
            resp.raise_for_status()
