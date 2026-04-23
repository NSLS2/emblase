"""Submit a simple test job to Orion to verify API connectivity and access."""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx


ORION_API_URL = "https://orion-api-staging.nsls2.bnl.gov"
API_KEY = os.environ["COMPUTE_STAGING_API_KEY"]
CLUSTER = "orion"


async def submit_test_job():
    payload = {
        "script": (
            "#!/bin/bash\n"
            "#SBATCH --job-name=mlex-test\n"
            "#SBATCH --time=0-00:05:00\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --ntasks-per-node=1\n"
            "#SBATCH --error=%x-%j.err\n"
            "#SBATCH --output=%x-%j.out\n"
            "\n"
            "echo '=== MLEX connectivity test ==='\n"
            "hostname\n"
            "whoami\n"
            "pwd\n"
            "ls ~/code/mlex/ 2>/dev/null || echo 'No ~/code/mlex directory found'\n"
            "which python3\n"
            "python3 -c 'import torch; print(f\"torch={torch.__version__}, cuda={torch.cuda.is_available()}\")' 2>/dev/null || echo 'torch not available'\n"
            "nvidia-smi 2>/dev/null || echo 'no GPU'\n"
            "echo '=== done ==='\n"
        ),
        "working_dir_path": "/nsls2/users/ymatviychuk/code/mlex",
        "overrides": {
            "tres_per_job": "gres/gpu:1",
            "account": "staff",
        },
    }

    async with httpx.AsyncClient(timeout=30.0, verify=True) as client:
        print(f"Submitting test job to {ORION_API_URL}...")
        resp = await client.post(
            f"{ORION_API_URL}/api/v1/compute/{CLUSTER}/jobs",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": API_KEY,
            },
        )
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")

        if resp.status_code == 200:
            data = resp.json()
            job_id = data.get("job_id")
            print(f"\nJob submitted! ID: {job_id}")
            print(
                f"Check status: curl -H 'x-api-key: $COMPUTE_STAGING_API_KEY' {ORION_API_URL}/api/v1/compute/{CLUSTER}/jobs/{job_id}"
            )


if __name__ == "__main__":
    asyncio.run(submit_test_job())
