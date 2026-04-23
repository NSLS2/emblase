"""Quick smoke test: run local backend with dummy images."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np


async def main():
    from mlex.compute.local import LocalBackend

    backend = LocalBackend()

    # Use 512x512 to match pre-trained weight dimensions
    images = np.random.rand(2, 512, 512).astype(np.float32)

    print("Submitting VAE inference job...")
    job_id = await backend.submit(
        model_name="vae",
        images=images,
        latent_dim=512,
    )

    result = await backend.result(job_id)
    print(f"Job {job_id}: status={result.status}")
    if result.latent_vectors is not None:
        print(f"  Latent vectors shape: {result.latent_vectors.shape}")
    if result.error:
        print(f"  Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
