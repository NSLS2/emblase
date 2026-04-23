"""Manage Orion jobs: connectivity test, inference submission, status, cancel."""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emblase.compute.orion import OrionBackend, OrionClient  # noqa: E402

CONNECTIVITY_SCRIPT = """\
#!/bin/bash
#SBATCH --job-name=emblase-test
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

echo '=== Emblase connectivity test ==='
hostname
whoami
pwd
ls ~/code/emblase/ 2>/dev/null || echo 'No ~/code/emblase directory found'
which python3
python3 -c 'import torch; print(f"torch={torch.__version__}, \
    cuda={torch.cuda.is_available()}")' 2>/dev/null || echo 'torch not available'
nvidia-smi 2>/dev/null || echo 'no GPU'
echo '=== done ==='
"""


async def _run_script(script: str, working_dir: str, gpu: bool, wait: bool):
    overrides = {"account": "staff"}
    if gpu:
        overrides["tres_per_job"] = "gres/gpu:1"

    async with OrionClient() as client:
        print(f"Submitting job to {client.api_url} ({client.cluster})...")
        job_id = await client.submit_job(
            script=script,
            working_dir=working_dir,
            overrides=overrides,
        )
        print(f"Job submitted: {job_id}")

        if wait:
            info = await client.wait_for_job(job_id)
            print(f"State: {info.state}  Node: {info.node}")
            print(f"Stdout: {info.stdout}")
        else:
            print("(not waiting — use 'status <id>' to check)")


async def _infer(
    model: str, n_images: int, image_size: int, latent_dim: int, wait: bool
):
    images = np.random.rand(n_images, image_size, image_size).astype(np.float32)
    print(f"Dummy images: {images.shape}  dtype={images.dtype}")

    backend = OrionBackend()
    print(f"Working dir : {backend.working_dir}")
    print(f"Models dir  : {backend.models_dir}")

    print(f"\nSubmitting {model!r} inference job to Orion...")
    job_id = await backend.submit(
        model_name=model, images=images, latent_dim=latent_dim
    )
    print(f"Job submitted: {job_id}")

    if not wait:
        print("(not waiting — use 'status <id>' to check)")
        return

    print("Waiting for job to complete...")
    final_status = await backend.wait(job_id)
    print(f"\nJob {job_id} finished: {final_status.value}")

    if final_status.value != "completed":
        print(
            "Job did not complete successfully — check the stdout on the Orion filesystem."
        )


def main():
    parser = argparse.ArgumentParser(description="Manage Orion jobs for Emblase")
    sub = parser.add_subparsers(dest="command")

    # -- test / run subcommand --
    run_p = sub.add_parser(
        "run", aliases=["test"], help="Submit a connectivity test or custom script"
    )
    run_p.add_argument(
        "script_file",
        nargs="?",
        default="test",
        help="Path to a bash script to submit, or 'test' for the built-in connectivity check",
    )
    run_p.add_argument("--workdir", default="/tmp", help="Remote working directory")
    run_p.add_argument("--gpu", action="store_true", help="Request a GPU node")
    run_p.add_argument(
        "--no-wait", action="store_true", help="Don't wait for completion"
    )

    # -- infer subcommand --
    infer_p = sub.add_parser("infer", help="Submit an inference job with dummy images")
    infer_p.add_argument("--model", choices=["vae", "vit"], default="vae")
    infer_p.add_argument("--n-images", type=int, default=2)
    infer_p.add_argument("--image-size", type=int, default=512)
    infer_p.add_argument("--latent-dim", type=int, default=512)
    infer_p.add_argument(
        "--no-wait", action="store_true", help="Don't wait for completion"
    )

    # -- status subcommand --
    status_p = sub.add_parser("status", help="Check job status")
    status_p.add_argument("job_id", type=int)

    # -- cancel subcommand --
    cancel_p = sub.add_parser("cancel", help="Cancel a job")
    cancel_p.add_argument("job_id", type=int)

    args = parser.parse_args()

    if args.command in ("run", "test"):
        script = (
            CONNECTIVITY_SCRIPT
            if args.script_file == "test"
            else Path(args.script_file).read_text()
        )
        asyncio.run(
            _run_script(
                script, working_dir=args.workdir, gpu=args.gpu, wait=not args.no_wait
            )
        )

    elif args.command == "infer":
        asyncio.run(
            _infer(
                model=args.model,
                n_images=args.n_images,
                image_size=args.image_size,
                latent_dim=args.latent_dim,
                wait=not args.no_wait,
            )
        )

    elif args.command == "status":

        async def _status():
            async with OrionClient() as client:
                info = await client.get_job(args.job_id)
                print(f"Job {info.job_id}: state={info.state} node={info.node}")
                print(f"  stdout: {info.stdout}")

        asyncio.run(_status())

    elif args.command == "cancel":

        async def _cancel():
            async with OrionClient() as client:
                await client.cancel_job(args.job_id)
                print(f"Job {args.job_id} cancelled")

        asyncio.run(_cancel())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
