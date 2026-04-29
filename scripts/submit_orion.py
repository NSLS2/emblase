"""Manage Orion jobs: connectivity test, inference submission, status, cancel."""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emblase.compute.orion import OrionBackend, OrionClient, _SLURM_STATE_MAP  # noqa: E402

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


async def _infer(args):
    backend = OrionBackend()
    print(f"Working dir : {backend.working_dir}")
    print(f"Models dir  : {backend.models_dir}")

    submit_kwargs = dict(
        model_name=args.model,
        batch_size=args.batch_size,
        output=args.output or "",
        thumb_mode=args.thumb_mode,
    )

    if args.npy_file:
        images = np.load(args.npy_file)
        print(f"Loaded from {args.npy_file}: {images.shape}  dtype={images.dtype}")
        submit_kwargs["images"] = images
    elif args.npy_path:
        submit_kwargs["npy_path"] = args.npy_path
    elif args.inputs:
        # Parse each entry: bare "path/to/node" or "path/to/node:slice_expr"
        parsed = []
        for e in args.inputs:
            if ":" in e:
                path, slc = e.split(":", 1)
                parsed.append((path, slc))
            else:
                parsed.append(e)
        submit_kwargs["inputs"] = parsed
    else:
        images = np.random.rand(args.n_images, args.image_size, args.image_size).astype(np.float32)
        print(f"Dummy images: {images.shape}  dtype={images.dtype}")
        submit_kwargs["images"] = images

    print(f"\nSubmitting {args.model!r} inference job to Orion...")
    job_id = await backend.submit(**submit_kwargs)
    print(f"Job submitted: {job_id}")

    if args.no_wait:
        print("(not waiting — use 'status <id>' to check)")
        return

    print("Waiting for job to complete (polling every 5s)...")
    import time
    t0 = time.monotonic()
    while True:
        await asyncio.sleep(5)
        elapsed = int(time.monotonic() - t0)
        async with OrionClient() as client:
            info = await client.get_job(int(job_id))
        print(f"  [{elapsed:>4}s] state={info.state} node={info.node or '(queued)'}")
        if info.state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"):
            break

    from emblase.compute.base import JobStatus
    final_status = _SLURM_STATE_MAP.get(info.state, JobStatus.failed)
    print(f"\nJob {job_id} finished: {final_status.value}")
    if final_status != JobStatus.completed:
        print("Job did not complete successfully — check the log on the Orion filesystem.")
    elif not args.output:
        print(f"Output: {backend.working_dir}/job_{job_id}/output.npy")


async def _run_script(script: str, working_dir: str, gpu: bool, wait: bool):
    overrides = {"account": "staff"}
    if gpu:
        overrides["tres_per_job"] = "gres/gpu:1"
    async with OrionClient() as client:
        print(f"Submitting job to {client.api_url} ({client.cluster})...")
        job_id = await client.submit_job(
            script=script, working_dir=working_dir, overrides=overrides
        )
        print(f"Job submitted: {job_id}")
        if wait:
            info = await client.wait_for_job(job_id)
            print(f"State: {info.state}  Node: {info.node}")
            print(f"Stdout: {info.stdout}")
        else:
            print("(not waiting — use 'status <id>' to check)")


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
        help="Path to a bash script, or 'test' for the built-in connectivity check",
    )
    run_p.add_argument("--workdir", default="/tmp")
    run_p.add_argument("--gpu", action="store_true")
    run_p.add_argument("--no-wait", action="store_true")

    # -- infer subcommand --
    infer_p = sub.add_parser("infer", help="Submit an inference job")
    infer_p.add_argument(
        "--model",
        default="vae",
        help=(
            "Model name. Use a short architecture name ('vae', 'vit') to load "
            "from local weights, or an MLflow registry name to pull from the registry."
        ),
    )
    infer_p.add_argument("--image-size", type=int, default=512)
    infer_p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Images per encode call on the node. Default 8; reduce if OOM on large images.",
    )
    infer_p.add_argument("--no-wait", action="store_true")
    infer_p.add_argument(
        "--output",
        metavar="TILED_PATH",
        default="",
        help=(
            "Tiled path to write embeddings into (e.g. 'proposal/embeddings/scan1'). "
            "If omitted, output.npy is saved in the job working directory only."
        ),
    )

    # image sources (mutually exclusive; if none given, dummy data is used)
    src = infer_p.add_mutually_exclusive_group()
    src.add_argument(
        "--npy-file",
        metavar="PATH",
        help="Local .npy file to upload and run on Orion",
    )
    src.add_argument(
        "--npy-path",
        metavar="PATH",
        help="Absolute path to a .npy file already on Orion (symlinked into job dir)",
    )
    src.add_argument(
        "--inputs",
        nargs="+",
        metavar="PATH[:SLICE]",
        help=(
            "One or more Tiled entries to read as input. Each is a slash-separated "
            "path optionally followed by a colon and a numpy-style slice, e.g. "
            "'proposal/scan' or 'proposal/scan:0:10'"
        ),
    )
    infer_p.add_argument(
        "--n-images",
        type=int,
        default=2,
        help="Number of dummy images (ignored if an image source is given)",
    )
    infer_p.add_argument(
        "--thumb-mode",
        default="default",
        choices=["default", "log"],
        help=(
            "Thumbnail generation mode. "
            "'default': nearest-neighbour resize of the full frame. "
            "'log': crop ROI (rows 0:180, cols 220:400), clip negatives, "
            "apply log1p, then resize. Recommended for X-ray photon-count data."
        ),
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
        asyncio.run(_infer(args))

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
