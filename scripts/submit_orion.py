"""Manage Orion jobs: connectivity test, batch inference, status, cancel.

Subcommands
-----------
infer   Submit a batch inference job (encode all frames, write to Tiled).
run     Submit a connectivity test or an arbitrary bash script.
status  Poll the state of a job by ID.
cancel  Cancel a running job.

Typical workflow — quick iteration on a complete run
-----------------------------------------------------
Push commits, pull on Orion, then submit::

    BRANCH=$(git branch --show-current)
    git push origin $BRANCH
    ssh orion-staging.nsls2.bnl.gov "cd /nsls2/users/ymatviych/code/emblase && git fetch && git checkout $BRANCH && git pull"

    python scripts/submit_orion.py infer \\
        --model   bnl-nsls2-smi-vit \\
        --run     smi/sandbox/confab26_demo/inputs/run_1086139 \\
        --output  smi/sandbox/confab26_demo/results/run_1086139_vit \\
        --batch-size 1 \\
        --thumb-mode logroi \\
        --param temperature:primary.LinkamThermal_temperature_current:float:°C \\
        --param piezo_x:primary.piezo_x:float:μm

The job reads all frames from ``run_path/primary/<image_key>``, encodes them,
writes a ``LatentSpaceEmbedding`` container to ``--output``, and stores the
requested scalar params in the ``_index`` table alongside each embedding.

``--projector`` controls projector (UMAP) projection behaviour::

    # fit from scratch over all embeddings (default when --projector is omitted)
    python scripts/submit_orion.py infer --model bnl-nsls2-smi-vit ...

    # use a saved approximator by name (checks models_dir/<name> → MLflow)
    python scripts/submit_orion.py infer --model bnl-nsls2-smi-vit \\
        --projector umap_approx ...

    # disable projections entirely (write NaN)
    python scripts/submit_orion.py infer --model bnl-nsls2-smi-vit \\
        --projector false ...

Check the job log::

    ssh orion-staging.nsls2.bnl.gov \\
        "tail -50 /nsls2/users/ymatviych/orion_jobs/slurm-<jobid>.out"

--param syntax
--------------
    --param name:source:dtype:units

``source`` must be ``primary.<array_key>`` — a scalar stream in the run's primary
event stream aligned 1-to-1 with image frames.
``dtype`` defaults to ``float``; ``units`` defaults to empty string.
Repeat for multiple params.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emblase.compute.base import JobStatus  # noqa: E402
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


def _parse_param_specs(param_strs: list[str]) -> dict | None:
    """Parse --param name:source[:dtype[:units]] entries into a ParamSpec dict."""
    if not param_strs:
        return None
    specs: dict = {}
    for s in param_strs:
        parts = s.split(":")
        if len(parts) < 2:
            sys.exit(f"Invalid --param spec {s!r}: expected name:source[:dtype[:units]]")
        name = parts[0]
        source = parts[1]
        dtype = parts[2] if len(parts) > 2 else "float"
        units = parts[3] if len(parts) > 3 else ""
        specs[name] = {"source": source, "dtype": dtype, "units": units}
    return specs


async def _infer(args):
    backend = OrionBackend()
    print(f"Working dir : {backend.working_dir}")
    print(f"Models dir  : {backend.models_dir}")

    param_specs = _parse_param_specs(args.params)
    projector = args.projector
    classifier = args.classifier

    submit_kwargs = dict(
        model_name=args.model,
        batch_size=args.batch_size,
        output=args.output or "",
        thumb_mode=args.thumb_mode,
        mlflow_version=args.mlflow_version,
        param_specs=param_specs,
        projector=projector,
        classifier=classifier,
    )

    if args.run:
        submit_kwargs["run_path"] = args.run
        submit_kwargs["image_key"] = args.image_key
    elif args.npy_file:
        images = np.load(args.npy_file)
        print(f"Loaded from {args.npy_file}: {images.shape}  dtype={images.dtype}")
        submit_kwargs["images"] = images
    elif args.npy_path:
        submit_kwargs["npy_path"] = args.npy_path
    elif args.inputs:
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
    t0 = time.monotonic()
    while True:
        await asyncio.sleep(5)
        elapsed = int(time.monotonic() - t0)
        async with OrionClient() as client:
            info = await client.get_job(int(job_id))
        print(f"  [{elapsed:>4}s] state={info.state} node={info.node or '(queued)'}")
        if _SLURM_STATE_MAP.get(info.state) in (JobStatus.completed, JobStatus.failed):
            break

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
    parser = argparse.ArgumentParser(
        description="Manage Orion jobs for Emblase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    infer_p = sub.add_parser("infer", help="Submit a batch inference job")
    infer_p.add_argument(
        "--model",
        default="vae",
        metavar="MODEL_NAME",
        help=(
            "Model name. Use a short architecture name ('vae', 'vit') to load "
            "from local weights, or an MLflow registry name (e.g. 'bnl-nsls2-smi-vit') "
            "to pull from the registry."
        ),
    )
    infer_p.add_argument(
        "--mlflow-version",
        default="",
        metavar="VERSION",
        help="MLflow model version string (default: latest).",
    )
    infer_p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Images per encode call on the node (default: 1). "
            "Reduce if OOM on large images; SMI frames (619×1475) are safe at 1."
        ),
    )
    infer_p.add_argument("--no-wait", action="store_true")
    infer_p.add_argument(
        "--output",
        metavar="TILED_PATH",
        default="",
        help=(
            "Tiled path to write the LatentSpaceEmbedding container "
            "(e.g. 'smi/sandbox/confab26_demo/results/run_1086139_vit'). "
            "If omitted, output.npy is saved in the job working directory only."
        ),
    )
    infer_p.add_argument(
        "--thumb-mode",
        default="logroi",
        choices=["default", "logroi"],
        help=(
            "Thumbnail generation mode (default: logroi). "
            "'default': nearest-neighbour resize of the full frame. "
            "'logroi': crop ROI (rows 0:180, cols 220:400), clip negatives, "
            "apply log1p, then resize. Recommended for X-ray photon-count data."
        ),
    )
    infer_p.add_argument(
        "--image-key",
        default="pil900KW_image",
        metavar="KEY",
        help="Array key within the primary stream (default: pil900KW_image). Used with --run.",
    )
    infer_p.add_argument(
        "--param",
        action="append",
        default=[],
        dest="params",
        metavar="name:source[:dtype[:units]]",
        help=(
            "Scalar parameter to store alongside each embedding in the _index table. "
            "source must be 'primary.<array_key>' — a scalar stream aligned 1-to-1 "
            "with image frames. dtype defaults to 'float', units to ''. "
            "Repeat for multiple params, e.g.: "
            "--param temperature:primary.LinkamThermal_temperature_current:float:°C "
            "--param piezo_x:primary.piezo_x:float:μm"
        ),
    )
    infer_p.add_argument(
        "--projector",
        default=None,
        metavar="NAME|false",
        help=(
            "Projector (UMAP) projection behaviour. "
            "Omit (default): fit UMAP from scratch over all embeddings after encoding. "
            "NAME: resolve a saved approximator by name — checks models_dir/<NAME> on the "
            "node, then falls back to MLflow (e.g. --projector umap_approx). "
            "'false' or '0': skip projections entirely and write NaN."
        ),
    )

    # image sources (mutually exclusive; if none given, dummy data is used)
    src = infer_p.add_mutually_exclusive_group()
    src.add_argument(
        "--run",
        metavar="TILED_PATH",
        default="",
        help=(
            "Tiled path to a BlueskyRun container "
            "(e.g. 'smi/sandbox/confab26_demo/inputs/run_1086139'). "
            "Frames are read from run/primary/<image_key>. "
            "Required for --param support. Preferred over --inputs."
        ),
    )
    src.add_argument(
        "--npy-file",
        metavar="PATH",
        help="Local .npy file to upload and run on Orion (no param support).",
    )
    src.add_argument(
        "--npy-path",
        metavar="PATH",
        help="Absolute path to a .npy file already on Orion (no param support).",
    )
    src.add_argument(
        "--inputs",
        nargs="+",
        metavar="PATH[:SLICE]",
        help=(
            "One or more raw Tiled array paths, optionally with a slice suffix "
            "(e.g. 'smi/sandbox/.../pil900KW_image' or 'proposal/scan:0:10'). "
            "No param support; prefer --run for BlueskyRun data."
        ),
    )
    infer_p.add_argument(
        "--n-images",
        type=int,
        default=2,
        metavar="N",
        help="Number of dummy images when no image source is given (default: 2).",
    )
    infer_p.add_argument("--image-size", type=int, default=512, metavar="PX")

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
