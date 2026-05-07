"""Manage Orion jobs: connectivity test, batch inference, status, cancel, logs.

Subcommands
-----------
infer   Submit a batch inference job (encode all frames, write to Tiled).
run     Submit a connectivity test or an arbitrary bash script.
status  Poll the state of a job by ID.
cancel  Cancel a running job.
logs    Stream the Slurm log of a job live from the Orion login node via SSH.

Typical workflow
----------------
Push commits, pull on Orion, then submit::

    BRANCH=$(git branch --show-current)
    git push origin $BRANCH
    ssh orion-staging.nsls2.bnl.gov \\
        "cd $EMBLASE_ORION_PROJECT_DIR && git fetch && git checkout $BRANCH && git pull"

    python scripts/submit_orion.py infer \\
        --model      bnl-nsls2-smi-vit \\
        --run        smi/sandbox/confab26_demo/inputs/run_1086139 \\
        --output     smi/sandbox/confab26_demo/results/run_1086139_vit \\
        --batch-size 1 \\
        --thumb-mode logroi \\
        --param      temperature:primary/LinkamThermal_temperature_current:float:°C \\
        --param      piezo_x:primary/piezo_x:float:μm

By default ``infer`` streams the Slurm log via SSH while polling job state.
Pass ``--no-wait`` to detach immediately.  Tail any job later with::

    python scripts/submit_orion.py logs 999
    python scripts/submit_orion.py logs 999 --tail 100

--param syntax
--------------
    --param name:source[:dtype[:units]]

``source`` is ``<stream>/<array_key>`` (e.g. ``primary/LinkamThermal_temperature_current``).
``dtype`` defaults to ``float``; ``units`` defaults to empty string.
Repeat for multiple params.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emblase.compute import OrionBackend, OrionClient, parse_param_specs  # noqa: E402
from emblase.compute.base import JobStatus  # noqa: E402
from emblase.compute.orion import _SLURM_STATE_MAP  # noqa: E402

CONNECTIVITY_SCRIPT = """\
#!/bin/bash
#SBATCH --job-name=emblase-test
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

echo '=== Emblase connectivity test ==='
hostname && whoami && pwd
ls ~/code/emblase/ 2>/dev/null || echo 'No ~/code/emblase directory found'
python3 -c 'import torch; print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")' \
    2>/dev/null || echo 'torch not available'
nvidia-smi 2>/dev/null || echo 'no GPU'
echo '=== done ==='
"""


async def _infer(args: argparse.Namespace) -> None:
    backend = OrionBackend()
    print(f"Working dir : {backend.working_dir}")
    print(f"Models dir  : {backend.models_dir}")

    submit_kwargs: dict = dict(
        model_name=args.model,
        batch_size=args.batch_size,
        output=args.output or "",
        thumb_mode=args.thumb_mode,
        mlflow_version=args.mlflow_version,
        param_specs=parse_param_specs(args.params),
        projector=args.projector,
        classifier=args.classifier,
    )

    if args.run:
        submit_kwargs.update(run_path=args.run, image_key=args.image_key)
    elif args.npy_file:
        images = np.load(args.npy_file)
        print(f"Loaded {args.npy_file}: {images.shape}  dtype={images.dtype}")
        submit_kwargs["images"] = images
    elif args.npy_path:
        submit_kwargs["npy_path"] = args.npy_path
    elif args.inputs:
        submit_kwargs["inputs"] = [tuple(e.split(":", 1)) if ":" in e else e for e in args.inputs]
    else:
        images = np.random.rand(args.n_images, args.image_size, args.image_size).astype(np.float32)
        print(f"Dummy images: {images.shape}  dtype={images.dtype}")
        submit_kwargs["images"] = images

    print(f"\nSubmitting {args.model!r} inference job to Orion ...")
    job_id = await backend.submit(**submit_kwargs)
    print(f"Job submitted: {job_id}")
    print(f"  → stream log: pixi run orion logs {job_id}")

    if args.no_wait:
        print("(not waiting — use 'status <id>' or 'logs <id>' to check)")
        return

    print("Polling job state every 10 s ...")
    await asyncio.to_thread(backend.monitor_job, job_id)

    final = _SLURM_STATE_MAP.get(
        (await backend.client.get_job(int(job_id))).state, JobStatus.failed
    )
    print(f"\nJob {job_id} finished: {final.value}")
    if final != JobStatus.completed:
        print("Job did not complete successfully.")
    elif not args.output:
        print(f"Output: {backend.working_dir}/job_{job_id}/output.npy")


async def _run_script(script: str, working_dir: str, gpu: bool, wait: bool) -> None:
    overrides: dict = {"account": "staff"}
    if gpu:
        overrides["tres_per_job"] = "gres/gpu:1"
    async with OrionClient() as client:
        print(f"Submitting to {client.api_url} ({client.cluster}) ...")
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        default="vae",
        metavar="MODEL_NAME",
        help="Model name: short architecture ('vae', 'vit') or MLflow registry name.",
    )
    p.add_argument(
        "--mlflow-version",
        default="",
        metavar="VERSION",
        help="MLflow model version (default: latest).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Images per encode call on the node (default: 1).",
    )
    p.add_argument(
        "--thumb-mode",
        default="logroi",
        choices=["default", "logroi"],
        help="Thumbnail mode (default: logroi).",
    )
    p.add_argument(
        "--image-key",
        default="primary/pil900KW_image",
        metavar="KEY",
        help="Slash-separated image-array key, e.g. 'primary/pil900KW_image'.",
    )
    p.add_argument(
        "--param",
        action="append",
        default=[],
        dest="params",
        metavar="name:source[:dtype[:units]]",
        help="Scalar param to store per embedding. Repeat for multiple.",
    )
    p.add_argument(
        "--projector",
        default=None,
        metavar="NAME|false",
        help="Projector: omit=scratch, NAME=saved approx, false=NaN.",
    )
    p.add_argument(
        "--classifier",
        default=None,
        metavar="NAME",
        help="Classifier name (omit = no labels).",
    )
    p.add_argument(
        "--output",
        default="",
        metavar="TILED_PATH",
        help="Tiled path for the output LatentSpaceEmbedding container.",
    )
    p.add_argument("--no-wait", action="store_true", help="Return immediately after submission.")


def main() -> None:
    import logging

    parser = argparse.ArgumentParser(
        description="Manage Orion jobs for Emblase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    sub = parser.add_subparsers(dest="command")

    # run / test
    run_p = sub.add_parser(
        "run",
        aliases=["test"],
        help="Submit a connectivity test or custom bash script.",
    )
    run_p.add_argument(
        "script_file",
        nargs="?",
        default="test",
        help="Path to a bash script, or 'test' for the built-in check.",
    )
    run_p.add_argument("--workdir", default="/tmp")
    run_p.add_argument("--gpu", action="store_true")
    run_p.add_argument("--no-wait", action="store_true")

    # infer
    infer_p = sub.add_parser("infer", help="Submit a batch inference job.")
    _add_model_args(infer_p)
    src = infer_p.add_mutually_exclusive_group()
    src.add_argument(
        "--run",
        default="",
        metavar="TILED_PATH",
        help="BlueskyRun Tiled path; frames read from run/<image_key>.",
    )
    src.add_argument(
        "--npy-file",
        metavar="PATH",
        help="Local .npy file to upload (no param support).",
    )
    src.add_argument(
        "--npy-path",
        metavar="PATH",
        help="Absolute .npy path already on Orion (no param support).",
    )
    src.add_argument(
        "--inputs",
        nargs="+",
        metavar="PATH[:SLICE]",
        help="Raw Tiled array paths (no param support).",
    )
    infer_p.add_argument(
        "--n-images",
        type=int,
        default=2,
        metavar="N",
        help="Dummy image count when no source is given (default: 2).",
    )
    infer_p.add_argument("--image-size", type=int, default=512, metavar="PX")

    # status / cancel
    for name, help_ in [("status", "Check job status."), ("cancel", "Cancel a job.")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("job_id", type=int)

    # logs
    logs_p = sub.add_parser("logs", help="Stream the Slurm log via SSH.")
    logs_p.add_argument("job_id", type=int)
    logs_p.add_argument(
        "--tail",
        type=int,
        default=50,
        metavar="N",
        help="Existing lines to show before following (default: 50).",
    )
    logs_p.add_argument("--host", default="", metavar="HOSTNAME", help="SSH hostname override.")
    logs_p.add_argument("--user", default="", metavar="USERNAME", help="SSH username override.")

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx/httpcore/hpack log every request at INFO; suppress unless DEBUG
    if args.log_level != "DEBUG":
        for _noisy in ("httpx", "httpcore", "hpack"):
            logging.getLogger(_noisy).setLevel(logging.WARNING)

    if args.command in ("run", "test"):
        script = (
            CONNECTIVITY_SCRIPT
            if args.script_file == "test"
            else Path(args.script_file).read_text()
        )
        asyncio.run(
            _run_script(script, working_dir=args.workdir, gpu=args.gpu, wait=not args.no_wait)
        )

    elif args.command == "infer":
        asyncio.run(_infer(args))

    elif args.command == "status":

        async def _status() -> None:
            async with OrionClient() as client:
                info = await client.get_job(args.job_id)
            print(f"Job {info.job_id}: state={info.state}  node={info.node}")
            print(f"  stdout: {info.stdout}")

        asyncio.run(_status())

    elif args.command == "cancel":

        async def _cancel() -> None:
            async with OrionClient() as client:
                await client.cancel_job(args.job_id)
            print(f"Job {args.job_id} cancelled.")

        asyncio.run(_cancel())

    elif args.command == "logs":
        from emblase.compute.orion import (  # noqa: E402
            _log_path,
            _ssh_host,
            _ssh_user,
            stream_logs,
        )

        host = args.host or _ssh_host()
        user = args.user or _ssh_user()
        log_file = _log_path(args.job_id)
        print(f"Streaming {user}@{host}:{log_file}")
        print("(Ctrl+C to stop)\n")
        try:
            for line in stream_logs(
                args.job_id,
                tail_n=args.tail,
                ssh_host=args.host or None,
                ssh_user=args.user or None,
            ):
                print(line)
        except KeyboardInterrupt:
            pass

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
