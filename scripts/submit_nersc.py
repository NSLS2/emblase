"""Manage NERSC jobs: batch inference, streaming inference, status, cancel.

Subcommands
-----------
infer   Submit a batch inference job (encode all frames, write to Tiled).
stream  Submit a streaming inference job (subscribe to a BlueskyRun WebSocket).
status  Poll the state of a job/task by ID.
cancel  Cancel a running job.
resources  List available NERSC compute resources (discover resource IDs).

Typical workflow
----------------
Submit a batch job::

    python scripts/submit_nersc.py infer \\
        --model   bnl-nsls2-smi-vit \\
        --run     smi/sandbox/confab26_demo/inputs/run_1086139 \\
        --output  smi/sandbox/confab26_demo/results/run_1086139_vit \\
        --batch-size 1 \\
        --thumb-mode logroi \\
        --param temperature:primary.LinkamThermal_temperature_current:float:°C

Submit a streaming job::

    python scripts/submit_nersc.py stream \\
        --model  bnl-nsls2-smi-vit \\
        --run    smi/sandbox/confab26_demo/inputs_copy/run_xyz \\
        --output smi/sandbox/confab26_demo/results/run_xyz \\
        --thumb-mode logroi

Requires ``EMBLASE_NERSC_API_TOKEN`` in your ``.env`` file.

Check the job log (once you have the Slurm job ID from the task result)::

    ssh perlmutter.nersc.gov \\
        "tail -50 {EMBLASE_NERSC_WORKING_DIR}/slurm-<jobid>.out"

--param syntax
--------------
    --param name:source:dtype:units

``source`` must be ``primary.<array_key>`` — a scalar stream aligned 1-to-1
with image frames.  ``dtype`` defaults to ``float``; ``units`` defaults to "".
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emblase.compute.base import JobStatus  # noqa: E402
from emblase.compute.nersc import NERSCBackend, NERSCClient, _NERSC_STATE_MAP  # noqa: E402


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
    backend = NERSCBackend()
    print(f"Working dir     : {backend.working_dir}")
    print(f"Models dir      : {backend.models_dir}")
    print(f"Container image : {backend.container_image}")
    print(f"Resource        : {backend.client.resource_id}")

    param_specs = _parse_param_specs(args.params)

    submit_kwargs = dict(
        model_name=args.model,
        batch_size=args.batch_size,
        output=args.output or "",
        thumb_mode=args.thumb_mode,
        mlflow_version=args.mlflow_version,
        param_specs=param_specs,
        projector=args.projector,
        classifier=args.classifier,
        run_path=args.run or "",
        image_key=args.image_key,
    )

    print(f"\nSubmitting {args.model!r} batch inference job to NERSC...")
    task_id = await backend.submit(**submit_kwargs)
    print(f"Task submitted: {task_id}")

    if args.no_wait:
        print("(not waiting — use 'status <task_id>' to check)")
        return

    print("Waiting for job to complete (polling every 15s)...")
    t0 = time.monotonic()
    while True:
        await asyncio.sleep(15)
        elapsed = int(time.monotonic() - t0)
        async with NERSCClient() as client:
            info = await client.get_job(task_id)
        status = _NERSC_STATE_MAP.get(info.state, JobStatus.pending)
        print(f"  [{elapsed:>4}s] state={info.state} ({status.value})")
        if status in (JobStatus.completed, JobStatus.failed):
            break

    print(f"\nTask {task_id} finished: {status.value}")
    if status != JobStatus.completed:
        print("Job did not complete successfully — check the Slurm log on NERSC.")
    elif not args.output:
        print(f"Output: {backend.working_dir}/job_<slurm_id>/output.npy (on NERSC filesystem)")


async def _stream(args):
    backend = NERSCBackend()
    print(f"Working dir     : {backend.working_dir}")
    print(f"Models dir      : {backend.models_dir}")
    print(f"Container image : {backend.container_image}")
    print(f"Resource        : {backend.client.resource_id}")

    param_specs = _parse_param_specs(args.params)

    print(f"\nSubmitting {args.model!r} streaming inference job to NERSC...")
    task_id = await backend.submit_streaming(
        run_path=args.run,
        output=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        mlflow_version=args.mlflow_version,
        thumb_mode=args.thumb_mode,
        image_key=args.image_key,
        param_specs=param_specs,
        projector=args.projector,
        classifier=args.classifier,
    )
    print(f"Task submitted: {task_id}")

    if args.no_wait:
        print("(not waiting — use 'status <task_id>' to check)")
        return

    print("Waiting for job to complete (polling every 30s — streaming jobs run for up to 2h)...")
    t0 = time.monotonic()
    while True:
        await asyncio.sleep(30)
        elapsed = int(time.monotonic() - t0)
        async with NERSCClient() as client:
            info = await client.get_job(task_id)
        status = _NERSC_STATE_MAP.get(info.state, JobStatus.pending)
        print(f"  [{elapsed:>4}s] state={info.state} ({status.value})")
        if status in (JobStatus.completed, JobStatus.failed):
            break

    print(f"\nTask {task_id} finished: {status.value}")


async def _status(task_id: str):
    async with NERSCClient() as client:
        info = await client.get_job(task_id)
    status = _NERSC_STATE_MAP.get(info.state, JobStatus.pending)
    print(f"Task {task_id}: state={info.state}  mapped={status.value}")
    if info.raw:
        print(json.dumps(info.raw, indent=2, default=str))


async def _cancel(task_id: str):
    async with NERSCClient() as client:
        await client.cancel_job(task_id)
    print(f"Task {task_id} cancelled")


async def _resources():
    async with NERSCClient() as client:
        items = await client.discover_resources()
    print("Available NERSC resources:")
    for item in items:
        if isinstance(item, dict):
            name = item.get("name") or item.get("hostname") or item.get("system") or str(item)
            status = item.get("status", "")
            print(f"  {name}  {status}")
        else:
            print(f"  {item}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage NERSC jobs for Emblase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # -- resources subcommand --
    sub.add_parser("resources", help="List available NERSC compute resources")

    # shared inference arguments builder
    def _add_infer_args(p):
        p.add_argument(
            "--model",
            default="bnl-nsls2-smi-vit",
            metavar="MODEL_NAME",
            help="Model name (short or MLflow registry name).",
        )
        p.add_argument("--mlflow-version", default="", metavar="VERSION")
        p.add_argument("--batch-size", type=int, default=1, metavar="N")
        p.add_argument("--no-wait", action="store_true")
        p.add_argument("--output", metavar="TILED_PATH", default="")
        p.add_argument(
            "--thumb-mode",
            default="logroi",
            choices=["default", "logroi"],
        )
        p.add_argument("--image-key", default="pil900KW_image", metavar="KEY")
        p.add_argument(
            "--param",
            action="append",
            default=[],
            dest="params",
            metavar="name:source[:dtype[:units]]",
        )
        p.add_argument("--projector", default=None, metavar="NAME|false")
        p.add_argument("--classifier", default=None, metavar="NAME")

    # -- infer subcommand --
    infer_p = sub.add_parser("infer", help="Submit a batch inference job")
    _add_infer_args(infer_p)
    infer_p.add_argument(
        "--run",
        metavar="TILED_PATH",
        default="",
        help="Tiled path to a BlueskyRun container.",
    )

    # -- stream subcommand --
    stream_p = sub.add_parser("stream", help="Submit a streaming inference job")
    _add_infer_args(stream_p)
    stream_p.add_argument(
        "--run",
        required=True,
        metavar="TILED_PATH",
        help="Tiled path to the BlueskyRun to watch (inputs_copy container).",
    )

    # -- status subcommand --
    status_p = sub.add_parser("status", help="Check job/task status")
    status_p.add_argument("task_id")

    # -- cancel subcommand --
    cancel_p = sub.add_parser("cancel", help="Cancel a running job")
    cancel_p.add_argument("task_id")

    args = parser.parse_args()

    if args.command == "resources":
        asyncio.run(_resources())
    elif args.command == "infer":
        asyncio.run(_infer(args))
    elif args.command == "stream":
        asyncio.run(_stream(args))
    elif args.command == "status":
        asyncio.run(_status(args.task_id))
    elif args.command == "cancel":
        asyncio.run(_cancel(args.task_id))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
