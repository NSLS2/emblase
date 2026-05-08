"""Manage NERSC jobs: batch inference, streaming inference, status, cancel.

Subcommands
-----------
infer      Submit a batch inference job (encode all frames, write to Tiled).
stream     Submit a streaming inference job (subscribe to a BlueskyRun WebSocket).
status     Poll the state of a job/task by ID.
cancel     Cancel a running job.
logs       Tail the stdout/stderr log of a job via the IRI filesystem API.
resources  List available NERSC compute resources (discover resource IDs).
ls         List a remote directory via the IRI filesystem API.
cat        Print the contents of a remote file via the IRI filesystem API.

Secrets are baked into the job script at submit time (from your local .env) and
the script is uploaded chmod 400.  No separate secrets setup step is needed.

Submit a batch job::

    python scripts/submit_nersc.py infer \\
        --model   bnl-nsls2-smi-vit \\
        --run     smi/sandbox/confab26_demo/inputs/run_1086139 \\
        --output  smi/sandbox/confab26_demo/results/run_1086139_vit \\
        --batch-size 1 \\
        --thumb-mode logroi \\
        --param temperature:primary/LinkamThermal_temperature_current:float:°C

Submit a streaming job::

    python scripts/submit_nersc.py stream \\
        --model  bnl-nsls2-smi-vit \\
        --run    smi/sandbox/confab26_demo/inputs_copy/run_xyz \\
        --output smi/sandbox/confab26_demo/results/run_xyz \\
        --thumb-mode logroi

Check status of a submitted task::

    python scripts/submit_nersc.py status <task_id>

Read the job log (stdout+stderr are merged into job.out)::

    python scripts/submit_nersc.py logs <task_id>

Requires ``EMBLASE_NERSC_API_TOKEN`` in your ``.env`` file.

--param syntax
--------------
    --param name:source:dtype:units

``source`` must be ``primary/<array_key>`` — a scalar stream aligned 1-to-1
with image frames.  ``dtype`` defaults to ``float``; ``units`` defaults to "".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import certifi as _certifi

    os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
except ImportError:
    pass

from emblase.compute import parse_param_specs  # noqa: E402
from emblase.compute.base import JobStatus  # noqa: E402
from emblase.compute.nersc import _NERSC_STATE_MAP, NERSCBackend, NERSCClient  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage NERSC jobs for Emblase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    sub = parser.add_subparsers(dest="command")

    # -- resources subcommand --
    sub.add_parser("resources", help="List available NERSC compute resources")

    # -- ls subcommand --
    ls_p = sub.add_parser("ls", help="List a remote directory via the IRI filesystem API")
    ls_p.add_argument(
        "path",
        help="Absolute remote path to list, e.g. /pscratch/sd/d/dallan",
    )
    ls_p.add_argument(
        "--resource",
        default="scratch",
        metavar="RESOURCE_ID",
        help="Filesystem resource ID (default: scratch).",
    )

    # -- cat subcommand --
    cat_p = sub.add_parser(
        "cat", help="Print the contents of a remote file via the IRI filesystem API"
    )
    cat_p.add_argument(
        "path",
        help="Absolute remote path to read, e.g. /pscratch/sd/d/dallan/somefile",
    )
    cat_p.add_argument(
        "--resource",
        default="scratch",
        metavar="RESOURCE_ID",
        help="Filesystem resource ID (default: scratch).",
    )

    def _add_infer_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--model",
            default="bnl-nsls2-smi-vit",
            metavar="MODEL_NAME",
            help="Model name (short or MLflow registry name).",
        )
        p.add_argument("--mlflow-version", default="", metavar="VERSION")
        p.add_argument("--batch-size", type=int, default=1, metavar="N")
        p.add_argument(
            "--no-wait", action="store_true", help="Return immediately after submission."
        )
        p.add_argument(
            "--output",
            metavar="TILED_PATH",
            default="",
            help="Tiled path for output.  If the path is an existing LatentSpaceEmbedding "
            "it is appended to; if it doesn't exist an LSE is created there; if it "
            "exists but is not an LSE a new LSE named {run_key}_{mode}_{timestamp} "
            "is created inside it.",
        )
        p.add_argument(
            "--thumb-mode",
            default="logroi",
            choices=["default", "logroi"],
        )
        p.add_argument(
            "--image-key",
            default="primary/pil900KW_image",
            metavar="STREAM.KEY",
            help="Slash-separated image array key, e.g. 'primary/pil900KW_image'.",
        )
        p.add_argument(
            "--param",
            action="append",
            default=[],
            dest="params",
            metavar="name:source[:dtype[:units]]",
            help="Parameter spec (repeatable).",
        )
        p.add_argument(
            "--projector",
            default=None,
            metavar="NAME|false",
            help="Projector approximator name, or 'false' to skip.",
        )
        p.add_argument("--classifier", default=None, metavar="NAME", help="Classifier model name.")

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

    # -- logs subcommand --
    logs_p = sub.add_parser("logs", help="Tail job stdout/stderr via the IRI filesystem API")
    logs_p.add_argument("task_id", help="Task ID returned at submission time")
    logs_p.add_argument(
        "--lines",
        type=int,
        default=100,
        metavar="N",
        help="Number of tail lines to fetch (default: 100).",
    )

    return parser


async def _infer(args: argparse.Namespace) -> None:
    backend = NERSCBackend()
    param_specs = parse_param_specs(args.params)

    print(f"Backend         : NERSC")
    print(f"Model           : {args.model}  ver={args.mlflow_version or 'latest'}  batch_size={args.batch_size}  thumb_mode={args.thumb_mode}")
    print(f"Run             : {args.run or '(none)'}")
    print(f"Output          : {args.output or '(none)'}")
    print(f"Image key       : {args.image_key}")
    if param_specs:
        print(f"Params          : {list(param_specs)}")
    if args.projector:
        print(f"Projector       : {args.projector}")
    if args.classifier:
        print(f"Classifier      : {args.classifier}")
    print(f"Working dir     : {backend.working_dir}")
    print(f"Models dir      : {backend.models_dir}")
    print(f"Container image : {backend.container_image}")
    print(f"Resource        : {backend.client.resource_id}")
    print(f"Account         : {backend.account}")
    print(f"Queue           : {backend.queue or '(scheduler default)'}")
    print(f"Constraint      : {backend.constraint or '(none)'}")
    print(f"Time limit      : {backend.time_limit}")

    print(f"\nSubmitting {args.model!r} batch inference job to NERSC...")
    task_id = await backend.submit(
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
    print(f"Task submitted  : {task_id}")
    print(f"Log path        : {backend._jobs[task_id]['log_path']}")

    if args.no_wait:
        print("(not waiting — use 'status <task_id>' to check)")
        return

    await asyncio.to_thread(backend.monitor_job, task_id)


async def _stream(args: argparse.Namespace) -> None:
    backend = NERSCBackend()
    param_specs = parse_param_specs(args.params)

    print(f"Backend         : NERSC")
    print(f"Model           : {args.model}  ver={args.mlflow_version or 'latest'}  batch_size={args.batch_size}  thumb_mode={args.thumb_mode}")
    print(f"Run             : {args.run}")
    print(f"Output          : {args.output}")
    print(f"Image key       : {args.image_key}")
    if param_specs:
        print(f"Params          : {list(param_specs)}")
    if args.projector:
        print(f"Projector       : {args.projector}")
    if args.classifier:
        print(f"Classifier      : {args.classifier}")
    print(f"Working dir     : {backend.working_dir}")
    print(f"Models dir      : {backend.models_dir}")
    print(f"Container image : {backend.container_image}")
    print(f"Resource        : {backend.client.resource_id}")
    print(f"Account         : {backend.account}")
    print(f"Queue           : {backend.queue or '(scheduler default)'}")
    print(f"Constraint      : {backend.constraint or '(none)'}")
    print(f"Time limit      : {backend.time_limit}")

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
    print(f"Task submitted  : {task_id}")
    print(f"Log path        : {backend._jobs[task_id]['log_path']}")

    if args.no_wait:
        print("(not waiting — use 'status <task_id>' to check)")
        return

    await asyncio.to_thread(backend.monitor_job, task_id, poll_interval=30.0)


async def _status(task_id: str) -> None:
    async with NERSCClient() as client:
        info = await client.get_job(task_id)
    status = _NERSC_STATE_MAP.get(info.state, JobStatus.pending)
    print(f"Task {task_id}: state={info.state}  mapped={status.value}")
    if info.raw:
        print(json.dumps(info.raw, indent=2, default=str))


async def _cancel(task_id: str) -> None:
    async with NERSCClient() as client:
        await client.cancel_job(task_id)
    print(f"Task {task_id} cancelled")


async def _ls(path: str, resource: str) -> None:
    async with NERSCClient() as client:
        entries = await client.ls(path, filesystem_resource_id=resource)
    if not entries:
        print("(empty)")
        return
    for entry in entries:
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("filename") or str(entry)
            perms = entry.get("permissions") or entry.get("mode") or ""
            size = entry.get("size") or ""
            print(f"{perms:12}  {size:10}  {name}")
        else:
            print(entry)


async def _cat(path: str, resource: str) -> None:
    async with NERSCClient() as client:
        data = await client.download_file(path, filesystem_resource_id=resource)
    print(data.decode(errors="replace"), end="")


async def _resources() -> None:
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


async def _logs(task_id: str, lines: int = 100) -> None:
    import json as _json

    async with NERSCClient() as client:
        info = await client.get_job(task_id)
        meta = (info.raw or {}).get("status", {}).get("meta_data", {})

        # --- strategy 1: admincomment.stdoutPath (populated once job starts) ---
        path = ""
        try:
            admin = _json.loads(meta.get("admincomment", "{}") or "{}")
            path = admin.get("stdoutPath", "")
        except (ValueError, TypeError):
            pass

        # --- strategy 2: ls {workdir}/scripts/ and match by job-name slug ---
        if not path:
            workdir = meta.get("workdir", "")
            jobname = meta.get("jobname", "")  # e.g. "emblase-stream-bnl-nsls2-smi-vit"
            slug = jobname.removeprefix("emblase-")  # e.g. "stream-bnl-nsls2-smi-vit"
            if workdir and slug:
                scripts_dir = workdir.rstrip("/") + "/scripts"
                try:
                    entries = await client.ls(scripts_dir, filesystem_resource_id="scratch")
                    # entries have full paths in "name"; find dirs ending with _{slug}
                    matches = [
                        e["name"]
                        for e in entries
                        if e.get("name", "").rstrip("/").endswith(f"_{slug}")
                        and e.get("type") == "d"
                    ]
                    if matches:
                        # pick the newest (highest timestamp prefix = lexicographically last)
                        matches.sort(reverse=True)
                        path = f"{matches[0]}/job.out"
                except Exception as exc:
                    logging.debug("ls fallback failed: %s", exc)

        if not path:
            print(
                "Could not determine log path from job metadata.\n"
                "Use 'cat <path>' to read a specific file, e.g.:\n"
                "  python scripts/submit_nersc.py cat "
                "/pscratch/sd/d/<user>/emblase/jobs/scripts/<ts>_<model>/job.out",
                file=sys.stderr,
            )
            return
        print(f"Fetching last {lines} lines of: {path}")
        content = await client.read_file_tail(path, lines=lines)
        print(content)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.log_level != "DEBUG":
        for _noisy in ("httpx", "httpcore", "hpack"):
            logging.getLogger(_noisy).setLevel(logging.WARNING)

    if args.command == "resources":
        asyncio.run(_resources())
    elif args.command == "ls":
        asyncio.run(_ls(args.path, args.resource))
    elif args.command == "cat":
        asyncio.run(_cat(args.path, args.resource))
    elif args.command == "infer":
        asyncio.run(_infer(args))
    elif args.command == "stream":
        asyncio.run(_stream(args))
    elif args.command == "status":
        asyncio.run(_status(args.task_id))
    elif args.command == "cancel":
        asyncio.run(_cancel(args.task_id))
    elif args.command == "logs":
        asyncio.run(_logs(args.task_id, lines=args.lines))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
