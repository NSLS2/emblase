#!/usr/bin/env python
"""CLI for the Emblase MLflow model registry.

Subcommands
-----------
push   <path>       Upload a local file or directory and register it.
pull   <name>       Download a registered model.
list               List all registered models.
delete <name>      Delete a registered model.

Common flags
------------
--tracking-uri <uri>   Override EMBLASE_MLFLOW_TRACKING_URI.

Works with any MLflow-compatible server: ASC, Azure ML (via azureml-mlflow
tracking URI), Databricks, or a self-hosted MLflow server.

Examples
--------
# Push VAE weights
pixi run mlflow push models/vae/vae_model_512_weights.npz \\
    --name vae-512 \\
    --tracking-uri https://mlflow.example.com

# Pull latest version
pixi run mlflow pull vae-512 --output /tmp/vae

# Pull specific version
pixi run mlflow pull vae-512 --version 3 --output /tmp/vae

# List all models
pixi run mlflow list

# Delete a model
pixi run mlflow delete vae-512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emblase import mlflow_registry  # noqa: E402

# ── helpers ───────────────────────────────────────────────────────────────────


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tracking-uri",
        default=None,
        metavar="URI",
        help="MLflow tracking URI (default: EMBLASE_MLFLOW_TRACKING_URI env var)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="X-Api-Key for servers that require it, e.g. AmSC (default: EMBLASE_MLFLOW_API_KEY env var)",
    )


# ── subcommand handlers ───────────────────────────────────────────────────────


def _push(args: argparse.Namespace) -> None:
    chunk_size = args.chunk_size * 1024 * 1024 if args.chunk_size else 0
    mlflow_registry.push(
        path=args.path,
        name=args.name,
        description=args.description,
        tracking_uri=args.tracking_uri,
        api_key=args.api_key,
        chunk_size=chunk_size,
    )


def _pull(args: argparse.Namespace) -> None:
    mlflow_registry.pull(
        model_name=args.model_name,
        version=args.version,
        output_dir=args.output,
        tracking_uri=args.tracking_uri,
        api_key=args.api_key,
    )


def _list(args: argparse.Namespace) -> None:
    models = mlflow_registry.list_models(
        tracking_uri=args.tracking_uri, api_key=args.api_key
    )
    if not models:
        print("No registered models found.")
        return
    col = max(len(m.name) for m in models)
    print(f"{'Name':<{col}}  {'Latest':>8}  Description")
    print("-" * (col + 32))
    for m in models:
        ver = str(m.latest_version) if m.latest_version is not None else "?"
        desc = m.description or ""
        print(f"{m.name:<{col}}  {ver:>8}  {desc}")


def _delete(args: argparse.Namespace) -> None:
    mlflow_registry.delete(
        model_name=args.model_name, tracking_uri=args.tracking_uri, api_key=args.api_key
    )


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mlflow_registry",
        description="Emblase MLflow model registry CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    push_p = sub.add_parser(
        "push", help="Upload and register a model file or directory"
    )
    push_p.add_argument("path", help="Local file or directory to upload")
    push_p.add_argument(
        "--name", default=None, help="Registry name (default: derived from path)"
    )
    push_p.add_argument(
        "--description", default=None, help="Human-readable description"
    )
    push_p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="MB",
        help="Split files into chunks of this size before upload (in MB). "
        "Disabled by default. Use 10 for AmSC MLflow (~16 MB per-request limit).",
    )
    _add_common(push_p)

    pull_p = sub.add_parser("pull", help="Download a registered model")
    pull_p.add_argument("model_name", help="Registered model name")
    pull_p.add_argument(
        "--version", default=None, help="Version to download (default: latest)"
    )
    pull_p.add_argument("--output", default=".", help="Output directory (default: .)")
    _add_common(pull_p)

    list_p = sub.add_parser("list", help="List all registered models")
    _add_common(list_p)

    del_p = sub.add_parser("delete", help="Delete a registered model")
    del_p.add_argument("model_name", help="Registered model name to delete")
    _add_common(del_p)

    args = parser.parse_args()
    dispatch = {"push": _push, "pull": _pull, "list": _list, "delete": _delete}
    try:
        dispatch[args.command](args)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
