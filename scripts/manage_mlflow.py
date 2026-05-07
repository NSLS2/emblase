#!/usr/bin/env python
"""Manage the Emblase MLflow model registry: push, pull, list, delete.

Subcommands
-----------
push   <path>       Upload a local file or directory and register it.
pull   <name>       Download a registered model.
list               List all registered models.
delete <name>       Delete a registered model and all its versions.

Common flags
------------
--tracking-uri <uri>   Override EMBLASE_MLFLOW_TRACKING_URI.
--api-key <key>        Override EMBLASE_MLFLOW_API_KEY (required for AmSC).

Works with any MLflow-compatible server: American Science Cloud (AmSC),
Azure ML (via azureml-mlflow tracking URI), or a self-hosted MLflow server.
Server coordinates are read from EMBLASE_MLFLOW_TRACKING_URI and
EMBLASE_MLFLOW_API_KEY in your .env file.

Large-file uploads
------------------
AmSC MLflow enforces a ~16 MB per-request limit.  Pass --chunk-size 10 to
split files into 10 MB chunks; they are reassembled transparently on pull.

Examples
--------
# List all registered models
pixi run mlflow list

# Push the ViT encoder weights (single file)
pixi run mlflow push models/vit/vit_model_weights.npz \\
    --name bnl-nsls2-smi-vit

# Push the VAE encoder weights (single file)
pixi run mlflow push models/vae/vae_model_512_weights.npz \\
    --name bnl-nsls2-smi-vae \\
    --chunk-size 10

# Push the parametric UMAP approximator (whole directory — uploads all files)
pixi run mlflow push models/umap_approx \\
    --name bnl-nsls2-smi-umap

# Push the cluster classifier (whole directory)
pixi run mlflow push models/class_5 \\
    --name bnl-nsls2-smi-classifier \\
    --description "5-class unsupervised cluster classifier for SMI SAXS"

# Pull the latest version of a model to a local directory
pixi run mlflow pull bnl-nsls2-smi-umap --output /tmp/umap

# Pull a specific version
pixi run mlflow pull bnl-nsls2-smi-vit --version 2 --output /tmp/vit

# Delete a model (all versions)
pixi run mlflow delete bnl-nsls2-smi-umap
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
    models = mlflow_registry.list_models(tracking_uri=args.tracking_uri, api_key=args.api_key)
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
    import logging

    parser = argparse.ArgumentParser(
        prog="manage_mlflow",
        description="Manage the Emblase MLflow model registry",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    push_p = sub.add_parser("push", help="Upload and register a model file or directory")
    push_p.add_argument("path", help="Local file or directory to upload")
    push_p.add_argument("--name", default=None, help="Registry name (default: derived from path)")
    push_p.add_argument("--description", default=None, help="Human-readable description")
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
    pull_p.add_argument("--version", default=None, help="Version to download (default: latest)")
    pull_p.add_argument("--output", default=".", help="Output directory (default: .)")
    _add_common(pull_p)

    list_p = sub.add_parser("list", help="List all registered models")
    _add_common(list_p)

    del_p = sub.add_parser("delete", help="Delete a registered model")
    del_p.add_argument("model_name", help="Registered model name to delete")
    _add_common(del_p)

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    dispatch = {"push": _push, "pull": _pull, "list": _list, "delete": _delete}
    try:
        dispatch[args.command](args)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
