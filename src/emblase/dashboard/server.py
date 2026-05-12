"""EMBLASE Dashboard server — entry point.

Run with:
    python -m emblase.dashboard.server
or via pixi:
    pixi run dashboard

The server auto-builds the React UI (if npm is available and dist is stale)
and serves it at http://localhost:<EMBLASE_DASHBOARD_PORT> (default 8765).
"""

from __future__ import annotations

import logging
import subprocess
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import settings
from .routers import compute, instrument, jobs, services, tiled, watcher

logger = logging.getLogger(__name__)

# Paths
_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent.parent  # src/emblase/dashboard -> src/emblase -> src -> repo root
_UI_DIR = _REPO_ROOT / "ui" / "dashboard"
_DIST_DIR = _HERE / "static"


# ── Auto-build React UI ───────────────────────────────────────────────────────


def _ui_is_stale() -> bool:
    """Return True if the built UI is missing or older than any source file."""
    index = _DIST_DIR / "index.html"
    if not index.exists():
        return True
    built_at = index.stat().st_mtime
    src_dir = _UI_DIR / "src"
    if not src_dir.exists():
        return False
    for f in src_dir.rglob("*"):
        if f.is_file() and f.stat().st_mtime > built_at:
            return True
    return False


def _build_ui() -> None:
    """Run `npm run build` in the ui/dashboard directory and copy dist to static/."""
    if not _UI_DIR.exists():
        logger.warning("ui/dashboard directory not found at %s — skipping UI build", _UI_DIR)
        return

    # Check if npm is available
    npm = "npm"
    try:
        subprocess.run([npm, "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("npm not found — cannot auto-build dashboard UI")
        return

    logger.info("Building dashboard UI (npm run build)…")
    result = subprocess.run(
        [npm, "run", "build"],
        cwd=_UI_DIR,
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error("UI build failed (exit code %d)", result.returncode)
        return

    # Copy dist → static
    import shutil

    dist = _UI_DIR / "dist"
    if dist.exists():
        if _DIST_DIR.exists():
            shutil.rmtree(_DIST_DIR)
        shutil.copytree(dist, _DIST_DIR)
        logger.info("UI built and copied to %s", _DIST_DIR)


# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(
    title="EMBLASE Dashboard",
    description=(
        "Monitoring and control panel for the EMBeddings and LAtent Space Explorer — "
        "a multi-service pipeline for AI/ML latent space exploration of synchrotron data."
    ),
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routers ───────────────────────────────────────────────────────────────

app.include_router(tiled.router, prefix="/api")
app.include_router(compute.router, prefix="/api")
app.include_router(services.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(instrument.router, prefix="/api")
app.include_router(watcher.router, prefix="/api")


@app.get("/api/config")
async def get_config():
    """Return non-secret configuration for the dashboard UI."""
    return {
        "tiled_server_uri": settings.tiled_server_uri,
        "tiled_input_container": settings.tiled_input_container,
        "tiled_output_container": settings.tiled_output_container,
        "mlflow_tracking_uri": settings.mlflow_tracking_uri,
        "mlflow_experiment": settings.mlflow_experiment,
        "mlflow_model_prefix": settings.mlflow_model_prefix,
        "chatapp_url": settings.chatapp_url,
        "chatapp_model": settings.chatapp_model,
        "orion_api_url": settings.orion_api_url,
        "orion_cluster": settings.orion_cluster,
        "orion_account": settings.orion_account,
        "nersc_api_uri": settings.nersc_api_uri,
        "nersc_resource_id": settings.nersc_resource_id,
        "nersc_account": settings.nersc_account,
        "nersc_queue": settings.nersc_queue,
        "nersc_time_limit": settings.nersc_time_limit,
        "nersc_container_image": settings.nersc_container_image,
        "compute_backend": settings.compute_backend,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


# ── Static files (React SPA) ──────────────────────────────────────────────────

# Serve shared logo assets at /logos (distinct from Vite's /assets output path)
_LOGOS_DIR = _REPO_ROOT / "ui" / "assets"
if _LOGOS_DIR.exists():
    app.mount("/logos", StaticFiles(directory=str(_LOGOS_DIR)), name="logos")

# Serve the entire Vite dist/ at the root so that:
#   /index.html           → dist/index.html
#   /assets/index-*.js    → dist/assets/index-*.js   (Vite chunk naming)
#   /assets/index-*.css   → dist/assets/index-*.css
# This must come after /api and /logos mounts, and before the SPA fallback.
if _DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_DIST_DIR), html=True), name="spa")


# ── Main entry point ──────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    port = settings.dashboard_port

    # Auto-build UI if stale
    if _ui_is_stale():
        _build_ui()
    else:
        logger.info("Dashboard UI is up-to-date — skipping build")

    logger.info("Starting EMBLASE Dashboard on http://localhost:%d", port)

    if settings.dashboard_open_browser:
        # Open browser slightly after server starts
        import threading

        def _open():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(
        "emblase.dashboard.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
