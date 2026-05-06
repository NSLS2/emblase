# Emblase — Shifter-compatible container image for NERSC (Perlmutter)
#
# Build (from repo root, M-series Mac):
#   docker build --platform linux/amd64 \
#     --build-arg EMBLASE_REF=$(git rev-parse HEAD) \
#     -t ghcr.io/genematx/emblase:latest .
#
# Shifter compatibility notes:
#   - CUDA 12.1 + cuDNN 8 runtime base (matches Perlmutter's CUDA 12.x stack)
#   - linux/amd64 only — Perlmutter nodes are x86_64
#   - No USER directives (Shifter runs as the submitting user)
#   - No VOLUME / EXPOSE (not needed for HPC batch/streaming jobs)

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/nsls2/emblase"
LABEL org.opencontainers.image.description="Emblase inference image for NERSC Shifter"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"

# ── system packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python / pip
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && python -m pip install --upgrade --no-cache-dir pip setuptools wheel

WORKDIR /app

# ── PyTorch (CUDA 12.1 wheel) ─────────────────────────────────────────────────
# Install before emblase so pip uses the cu121 index rather than PyPI's CPU wheel.
RUN pip install --no-cache-dir \
    "torch==2.3.*" \
    "torchvision==0.18.*" \
    --index-url https://download.pytorch.org/whl/cu121

# ── emblase + all runtime deps ────────────────────────────────────────────────
# Copy source into the image and install in editable mode.
# For CI builds the source is replaced by a git+https install once the repo
# is public and the workflow file lands on the default branch.
COPY . /app/emblase
RUN pip install --no-cache-dir /app/emblase

# hdbscan is not in pyproject.toml but used by the classifier
RUN pip install --no-cache-dir hdbscan

# ── sanity check ──────────────────────────────────────────────────────────────
RUN python -c "import emblase; import torch; print('emblase OK, torch', torch.__version__)"

ENV PYTHONUNBUFFERED=1
