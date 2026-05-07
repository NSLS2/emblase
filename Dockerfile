# Emblase — Shifter-compatible container image for NERSC (Perlmutter)
#
# Build (from repo root, M-series Mac):
#   docker build --platform linux/amd64 \
#     --build-arg EMBLASE_VERSION=$(git describe --tags --dirty 2>/dev/null || echo 0.0.0+dev) \
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
    "torch==2.5.1" \
    "torchvision==0.20.1" \
    --index-url https://download.pytorch.org/whl/cu121

# ── emblase + all runtime deps ────────────────────────────────────────────────
# .git is excluded from the build context (see .dockerignore) so hatch-vcs
# cannot derive a version from git tags.  Pass the version explicitly via a
# build arg; default to "0.0.0+dev" for local builds.
ARG EMBLASE_VERSION=0.0.0+dev
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${EMBLASE_VERSION}

COPY . /app/emblase
RUN pip install --no-cache-dir /app/emblase

# hdbscan is not in pyproject.toml but used by the classifier
RUN pip install --no-cache-dir hdbscan

# ── sanity check ──────────────────────────────────────────────────────────────
RUN python -c "import emblase; import torch; print('emblase OK, torch', torch.__version__)"

ENV PYTHONUNBUFFERED=1
