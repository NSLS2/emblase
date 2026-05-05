# Emblase — Shifter-compatible container image for NERSC (Perlmutter)
#
# Build:
#   docker build -t ghcr.io/nsls2/emblase:latest .
#
# The image installs emblase directly from the public GitHub repo at the
# Git ref captured by the CI workflow (tag or branch SHA).  All heavy
# inference dependencies (torch, umap-learn, etc.) are pre-installed so
# the Shifter container starts fast on NERSC nodes.
#
# Shifter compatibility notes:
#   - Uses CUDA 12.1 + cuDNN 8 base (matches Perlmutter's CUDA 12.x stack)
#   - No USER directives (Shifter runs as the submitting user)
#   - No VOLUME / EXPOSE (not needed for batch/streaming HPC jobs)
#   - WORKDIR /app (writable scratch is mounted at runtime via $PSCRATCH)

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/nsls2/emblase"
LABEL org.opencontainers.image.description="Emblase inference image for NERSC Shifter"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"

# ── system packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python / pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Install torch separately first so pip can choose the correct CUDA wheel.
RUN pip install --no-cache-dir \
    torch==2.3.* \
    torchvision==0.18.* \
    --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the emblase runtime dependencies.
# Versions are kept in sync with pyproject.toml; update as needed.
RUN pip install --no-cache-dir \
    numpy>=1.24 \
    scipy \
    scikit-learn>=1.3 \
    umap-learn>=0.5 \
    joblib>=1.3 \
    transformers>=4.30 \
    httpx>=0.28.1,<0.29 \
    pydantic>=2.0 \
    pydantic-settings>=2.0 \
    python-dotenv>=1.0 \
    "tiled[client]>=0.1" \
    mlflow>=2.0 \
    azureml-mlflow>=1.0 \
    hdbscan \
    certifi

# ── emblase itself ────────────────────────────────────────────────────────────
# Installed from the public GitHub repo at the exact Git ref pinned by CI.
# ARG is set to "main" for local builds; CI overrides with the tag SHA.
ARG EMBLASE_REF=main
RUN pip install --no-cache-dir \
    "git+https://github.com/nsls2/emblase.git@${EMBLASE_REF}"

# ── sanity check ──────────────────────────────────────────────────────────────
RUN python -c "import emblase; import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

ENV PYTHONUNBUFFERED=1
