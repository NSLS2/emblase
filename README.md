# Emblase

**EMBeddings and LAtent Space Explorer** — a lightweight service for dimensionality reduction of synchrotron images using pre-trained deep learning models.

## Overview

Emblase receives images, passes them through an autoencoder encoder to produce latent vectors, and returns the results. It supports pluggable compute backends — run inference locally for development, or dispatch jobs to an HPC cluster ([Orion](https://github.com/NSLS2/orion-api) at NSLS-II) via Slurm, with a single config change. Images can be uploaded directly, read from the Orion filesystem, or fetched from a [Tiled](https://github.com/bluesky/tiled) server on the compute node.

```
image source ──► ComputeBackend.submit() ──► encode ──► latent vectors
                          │
               ┌──────────┴──────────┐
               │                     │
         LocalBackend           OrionBackend
         (in-process,           (Slurm via
          threaded)              Orion API)
```

## Models

| Name | Architecture | Notes |
|------|-------------|-------|
| `vae` | Convolutional VAE with windowed self-attention | Requires `image_size` |
| `vit` | ViT-based autoencoder (`google/vit-base-patch16-224`) | `image_size` ignored |

Both models have a configurable `latent_dim`. Pre-trained weights are loaded from `.npz` files on the local or remote filesystem.

## Installation

Requires [pixi](https://pixi.sh).

```bash
git clone git@github.com:NSLS2/emblase.git
cd emblase
pixi install
```

Copy `.env.example` to `.env` and fill in the required values (see [Configuration](#configuration)).

## Usage

### Run the API server

```bash
pixi run serve
# → http://localhost:8000/docs
```

### Local inference (dev/testing)

```bash
pixi run test-local
```

### Orion job management

```bash
# connectivity test
pixi run orion test

# submit a custom bash script
pixi run orion run path/to/script.sh --gpu

# inference — upload images from a local .npy file
pixi run orion infer --model vae --images-npy /local/data.npy

# inference — images already on Orion filesystem
pixi run orion infer --model vit --orion-path /nsls2/data/images.npy --image-size 512

# inference — fetch images from Tiled on the compute node
pixi run orion infer --model vae --tiled-uris path/to/scan1 path/to/scan2

# inference — dummy data (for smoke testing)
pixi run orion infer --model vae --n-images 4 --image-size 512

# check / cancel a job
pixi run orion status 671
pixi run orion cancel 671
```

All `infer` options support `--model`, `--image-size`, `--latent-dim`, and `--no-wait`.

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health and active backend |
| `POST` | `/evaluate` | Submit images for encoding |
| `GET` | `/jobs/{id}` | Check job status / retrieve results |
| `DELETE` | `/jobs/{id}` | Cancel a job |

**Example:**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"model": "vae", "dummy_images": 2, "image_size": [512, 512]}'
```

## Configuration

All settings are read from `.env` (prefix `EMBLASE_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBLASE_COMPUTE_BACKEND` | `local` | `local` or `orion` |
| `EMBLASE_MODELS_DIR` | `./models` | Path to model weights (local) |
| `EMBLASE_TILED_URI` | — | Tiled server base URI (used by Orion node) |
| `EMBLASE_TILED_API_KEY` | — | API key for authenticated Tiled servers |
| `EMBLASE_ORION_API_URL` | `https://orion-api-staging.nsls2.bnl.gov` | Orion API base URL |
| `EMBLASE_ORION_API_KEY` | — | Orion API key |
| `EMBLASE_ORION_CLUSTER` | `orion` | Slurm cluster name |
| `EMBLASE_ORION_PROJECT_DIR` | `/nsls2/users/…/emblase` | Root of the emblase repo on Orion |
| `EMBLASE_ORION_WORKING_DIR` | `…/emblase/jobs` | Parent directory for job subdirectories |
| `EMBLASE_ORION_MODELS_DIR` | `…/emblase/models` | Model weights directory on Orion |
| `EMBLASE_ORION_HOME` | `/nsls2/users/…` | `HOME` injected into the Slurm job environment |
| `EMBLASE_ORION_ACCOUNT` | `staff` | Slurm account |
| `EMBLASE_ORION_PATH` | `/usr/bin:…:/nsls2/software/bin` | `PATH` injected into the Slurm job (must include pixi) |

### Orion job layout

Each submitted job creates a subdirectory named after its Slurm job ID:

```
$EMBLASE_ORION_WORKING_DIR/
├── slurm-671.out       # stdout + stderr
└── job_671/
    ├── input.npy       # input images
    └── output.npy      # latent vectors
```

## Development

```bash
pixi run test    # 32 tests, no GPU or live connections required
pixi run serve   # start API with auto-reload
```

Tests use `httpx.MockTransport` for HTTP and `monkeypatch` for model loading — no weights needed.

## Project structure

```
src/emblase/
├── app.py                  # FastAPI application
├── config.py               # Settings (pydantic-settings, EMBLASE_ prefix)
├── schemas.py              # Request / response models
├── models.py               # load_model(), encode() — VAE and ViT
├── compute/
│   ├── base.py             # ComputeBackend ABC, JobStatus, JobResult
│   ├── local.py            # LocalBackend (threaded in-process inference)
│   └── orion.py            # OrionClient (HTTP) + OrionBackend (Slurm)
└── worker/
    └── inference.py.tmpl   # Inference script rendered and embedded at submit time
scripts/
├── submit_local.py         # Local inference smoke test
└── submit_orion.py         # CLI: infer, test, status, cancel
tests/                      # pytest suite (mocked, no hardware required)
```

## Acknowledgements

Emblase builds on ideas from two prior projects at ALS / NSLS-II:

- **[mlex_latent_explorer](https://github.com/mlexchange/mlex_latent_explorer)** — the original MLExchange Latent Space Explorer application, a full Dash/Plotly web app for real-time latent space visualization of synchrotron data. It pioneered the autoencoder + UMAP dimensionality reduction pipeline, MLflow model registry integration, Tiled data I/O, and the streaming architecture via Arroyo and Redis that Emblase is designed to eventually integrate with.

- **[arroyosas](https://github.com/als-computing/arroyosas)** — the streaming small-angle scattering reduction pipeline at ALS, which established the data transport patterns (WebSockets, ZMQ, Tiled) and schema conventions (e.g. `RawFrameEvent`) that inform Emblase's planned Tiled listener integration.
