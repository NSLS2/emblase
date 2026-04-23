# Emblase

**EMBeddings and LAtent Space Explorer** — a lightweight service for dimensionality reduction of synchrotron images using pre-trained deep learning models.

## Overview

Emblase receives images, passes them through an autoencoder to produce latent vectors, and returns the results. It is designed to integrate with [Tiled](https://github.com/bluesky/tiled) for data I/O and supports pluggable compute backends — run inference locally for development or dispatch jobs to an HPC cluster (currently [Orion](https://github.com/NSLS2/orion-api) at NSLS-II) with a single config change.

```
image → ComputeBackend.submit() → encode → latent vectors
                 │
         ┌───────┴────────┐
         │                │
   LocalBackend      OrionBackend
   (in-process)      (Slurm via
                      Orion API)
```

## Models

Two autoencoder architectures are supported:

| Name | Architecture | Input |
|------|-------------|-------|
| `vae` | Convolutional VAE with windowed self-attention | 512 × 512 grayscale |
| `vit` | ViT-based autoencoder (google/vit-base-patch16-224) | any grayscale |

Pre-trained weights are loaded from `.npz` files under `models/`.

## Installation

Requires [pixi](https://pixi.sh).

```bash
git clone git@github.com:NSLS2/emblase.git
cd emblase
pixi install
```

Copy `.env.example` to `.env` and fill in the values (see [Configuration](#configuration)).

## Usage

### Run the API server

```bash
pixi run serve
# → http://localhost:8000/docs
```

### Evaluate images locally

```bash
pixi run test-local
```

### Submit a job to Orion

```bash
# connectivity test
pixi run orion test --gpu

# submit any bash script
pixi run orion run path/to/script.sh --gpu

# check / cancel a job
pixi run orion status 638
pixi run orion cancel 638
```

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health and active backend |
| `POST` | `/evaluate` | Submit images for encoding |
| `GET` | `/jobs/{id}` | Check job status and retrieve results |
| `DELETE` | `/jobs/{id}` | Cancel a job |

**Example request:**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"model": "vae", "dummy_images": 2, "image_size": [512, 512]}'
```

## Configuration

Settings are read from `.env` (prefix `EMBLASE_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBLASE_COMPUTE_BACKEND` | `local` | `local` or `orion` |
| `EMBLASE_MODELS_DIR` | `./models` | Path to model weights |
| `EMBLASE_ORION_API_URL` | `https://orion-api-staging.nsls2.bnl.gov` | Orion API base URL |
| `EMBLASE_ORION_API_KEY` | — | Orion API key |
| `EMBLASE_ORION_CLUSTER` | `orion` | Cluster name |
| `EMBLASE_ORION_WORKING_DIR` | `~/code/emblase/jobs` | Remote job working directory |
| `EMBLASE_ORION_ACCOUNT` | `staff` | Slurm account |

## Development

```bash
pixi run test       # run all tests (30 tests, no GPU required)
pixi run serve      # start API with auto-reload
```

Tests use mocked HTTP (via `httpx.MockTransport`) and mocked model loading — no weights or live connections needed.

## Project structure

```
src/emblase/
├── app.py              # FastAPI application
├── config.py           # Settings (pydantic-settings)
├── schemas.py          # Request/response models
├── models.py           # Model loading (VAE, ViT)
├── compute/
│   ├── base.py         # ComputeBackend ABC, JobStatus, JobResult
│   ├── local.py        # LocalBackend (in-process inference)
│   └── orion.py        # OrionClient + OrionBackend (Slurm/HPC)
└── worker/
    └── inference.py.tmpl  # Inference script template (rendered at submit time)
scripts/
├── submit_local.py     # Run local inference on dummy images
└── submit_orion.py     # CLI for Orion job management
tests/                  # pytest suite (mocked, no hardware required)
```

## Acknowledgements

Emblase builds on ideas and code from two prior projects at the ALS / NSLS-II:

- **[mlex_latent_explorer](https://github.com/mlexchange/mlex_latent_explorer)** — the original MLExchange Latent Space Explorer application, a full Dash/Plotly web app for real-time latent space visualization of synchrotron data. It pioneered the autoencoder + UMAP dimensionality reduction pipeline, MLflow model registry integration, Tiled data I/O, and the streaming architecture via Arroyo and Redis that Emblase is designed to eventually integrate with.

- **[arroyosas](https://github.com/als-computing/arroyosas)** — the streaming small-angle scattering reduction pipeline at ALS, which established the data transport patterns (WebSockets, ZMQ, Tiled) and schema conventions (e.g. `RawFrameEvent`) that inform Emblase's planned Tiled listener integration.
