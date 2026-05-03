# Emblase

**EMBeddings and LAtent Space Explorer** — streams live synchrotron detector images from [Tiled](https://github.com/bluesky/tiled), encodes them into embedding vectors using a ViT/VAE model, and writes results back to Tiled for interactive exploration.

```
detector → Tiled (inputs_copy) → stream_pipeline.py → Orion (Slurm) → Tiled (results)
                                        │                     │
                                  InputsWatcher         streaming_inference
                                  (WebSocket)           (ViT encode + write)
```

## Models

| Name | Architecture |
|------|-------------|
| `vit` | ViT-based autoencoder (`google/vit-base-patch16-224`) |
| `vae` | Convolutional VAE with windowed self-attention |

Weights are loaded from the local filesystem or pulled at job time from an MLflow model registry.

## Installation

Requires [pixi](https://pixi.sh).

```bash
git clone git@github.com:NSLS2/emblase.git
cd emblase
pixi install
cp .env.example .env   # fill in your keys and paths
```

## Batch Inference (quick iteration)

Use `submit_orion.py infer` to process a complete, already-acquired run in one
shot — no watcher, no WebSocket, just encode-all-then-write.  Ideal for trying
different models or hyperparameters on existing data.

```bash
# sync code first
rsync -az src/emblase/ orion-staging.nsls2.bnl.gov:/nsls2/users/ymatviych/code/emblase/src/emblase/

python scripts/submit_orion.py infer \
    --model      bnl-nsls2-smi-vit \
    --run        smi/sandbox/confab26_demo/inputs/run_1086139 \
    --output     smi/sandbox/confab26_demo/results/run_1086139_vit \
    --batch-size 1 \
    --thumb-mode logroi \
    --param      temperature:primary.LinkamThermal_temperature_current:float:°C \
    --param      piezo_x:primary.piezo_x:float:μm \
    --umap-dir   /nsls2/users/ymatviych/code/emblase/models/umap_approx
```

`--run` points at the BlueskyRun container; frames are read from
`run/primary/<image_key>` (default `pil900KW_image`).  `--param` stores scalar
streams from the same primary event stream alongside each embedding.
`--umap-dir` triggers on-node UMAP projection — omit if the approximator is not
yet trained.

The script polls every 5 s and prints the final state.  Check the full log with:

```bash
ssh orion-staging.nsls2.bnl.gov \
    "tail -100 /nsls2/users/ymatviych/orion_jobs/slurm-<jobid>.out"
```

### Batch vs. streaming

| | Batch (`submit_orion.py infer`) | Streaming (`stream_pipeline.py`) |
|---|---|---|
| **When to use** | Complete runs, model iteration | Live acquisition |
| **Params** | ✓ (`--param`) | ✓ (`--param`) |
| **UMAP** | ✓ (`--umap-dir`) | ✓ (auto via `EMBLASE_ORION_UMAP_DIR`) |
| **Latency** | All frames at once after job starts | Incremental, per batch |
| **Setup** | Single command | Watcher must start before data arrives |

---

## Running the Streaming Pipeline

The pipeline watches a Tiled container for new runs and submits an Orion (Slurm) job per run to encode frames and write embeddings incrementally.

> **Order is critical.** Start the watcher *before* copying data. The watcher fires when a new run container appears; if the run is already complete before the job starts, WebSocket events are not replayed and the job will hang.

**Terminal 1 — start the watcher:**

```bash
pixi run python scripts/stream_pipeline.py \
  --inputs  smi/sandbox/confab26_demo/inputs_copy \
  --output  smi/sandbox/confab26_demo/results \
  --model   vit \
  --batch-size 16 \
  --thumb-mode logroi \
  --param temperature:primary.LinkamThermal_temperature_current:float:deg C \
  --no-replay
```

Wait until the watcher prints `Watching … for new runs` before proceeding.

**Terminal 2 — copy a run into the watched container (only after watcher is up):**

```bash
pixi run python scripts/simulate_acquisition.py \
  --src  smi/sandbox/confab26_demo/inputs/run_1086139 \
  --dst  smi/sandbox/confab26_demo/inputs_copy \
  --rename run_live_1086139 \
  --access-tags smi_sandbox \
  --batch-delay 0.5
```

`simulate_acquisition.py` creates the run container first (triggering the watcher), then writes frames one at a time to simulate live acquisition. A Unix timestamp is appended to `--rename` automatically. The Orion job starts ~25–30 s after submission (pixi + model load), well before the copy finishes at `0.5 s/frame × 288 frames ≈ 144 s`.

### `--param` syntax

```
--param name:source:dtype:units
```

- `source` must be `primary.<array_key>` (a scalar stream aligned 1:1 with frames)
- `dtype`: `float`, `int`, `str`, or `bool`
- Repeat `--param` for multiple parameters

### `--no-replay`

Skips reprocessing runs already present in `--inputs` on startup. Omit to process all existing runs.

### Checking Orion job logs

```bash
ssh orion-staging.nsls2.bnl.gov "tail -50 /nsls2/users/ymatviych/orion_jobs/slurm-<jobid>.out"
ssh orion-staging.nsls2.bnl.gov "squeue -u ymatviych"
ssh orion-staging.nsls2.bnl.gov "scancel <jobid>"
```

### Syncing code to Orion

```bash
rsync -az src/emblase/ orion-staging.nsls2.bnl.gov:/nsls2/users/ymatviych/code/emblase/src/emblase/
```

Run this after any local changes before submitting a new job.

## UMAP Projections

The streaming job can compute 2D UMAP projections on-the-fly per batch using a pre-trained MLP approximator:

```bash
# on-the-fly (set umap_dir via EMBLASE_ORION_UMAP_DIR or --umap-dir, handled automatically by OrionBackend)

# post-process projections on an existing results container
pixi run python scripts/compute_umap.py \
  --dataset smi/sandbox/confab26_demo/results/run_live_1086139_<timestamp>

# retrain the approximator on existing embeddings
pixi run python scripts/train_umap.py \
  --dataset smi/sandbox/confab26_demo/results/run_live_1086139_1777749128
```

## Tiled Layout

```
smi/sandbox/confab26_demo/
├── inputs/             # original BlueskyRuns from the detector
├── inputs_copy/        # runs watched by the streaming pipeline
└── results/            # LatentSpaceEmbedding containers written by Orion jobs
```

Each `results/<run>/` container (`spec="LatentSpaceEmbedding"`) holds:
- `embeddings` — `(N, D)` float32 embedding vectors
- `thumbnails` — `(N, H, W)` float32 downsampled images
- `projections` — `(N, 2)` float32 UMAP coordinates (NaN until computed)
- `notes`, `user_labels` — `(N,)` unicode, mutable
- `_index` — PyArrow table with `path`, `slice`, `model_version`, `timestamp`, and any declared scalar params

## Configuration

All settings are read from `.env` (or environment variables), prefixed `EMBLASE_`.
Copy [`.env.example`](.env.example) to `.env` and fill in your values.

| Variable | Description |
|----------|-------------|
| `EMBLASE_TILED_SERVER_URI` | Tiled server URL |
| `EMBLASE_TILED_API_KEY` | Tiled API key (write access) |
| `EMBLASE_TILED_ACCESS_TAGS` | Comma-separated access tags applied to all written nodes |
| `EMBLASE_ORION_API_KEY` | Orion REST API key |
| `EMBLASE_ORION_UMAP_DIR` | Path to `umap_approx/` on Orion (default: `/nsls2/users/ymatviych/code/emblase/models/umap_approx`) |
| `EMBLASE_MLFLOW_TRACKING_URI` | MLflow tracking server URI |
| `EMBLASE_MLFLOW_API_KEY` | MLflow API key |

## Development

```bash
pixi run python -m pytest tests/ -x -q   # 122 tests, no GPU or live connections required
```

## Project Structure

```
src/emblase/
├── config.py               # Settings (pydantic-settings, EMBLASE_ prefix)
├── models.py               # load_model(), encode()
├── mlflow_registry.py      # MLflow push/pull/list
├── compute/
│   ├── base.py             # ComputeBackend ABC
│   ├── orion.py            # OrionBackend, script rendering, submit_streaming
│   └── local.py            # LocalBackend (dev/test)
├── pipeline/
│   ├── streaming.py        # InputsWatcher — subscribes to inputs_copy container
│   └── copy_tiled.py       # deepcopy() — copies a BlueskyRun between catalogs
├── tiled/
│   └── client.py           # read_images, write_output, LatentSpaceEmbedding
└── worker/
    ├── inference.py.tmpl           # batch inference node script (params + UMAP)
    └── streaming_inference.py.tmpl # streaming inference node script
scripts/
├── stream_pipeline.py      # CLI: watch inputs_copy → submit streaming jobs
├── simulate_acquisition.py # CLI: copy a run into inputs_copy frame-by-frame
├── simulate_results.py     # CLI/lib: replay a results container into a local Tiled (WebUI dev)
├── submit_orion.py         # CLI: submit one-off batch jobs (infer/status/cancel)
├── compute_umap.py         # apply UMAP approximator → write projections
└── train_umap.py           # fit UMAP + train MLP approximator
models/
├── vit/                    # ViT weights + loader.py
├── noop/                   # NoopEncoder (seeded random, for testing)
└── umap_approx/            # neural_dimred_wrapper.py, scaler.pkl, umap_approximator.pth
tests/                      # pytest suite (mocked, no hardware required)
```

## Acknowledgements

Emblase builds on ideas from two prior projects at ALS / NSLS-II:

- **[mlex_latent_explorer](https://github.com/mlexchange/mlex_latent_explorer)** — the original MLExchange Latent Space Explorer application, a full Dash/Plotly web app for real-time latent space visualization of synchrotron data. It pioneered the autoencoder + UMAP dimensionality reduction pipeline, MLflow model registry integration, Tiled data I/O, and the streaming architecture via Arroyo and Redis that Emblase is designed to eventually integrate with.

- **[arroyosas](https://github.com/als-computing/arroyosas)** — the streaming small-angle scattering reduction pipeline at ALS, which established the data transport patterns (WebSockets, ZMQ, Tiled) and schema conventions that inform Emblase's streaming architecture.
