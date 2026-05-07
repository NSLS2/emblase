# Emblase

**EMBeddings and LAtent Space Explorer** — encodes synchrotron detector images
into embedding vectors using ViT/VAE models, writes results to
[Tiled](https://github.com/bluesky/tiled) for interactive exploration, and
supports both live streaming acquisition and batch post-processing.

## Installation

Requires [pixi](https://pixi.sh).

```bash
git clone git@github.com:NSLS2/emblase.git
cd emblase
pixi install
cp .env.example .env   # fill in your keys and paths
```

## Syncing code to Orion

Both local and Orion checkouts track the same git remote. Push commits locally,
then pull on the compute node before submitting a job:

```bash
BRANCH=$(git branch --show-current)
git push origin $BRANCH
ssh orion-staging.nsls2.bnl.gov "cd $EMBLASE_ORION_PROJECT_DIR && git fetch && git checkout $BRANCH && git pull"
```

---

## Batch Inference

Use `submit_orion.py infer` to process a complete, already-acquired run in one
shot. No watcher, no WebSocket — ideal for trying different models or
hyperparameters on existing data.

```bash
python scripts/submit_orion.py infer \
    --model      bnl-nsls2-smi-vit \
    --run        smi/sandbox/confab26_demo/inputs/run_1086139 \
    --output     smi/sandbox/confab26_demo/results/run_1086139_vit \
    --batch-size 1 \
    --thumb-mode logroi \
    --param      temperature:primary/LinkamThermal_temperature_current:float:°C \
    --param      piezo_x:primary/piezo_x:float:μm
```

`--run` points at the BlueskyRun container; frames are read from
`run/primary/<image_key>` (default `pil900KW_image`).
`--param` stores scalar streams from the same primary event stream alongside
each embedding in the `_index` table.

**`--projector` controls projector (UMAP) behaviour:**

| `--projector` value | Effect |
|---|---|
| *(omitted)* | Fit UMAP from scratch over all embeddings after encoding (default) |
| `umap_approx` | Load saved approximator by name (`models_dir/umap_approx` → MLflow) |
| `false` | Skip projections entirely — write NaN |

**`--classifier` controls per-embedding label assignment:**

| `--classifier` value | Effect |
|---|---|
| *(omitted)* | No classification — `label` column is NULL (default) |
| `classifier` | Load saved classifier by name (`models_dir/classifier` → MLflow) and assign labels |

The script polls every 5 s and prints progress. Check the full Slurm log:

```bash
ssh orion-staging.nsls2.bnl.gov \
    "tail -100 $EMBLASE_ORION_WORKING_DIR/slurm-<jobid>.out"
```

---

## Streaming Pipeline

The pipeline watches a Tiled container for new BlueskyRuns and submits one
compute job per run to encode frames and write embeddings incrementally as data
arrives.  The backend is selected with `--backend` (default: `orion`).

> **Order is critical.** Start the watcher *before* data arrives. The watcher
> fires on the `child_created` WebSocket event when a run container is first
> created. If the run is already complete before the compute job starts, WS
> events are not replayed and the job will hang forever.

**Terminal 1 — start the watcher:**

```bash
pixi run python scripts/start_watcher.py \
  --inputs     smi/sandbox/confab26_demo/inputs_copy \
  --output     smi/sandbox/confab26_demo/results \
  --backend    orion \
  --model      bnl-nsls2-smi-vit \
  --image-key  pil900KW_image \
  --batch-size 1 \
  --thumb-mode logroi \
  --param      temperature:primary/LinkamThermal_temperature_current:float:°C \
  --param      piezo_x:primary/piezo_x:float:μm \
  --no-replay
```

Wait until `Press Ctrl+C to stop` appears before proceeding.

**Terminal 2 — simulate acquisition (after watcher is ready):**

```bash
pixi run python scripts/simulate_acquisition.py \
  --src         smi/sandbox/confab26_demo/inputs/run_1086139 \
  --dst         smi/sandbox/confab26_demo/inputs_copy \
  --rename      run_live_1086139 \
  --access-tags smi_sandbox \
  --batch-delay 0.5
```

`simulate_acquisition.py` creates the run container first (triggering the
watcher), then writes frames one at a time. A Unix timestamp is appended to
`--rename` automatically. The Orion job starts ~25–30 s after submission (pixi
env + model load), well before the copy finishes
(`0.5 s/frame × 288 frames ≈ 144 s`).

### `--backend` options

| Value | Description |
|---|---|
| `orion` (default) | Submits a Slurm job via the Orion REST API. Requires `EMBLASE_ORION_API_KEY`. |
| `local` | Runs inference in-process. Useful for development; no HPC account needed. |
| `nersc` | Submits to NERSC Perlmutter via the IRI API + Shifter. Requires `EMBLASE_NERSC_API_TOKEN`. |

### Batch vs. streaming

| | Batch (`submit_orion.py infer`) | Streaming (`start_watcher.py`) |
|---|---|---|
| **When to use** | Complete runs, model iteration | Live acquisition |
| **Params** | ✓ (`--param`) | ✓ (`--param`) |
| **Projector** | `--projector` (name / scratch / false) | `--projector` (name / scratch / false) |
| **Classifier** | `--classifier NAME` (omit = no labels) | `--classifier NAME` (omit = no labels) |
| **Latency** | All frames at once | Incremental, per batch |
| **Setup** | Single command | Watcher must start before data arrives |

### `--param` syntax

```
--param name:source:dtype:units
```

- `source` must be `primary/<array_key>` (a scalar stream aligned 1:1 with frames)
- `dtype`: `float`, `integer`, `string`, or `boolean`
- Repeat for multiple parameters

### `--no-replay`

Skips reprocessing runs already present in `--inputs` on startup. Omit to
replay all existing runs.

---

## Projector (UMAP)

The core logic lives in `emblase.projector` and can be called from IPython with an
already-initialised Tiled node — no CLI required:

```python
from emblase.projector import train_projector, apply_projector

node = client["smi/sandbox/confab26_demo/results/run_xyz"]

# Fit a new approximator and write projections back to Tiled:
train_projector(node, projector_dir="models/umap_approx")

# Apply an existing approximator and write projections back to Tiled:
apply_projector(node, projector_dir="models/umap_approx")
```

Or via the CLI scripts (thin wrappers over the same functions):

```bash
# apply existing approximator
pixi run python scripts/compute_projector.py \
  --dataset smi/sandbox/confab26_demo/results/run_live_1086139_<timestamp>

# retrain approximator then write projections
pixi run python scripts/train_projector.py \
  --dataset smi/sandbox/confab26_demo/results/run_live_1086139_<timestamp>
```

---

## Tiled Layout

```
smi/sandbox/confab26_demo/
├── inputs/             # original BlueskyRuns from the detector
├── inputs_copy/        # runs watched by the streaming pipeline
└── results/            # LatentSpaceEmbedding containers written by Orion
```

Each `results/<run>/` container (`spec="LatentSpaceEmbedding"`) holds:

| Child | Shape | Description |
|-------|-------|-------------|
| `embeddings` | `(N, D)` float32 | embedding vectors |
| `thumbnails` | `(N, H, W)` float32 | downsampled source images |
| `projections` | `(N, 2)` float32 | UMAP coordinates (NaN until computed) |
| `notes` | `(N,)` unicode | freeform mutable annotations |
| `user_labels` | `(N,)` unicode | user-assigned labels |
| `_index` | table | `indx`, `path`, `slice`, `model_version`, `timestamp`, `param_*` |

`indx` is the 0-based row offset into `embeddings`/`projections` — use it as
the stable join key between `_index` rows and array positions (SQL row order is
non-deterministic).

---

## Orion job management

```bash
# check status
python scripts/submit_orion.py status <job_id>

# cancel
python scripts/submit_orion.py cancel <job_id>

# full log
ssh orion-staging.nsls2.bnl.gov \
    "tail -50 $EMBLASE_ORION_WORKING_DIR/slurm-<job_id>.out"

# queue
ssh orion-staging.nsls2.bnl.gov "squeue -u <your-username>"
```

---

## Models

| Registry name | Architecture |
|---------------|-------------|
| `bnl-nsls2-smi-vit` | ViT-based autoencoder (canonical; v2) |
| `vae` | Convolutional VAE with windowed self-attention |

Weights are pulled at job time from the MLflow registry and cached at
`EMBLASE_MODEL_CACHE_DIR` (default `~/.cache/emblase/models`). A local model
directory under `models/<name>/` with a `loader.py` takes priority over the
registry.

---

## Configuration

All settings are read from `.env` (or environment variables), prefixed
`EMBLASE_`. Copy [`.env.example`](.env.example) to `.env` and fill in your
values.

| Variable | Description |
|----------|-------------|
| `EMBLASE_TILED_SERVER_URI` | Tiled server URL |
| `EMBLASE_TILED_API_KEY` | Tiled API key (write access) |
| `EMBLASE_TILED_ACCESS_TAGS` | Comma-separated access tags applied to all written nodes |
| `EMBLASE_ORION_API_KEY` | Orion REST API key |
| `EMBLASE_MLFLOW_TRACKING_URI` | MLflow tracking server URI |
| `EMBLASE_MLFLOW_API_KEY` | MLflow API key |
| `EMBLASE_MODEL_CACHE_DIR` | Local cache for downloaded MLflow model weights |

---

## Development

```bash
pixi run python -m pytest tests/ -x -q   # 122 tests, no GPU or live connections required
```

---

## Project Structure

```
src/emblase/
├── config.py               # Settings (pydantic-settings, EMBLASE_ prefix)
├── models.py               # load_model(), encode()
├── mlflow_registry.py      # MLflow push/pull/list
├── umap.py                 # (legacy) — use projector.py instead
├── projector.py            # train_projector(), apply_projector() — callable from IPython
├── classifier.py           # train_classifier(), apply_classifier() — unsupervised labelling
├── compute/
│   ├── base.py             # ComputeBackend ABC, JobStatus, JobResult
│   ├── orion.py            # OrionBackend, OrionClient, script rendering
│   └── local.py            # LocalBackend (dev/test)
├── pipeline/
│   ├── streaming.py        # InputsWatcher — subscribes to inputs_copy via WebSocket
│   └── copy_tiled.py       # deepcopy() for BlueskyRuns; copy_embedding() for LSE containers
├── tiled/
│   └── client.py           # read_images, write_output, LatentSpaceEmbedding, THUMB_MODES
└── worker/
    ├── inference.py.tmpl           # batch inference node script (params + projector)
    └── streaming_inference.py.tmpl # streaming inference node script (params + projector)
scripts/
├── start_watcher.py        # CLI: watch inputs_copy → submit streaming jobs (--backend orion/local/nersc)
├── simulate_acquisition.py # CLI: copy a run into inputs_copy frame-by-frame
├── simulate_results.py     # CLI/lib: replay a LSE container into a local Tiled (WebUI dev)
├── submit_orion.py         # CLI: submit batch jobs to Orion (infer / status / cancel)
├── submit_nersc.py         # CLI: submit batch/streaming jobs to NERSC (infer / stream / status / cancel / resources)
├── compute_projector.py    # post-process: apply projector approximator → write projections
├── train_projector.py      # fit UMAP + train MLP projector approximator on existing embeddings
└── train_classifier.py     # fit clustering → MLP classifier on existing embeddings
models/
├── noop/                   # NoopEncoder (seeded random, for testing)
├── vit/                    # ViT weights + loader.py (optional local override)
└── umap_approx/            # neural_dimred_wrapper.py, scaler.pkl, umap_approximator.pth (projector model layout)
tests/                      # pytest suite (mocked, no hardware required)
```

---

## Acknowledgements

Emblase builds on ideas from two prior projects at ALS / NSLS-II:

- **[mlex_latent_explorer](https://github.com/mlexchange/mlex_latent_explorer)** —
  the original MLExchange Latent Space Explorer, pioneering the autoencoder +
  UMAP pipeline, MLflow registry integration, Tiled I/O, and streaming
  architecture via Arroyo and Redis.

- **[arroyosas](https://github.com/als-computing/arroyosas)** — the streaming
  small-angle scattering reduction pipeline at ALS, establishing data transport
  patterns (WebSockets, ZMQ, Tiled) and schema conventions.
