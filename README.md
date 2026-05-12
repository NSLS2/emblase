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

### pip extras

Emblase is split into dependency groups so you can install only what each
environment needs:

| Extra | Use case | Command |
|---|---|---|
| *(none)* / `[base]` | Tiled server — router plugin + client only; no torch/sklearn/mlflow | `pip install emblase` |
| `[compute]` | Compute node — adds torch, transformers, sklearn, umap-learn, mlflow, etc. | `pip install emblase[compute]` |
| `[all]` | Full installation (base + compute) | `pip install emblase[all]` |
| `[dev]` | Development — adds pytest and coverage (does **not** include compute deps) | `pip install emblase[compute,dev]` |

In a Tiled server `requirements.txt`, use:

```
emblase[base]
```

This registers the router and Tiled client without pulling in any ML or HPC
dependencies.

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

Set `EMBLASE_TILED_INPUT_CONTAINER` and `EMBLASE_TILED_OUTPUT_CONTAINER` in
your `.env` file; then `--run` and `--output` accept bare relative keys.
Absolute paths (starting with `/`) override the env-var base.

```bash
python scripts/submit_orion.py infer \
    --model      bnl-nsls2-smi-vit \
    --run        run_1086139 \
    --batch-size 1 \
    --thumb-mode logroi \
    --param      temperature:primary/LinkamThermal_temperature_current:float:°C \
    --param      piezo_x:primary/piezo_x:float:μm
```

`--run` points at the BlueskyRun container key within the input container;
frames are read from `run/primary/<image_key>` (default `primary/pil900KW_image`).
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
  --backend    orion \
  --model      bnl-nsls2-smi-vit \
  --image-key  primary/pil900KW_image \
  --batch-size 1 \
  --thumb-mode logroi \
  --param      temperature:primary/LinkamThermal_temperature_current:float:°C \
  --param      piezo_x:primary/piezo_x:float:μm \
  --no-replay
```

`--inputs` and `--output` default to `EMBLASE_TILED_INPUT_CONTAINER` and
`EMBLASE_TILED_OUTPUT_CONTAINER` from your `.env`. Pass them explicitly
(relative or absolute) to override.

Wait until `Press Ctrl+C to stop` appears before proceeding.

**Terminal 2 — simulate acquisition (after watcher is ready):**

```bash
pixi run python scripts/simulate_acquisition.py \
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

node = client["<your-output-container>/<run_key>"]

# Fit a new approximator and write projections back to Tiled:
train_projector(node, projector_dir="models/umap_approx")

# Apply an existing approximator and write projections back to Tiled:
apply_projector(node, projector_dir="models/umap_approx")
```

Or via the CLI scripts (thin wrappers over the same functions):

```bash
# apply existing approximator
pixi run python scripts/compute_projector.py \
  --dataset <your-output-container>/<run_key>

# retrain approximator then write projections
pixi run python scripts/train_projector.py \
  --dataset <your-output-container>/<run_key>
```

---

## Tiled Layout

```
<proposal>/<pi>/<project>/
├── _inputs/            # original BlueskyRuns from the detector (EMBLASE_TILED_INPUT_CONTAINER)
└── results/            # LatentSpaceEmbedding containers written by jobs (EMBLASE_TILED_OUTPUT_CONTAINER)
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

## NERSC job management

Jobs are submitted to Perlmutter via the [IRI Superfacility API](https://api.iri.nersc.gov/docs)
— no SSH required.

### How it works

1. `NERSCBackend` renders the Python inference script locally (same template
   used by Orion).
2. All secrets (`EMBLASE_TILED_*`, `EMBLASE_MLFLOW_*`) and `JOB_DIR` are
   baked into a short Python preamble prepended to the script at submit time.
3. The script is uploaded to `/pscratch` via `POST /filesystem/upload/scratch`
   and immediately `chmod 400` (owner read-only).
4. The job runs `python <script_path>` inside the podman-hpc container with
   `/pscratch` bind-mounted at the same path.
5. When `--output` (Tiled) is set, embeddings are written directly to Tiled —
   no files land on `/pscratch`. When omitted, `output.npy` is saved to
   `JOB_DIR` and can be retrieved via `GET /filesystem/download/scratch`.

```bash
# submit batch inference (--run relative to EMBLASE_TILED_INPUT_CONTAINER)
python scripts/submit_nersc.py infer --model bnl-nsls2-smi-vit \
    --run run_xyz

# submit streaming (listens for new frames in a Tiled run)
python scripts/submit_nersc.py stream --model bnl-nsls2-smi-vit \
    --run run_xyz

# check status
python scripts/submit_nersc.py status <task_id>

# cancel
python scripts/submit_nersc.py cancel <task_id>

# tail job output log (path printed at submission time)
python scripts/submit_nersc.py logs <task_id>

# list available Perlmutter resources
python scripts/submit_nersc.py resources

# browse /pscratch (scratch is the default resource)
python scripts/submit_nersc.py ls /pscratch/sd/d/<user>

# browse $HOME on Perlmutter
python scripts/submit_nersc.py ls /global/u2/<i>/<user> --resource homes

# read a remote file
python scripts/submit_nersc.py cat /pscratch/sd/d/<user>/somefile
```

### Perlmutter queues (`EMBLASE_NERSC_QUEUE`)

The `queue_name` field maps to Slurm **partition** names (not QOS names).
Valid values on Perlmutter:

| `EMBLASE_NERSC_QUEUE` | Slurm partition | Slurm QOS | Notes |
|---|---|---|---|
| `shared` *(default)* | `shared_gpu_ss11` | `gpu_shared` | Shares nodes — fastest dispatch for single-GPU jobs |
| `debug` | `gpu_ss11` | `gpu_debug` | Fast dispatch, ≤ 30 min wall-clock cap |
| `regular` | `gpu_ss11` | `gpu_regular` | Standard allocation queue |
| *(empty)* | `gpu_ss11` | `gpu_debug` | Scheduler default (same as `debug`) |

> **Note:** `gpu_shared`, `gpu_ss11`, `gpu_debug` etc. are QOS names — the IRI
> API expects partition names (left column above) and returns 400 for QOS names.

### NERSC account suffix

GPU jobs on Perlmutter must bill against the `_g`-suffixed project code:

```
EMBLASE_NERSC_ACCOUNT=m3792_g   # not m3792
```

### Container and script delivery

- The Shifter container image (`ghcr.io/genematx/emblase:latest`) includes
  all `[compute]` dependencies: torch, transformers, mlflow, sklearn, etc.
- `/pscratch` is the only volume mounted into the container. Two mounts are
  not supported by podman-hpc on Perlmutter (produces `invalid reference
  format`), so scripts and model weights both live on `/pscratch`.
- The inference script is uploaded fresh at submission time via the IRI
  filesystem API; no container rebuild is needed when the script changes.
- Secrets are baked into the script preamble at submit time and never written
  to a separate file. The script is `chmod 400` immediately after upload.

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

### Compute / pipeline variables

| Variable | Description |
|----------|-------------|
| `EMBLASE_TILED_SERVER_URI` | Tiled server URL |
| `EMBLASE_TILED_API_KEY` | Tiled API key (write access) |
| `EMBLASE_TILED_ACCESS_TAGS` | Comma-separated access tags applied to all written nodes |
| `EMBLASE_ORION_API_KEY` | Orion REST API key |
| `EMBLASE_MLFLOW_TRACKING_URI` | MLflow tracking server URI |
| `EMBLASE_MLFLOW_API_KEY` | MLflow API key |
| `EMBLASE_ORION_MODELS_DIR` | Path to model weights on Orion NFS (default `<repo>/models`) |
| `EMBLASE_NERSC_API_TOKEN` | IRI Superfacility API bearer token (Globus `iri_api` scope) |
| `EMBLASE_NERSC_RESOURCE_ID` | Perlmutter resource ID (default `perlmutter`) |
| `EMBLASE_NERSC_ACCOUNT` | NERSC project account; use `_g` suffix for GPU (e.g. `m3792_g`) |
| `EMBLASE_NERSC_QUEUE` | Slurm partition name (default `shared` — see queue table above) |
| `EMBLASE_NERSC_CONSTRAINT` | Slurm node constraint (default `""` — leave empty to avoid long waits) |
| `EMBLASE_NERSC_CONTAINER_IMAGE` | Shifter container image (e.g. `ghcr.io/genematx/emblase:latest`) |
| `EMBLASE_NERSC_WORKING_DIR` | Absolute scratch path for scripts and logs (e.g. `/pscratch/sd/d/<user>/emblase/jobs`) |
| `EMBLASE_NERSC_MODELS_DIR` | Absolute scratch path where MLflow models are cached on Perlmutter |
| `EMBLASE_NERSC_TIME_LIMIT` | Wall-clock limit passed to Slurm (default `01:00:00`) |

### Chat proxy variables (Tiled router plugin)

The `emblase.tiled.router` plugin proxies browser chat requests to the AmSC
chat service.  Authentication uses **per-user Entra OBO** (On-Behalf-Of) when
all four `EMBLASE_ENTRA_*` variables are set, and falls back to a static
service-account token otherwise.

| Variable | Description |
|----------|-------------|
| `EMBLASE_CHATAPP_URL` | Base URL of the AmSC chat service (default: `https://chat-amsc-dev.nsls2.bnl.gov`) |
| `EMBLASE_CHATAPP_MODEL` | LLM model name forwarded to AmSC (default: `openai/gpt-oss-20b`) |
| `EMBLASE_ENTRA_TENANT_ID` | Azure AD tenant ID — required for OBO |
| `EMBLASE_ENTRA_CLIENT_ID` | Tiled's own Entra app registration client ID — required for OBO |
| `EMBLASE_ENTRA_CLIENT_SECRET` | Tiled's Entra client secret — required for OBO |
| `EMBLASE_CHATAPP_SCOPE` | AmSC scope, e.g. `api://<chatapp_client_id>/access_as_user` — required for OBO |
| `EMBLASE_CHATAPP_TOKEN` | Fallback static bearer token (service account); used only when OBO vars are absent |

#### How OBO works

When a user opens the Latent Space Explorer, their browser holds a short-lived
Tiled HMAC JWT (minted by Tiled after Entra OIDC login).  The Emblase router
extracts the Entra `access_token` and `refresh_token` stored inside that JWT's
`state` payload, and performs a Microsoft OBO exchange to obtain a fresh Entra
token scoped to AmSC.  This token is forwarded to AmSC in `Authorization:
Bearer`, giving AmSC the user's real Entra identity.

When the Entra access token expires (~1 h), the router silently refreshes it
using the stored refresh token and writes the new tokens back to the Tiled
session DB — so the user never needs to log in again for the lifetime of their
Tiled session.

**Prerequisites on the Entra app registration:**

1. The `offline_access` permission must be consented (enables refresh tokens).
2. The Tiled client secret (`EMBLASE_ENTRA_CLIENT_SECRET`) must be from the
   same app registration as `EMBLASE_ENTRA_CLIENT_ID`.
3. AmSC's app registration must expose the `access_as_user` scope and the
   Tiled app registration must be granted permission to request it.

**Security note:** The Entra access token lives inside the Tiled HMAC JWT,
which is base64-encoded (not encrypted) and sent to the browser.  The JWT is
short-lived (default 15 min) and only transmitted over HTTPS.  This is a
conscious design trade-off — the alternative would require an additional
server-side token store.

---

## Development

```bash
pixi run python -m pytest tests/ -x -q   # 188 tests, no GPU or live connections required
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
│   ├── nersc.py            # NERSCBackend, NERSCClient (IRI REST API + Shifter)
│   └── local.py            # LocalBackend (dev/test)
├── pipeline/
│   ├── streaming.py        # InputsWatcher — subscribes to inputs_copy via WebSocket
│   └── copy_tiled.py       # deepcopy() for BlueskyRuns; copy_embedding() for LSE containers
├── tiled/
│   ├── client.py           # read_images, write_output, LatentSpaceEmbedding, THUMB_MODES
│   ├── router.py           # FastAPI router plugin: static asset serving + chat OBO proxy
│   └── static/
│       └── main.js         # Compiled Latent Space Explorer UI (built from ui/latent-space-explorer/)
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
ui/
└── latent-space-explorer/  # React/TypeScript source for the Latent Space Explorer plugin
    └── src/
        └── embedding-scatter.tsx  # Main scatter plot component (build → static/main.js)
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
