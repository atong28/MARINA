# MARINA Backend

FastAPI inference backend for MARINA/SPECTRE molecular structure annotation.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | Model readiness + uptime |
| `GET`  | `/api/health/live` | Liveness probe (always 200 while process is alive) |
| `GET`  | `/api/models` | List available models and their load status |
| `POST` | `/api/predict` | Run MARINA inference on spectral data → top-k molecules |
| `POST` | `/api/smiles-search` | Nearest-neighbour retrieval from a SMILES query |
| `POST` | `/api/fingerprints/indices` | Return active entropy-fingerprint bit indices for a SMILES |

Interactive API docs are served at `http://localhost:5000/docs` when the server is running.

---

## Data layout

Each model lives in its own subdirectory and must contain:

```
data/
├── models.json              ← model manifest (see below)
└── marina_best/             ← one directory per model
    ├── params.json
    ├── best.ckpt
    ├── retrieval.pkl
    ├── metadata.json
    └── RankingEntropy/
        ├── rankingset.pt
        └── bitinfo_to_idx.pkl
```

**`models.json`** – exactly one entry must have `"default": true`:

```json
{
  "models": [
    {
      "id": "marina_best",
      "root": "marina_best",
      "type": "marina",
      "default": true,
      "display_name": "MARINA (best)"
    }
  ]
}
```

`type` must be `"marina"` or `"spectre"`.  
`root` is resolved relative to `DATA_DIR` (or as an absolute path).

---

## Quick start: Docker

### 1. Configure paths

```bash
cp .env.example .env
# Edit .env:
#   MARINA_PROJECT_ROOT=/absolute/path/to/MARINA
#   MODEL_DATA_DIR=/absolute/path/to/model-data
```

### 2. Build and run (CPU)

```bash
make build
make up
```

### 3. Verify

```bash
curl http://localhost:5000/api/health
```

### GPU support

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), then:

```bash
# Build with CUDA 12.4 wheels
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 make build

# Run with GPU inference
DEVICE=cuda make up
```

Uncomment the `deploy.resources.reservations` block in `docker-compose.yml` to enforce GPU assignment.

---

## Development mode (no Docker, hot-reload)

Dev mode runs uvicorn with `--reload` so the server restarts on every file save.
The compute worker pool is disabled (`MAX_COMPUTE_WORKERS=0`), so predictions run
inline in the main process – simpler to debug and fully compatible with hot-reload.

### 1. Install dependencies

```bash
cd backend/
make install
# Or manually:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Configure `.env.dev`

```bash
cp .env.example .env.dev
```

Edit `.env.dev` – the three paths you must set:

```dotenv
MARINA_ROOT=..                        # directory containing src/  (one level up)
DATA_DIR=/path/to/model-data
MODEL_ROOT=/path/to/model-data/marina_best
DATASET_ROOT=/path/to/model-data      # same as DATA_DIR is fine
```

### 3. Run

Requires **bash** (for `make dev`). The Makefile runs `set -a; source .env.dev`
from the `backend/` directory so every variable in that file is exported to
`uvicorn` (simple `KEY=value` lines only; quote values that contain spaces).

```bash
cd backend/
make dev
```

The server listens on `BACKEND_PORT` from `.env.dev` (default **5000** if unset)
and reloads whenever a file under `app/` changes. Model data and the MARINA
source tree are **not** watched (no spurious reloads from large checkpoints).

---

## Environment variables reference

### Paths

| Variable | Default | Description |
|---|---|---|
| `MARINA_ROOT` | `/marina` | MARINA project root (must contain `src/`) |
| `DATA_DIR` | `data` | Root for model subdirectories and `models.json` |
| `MODEL_ROOT` | `$DATA_DIR/marina_best` | Fallback model root when `models.json` is absent |
| `DEFAULT_MODEL_ID` | `marina_best` | ID used when no `model_id` is sent in a request |
| `MODELS_JSON_PATH` | `$DATA_DIR/models.json` | Explicit path to `models.json` |
| `DATASET_ROOT` | *(from `DATA_DIR`)* | Required by `src/modules/core/const.py` |

### Inference

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cpu` | PyTorch device string: `cpu`, `cuda`, `cuda:0`, … |
| `MAX_TOP_K` | `50` | Hard cap on `k` per request |
| `PREDICT_TIMEOUT_S` | `60` | Seconds before predict returns 504 |

### Compute pool

| Variable | Default | Description |
|---|---|---|
| `MAX_COMPUTE_WORKERS` | `2` | Worker processes for `/predict`. **Set to `0` to disable (dev mode).** |
| `MAX_COMPUTE_QUEUE` | `8` | Max queued jobs before 503 |

### Server

| Variable | Default | Description |
|---|---|---|
| `BACKEND_PORT` | `5000` | Listening port |
| `UVICORN_WORKERS` | `2` | Uvicorn worker processes (Docker only) |
| `PRELOAD_MODELS` | `default` | `default` / `all` / comma-separated model IDs |
| `MAX_LOADED_MODELS` | `0` | Max models in memory (0 = unlimited, LRU eviction when > 0) |

---

## API examples

### `POST /api/predict`

```bash
curl -s -X POST http://localhost:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "raw": {
      "hsqc": [7.23, 128.5, 1.0, 3.81, 55.2, 0.8],
      "mw": 350.0
    },
    "k": 5
  }' | python -m json.tool
```

**Request fields:**

| Field | Type | Description |
|---|---|---|
| `raw.hsqc` | `float[]` | Flat triplets `[H_shift, C_shift, intensity, …]` |
| `raw.h_nmr` | `float[]` | ¹H shifts (ppm) |
| `raw.c_nmr` | `float[]` | ¹³C shifts (ppm) |
| `raw.mass_spec` | `float[]` | Flat pairs `[m/z, intensity, …]` |
| `raw.mw` | `float` | Molecular weight (Da) |
| `k` | `int` | Number of results (1–50, default 10) |
| `model_id` | `string?` | Override the default model |
| `mw_min` / `mw_max` | `float?` | Filter retrieval set by molecular weight |

**Response:** `results[]` array of `ResultCard` objects, each containing `smiles`,
`cosine_similarity`, `tanimoto_similarity`, `svg`, `plain_svg`, `name`,
`database_links`, `retrieved_molecule_fp_indices`, and `exact_mass`.

---

### `POST /api/smiles-search`

```bash
curl -s -X POST http://localhost:5000/api/smiles-search \
  -H 'Content-Type: application/json' \
  -d '{"smiles": "c1ccccc1", "k": 5}' | python -m json.tool
```

---

### `POST /api/fingerprints/indices`

```bash
curl -s -X POST http://localhost:5000/api/fingerprints/indices \
  -H 'Content-Type: application/json' \
  -d '{"smiles": "c1ccccc1"}' | python -m json.tool
```

Returns `{"smiles": "c1ccccc1", "fp_indices": [42, 137, 891, …]}`.

---

## Multi-model setup

Add additional entries to `models.json`:

```json
{
  "models": [
    {"id": "marina_best", "root": "marina_best", "type": "marina", "default": true},
    {"id": "spectre_best", "root": "spectre_best", "type": "spectre", "default": false}
  ]
}
```

Pass `"model_id": "spectre_best"` in any request to use that model.
Models are loaded on demand and cached (LRU eviction configurable via `MAX_LOADED_MODELS`).
