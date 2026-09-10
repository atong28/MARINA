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
| `POST` | `/api/fingerprints/explain` | Per-bit calibrated substructure confidences and depiction geometry for the expanded view |
| `POST` | `/api/custom-smiles-card` | Score an arbitrary SMILES against a session fingerprint |
| `GET`  | `/api/queue` | Worker-pool queue depth |
| `GET`  | `/api/queue/{request_id}` | Queue position for one in-flight prediction |
| `GET`  | `/api/stats` | Cumulative query and visitor counts |

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

### Hosting a SPECTRE model

Set `"type": "spectre"` and use the same directory layout. The backend loads
`src.modules.spectre.model.SPECTRE` with `SPECTREArgs(**params.json)` and shares
the MARINA fingerprint loader, retrieval set and renderer — nothing else differs.

Inputs are flattened into one `(1, N, 3)` peak sequence with a parallel type
indicator, matching `SPECTREDataModule.format_inference_data`:

| Modality | Row layout | Type code |
|----------|-----------|-----------|
| HSQC | `[¹³C shift, ¹H shift, intensity]` | 0 |
| ¹³C NMR | `[shift, 0, 0]` | 1 |
| ¹H NMR | `[0, shift, 0]` | 2 |
| MW | `[mw, 0, 0]` | 3 |
| MS/MS | `[m/z, intensity, 0]` | 4 |

Column order matters: coordinate 0 is encoded with `c_wavelength_bounds`,
coordinate 1 with `h_wavelength_bounds`, and coordinate 2 is **sign**-encoded for
NMR (only the sign of the HSQC intensity reaches the model, not its magnitude).
The API accepts HSQC as `[¹H, ¹³C, intensity]` and swaps the first two columns
internally.

Three things are validated at load time, because each fails silently otherwise:

1. **Unknown `params.json` keys are rejected.** Pydantic dataclasses *discard*
   keys they do not define, so a `params.json` from the published SPECTRE code
   would build the model with this codebase's defaults instead of the settings
   the checkpoint was trained with. Translate the file rather than reusing it.
2. **`out_dim` must equal the feature-map size** in `RankingEntropy/bitinfo_to_idx.pkl`.
3. **Rankingset rows must be L2-normalised** (a warning is logged otherwise).
   `RankingSet`'s cosine metric normalises only the query, so unnormalised rows
   inflate every score past 1.0, where they are clamped and every result card
   reads `1.000`. `build_rankingset_csr` does this at build time; a rankingset
   copied from elsewhere may not have it.

Checkpoints are also rejected if `load_state_dict` reports missing parameters,
so an architecture mismatch surfaces as a startup error instead of random weights.

---

## Tests

```bash
make test        # everything (~35 s)
make test-fast   # skips worker-spawning and model-building tests (~9 s)
pytest tests/test_renderer.py -q      # one module
```

231 tests under `tests/`. Nothing requires a trained checkpoint: model-dependent
tests build a small, randomly-initialised model directory from
`tests/conftest.py::build_model_dir`, which is enough to exercise loading,
collation, retrieval and rendering. Tests needing the MARINA `src/` tree skip
rather than fail when it is not reachable.

| Module | Covers |
|---|---|
| `test_similarity.py` | cosine/Tanimoto, length-mismatch rejection |
| `test_rate_limit.py` | spec parsing, client identification, window behaviour |
| `test_rate_limit_integration.py` | throttling as wired into the app |
| `test_stats.py` | counters, IP hashing, persistence, restart |
| `test_api_validation.py` | size caps, malformed payloads, error-message leakage |
| `test_status_routes.py` | `/api/queue`, `/api/stats` |
| `test_session_internals.py` | CSR row filtering (property-tested), MW index, cache bound |
| `test_model_loading.py` | the `params.json` / `out_dim` / checkpoint guards |
| `test_predictor.py` | preprocessing and the SPECTRE collation contract |
| `test_renderer.py` | signed weights, both highlight polarities, payload size, the `HIGHLIGHT_ENABLED` switch |
| `test_bit_explain.py` | per-bit substructure explanations for the expanded view |
| `test_calibration.py` | per-bit probability calibration |
| `test_compute_pool.py` | restart survival, queue positions (marked `slow`) |
| `test_spectre_hosting.py` | end-to-end SPECTRE inference (marked `slow`) |

The collation tests assert against `src/modules/core/const.py` rather than
hard-coded type codes, so changing them on the model side fails here instead of
silently producing wrong predictions.

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

### Rendering

| Variable | Default | Description |
|---|---|---|
| `MOLECULE_IMG_SIZE` | `400` | Depiction size in pixels |
| `RDKIT_ENABLED` | `true` | `false` disables all depictions |
| `HIGHLIGHT_ENABLED` | `true` | `false` serves plain depictions only — see below |
| `HIGHLIGHT_METHOD` | `attribution` | How map weights are computed: `attribution` or `ablation` — see below |

Every result card carries two depictions: `plain_svg`, a transparent vector line
drawing, and `svg`, the similarity map — a transparent colour wash the frontend
layers *under* the line drawing (both are drawn with the same fit, so they
align). Green atoms support the retrieval, pink atoms contradict it. The
frontend's **Similarity map** checkbox adds or removes the wash client-side, so
toggling it never re-runs a search.

`HIGHLIGHT_METHOD` selects how the per-atom weights are computed:

- `attribution` (default): every substructure occurrence in the candidate is
  scored by the model's predicted probability for it minus 0.5 (calibrated when
  the model ships `calibration.json`), and the score is spread over the
  occurrence's atoms, smaller environments weighing more. For counting
  vocabularies a fragment present n times is scored by the mean of its
  ≥1×..≥n× buckets, every copy alike. Reads as "does this atom belong to
  substructures the model expected", agrees with the bit panel, one pass.
- `ablation`: the original SPECTRE rule — the cosine drop from removing the
  environments centred on each atom (for counting vocabularies, one bucket per
  occurrence, from the top). The sign is relative to the candidate's average
  alignment and the result depends on SMILES atom order; kept for comparison.

`HIGHLIGHT_ENABLED=false` skips the map entirely: `svg` comes back `null`, and
`/api/health` reports `highlight_available: false` so the UI hides the checkbox
rather than offering a switch with one position.

### Compute pool

| Variable | Default | Description |
|---|---|---|
| `MAX_COMPUTE_WORKERS` | `2` | Worker processes for `/predict`. **Set to `0` to disable (dev mode).** |
| `MAX_COMPUTE_QUEUE` | `8` | Max queued jobs before 503 |

### Server

| Variable | Default | Description |
|---|---|---|
| `BACKEND_PORT` | `5000` | Listening port |
| `UVICORN_WORKERS` | `2` | Uvicorn worker processes (Docker only). This is the default for `backend/docker-compose.yml`; the full-stack `docker-compose.yml` at the repo root defaults to `1`. |
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
| `mw_min` / `mw_max` | `float?` | Filter retrieval set by monoisotopic mass (same value shown as "Exact mass" on a result card) |

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
