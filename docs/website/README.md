# MARINA Website — Deployment Guide

The MARINA website is a three-container Docker stack: a React SPA, a FastAPI
inference backend, and an nginx reverse proxy in front of both.

Pick a deployment path:

| Guide | Use when |
|---|---|
| [**Local deployment**](local-deployment.md) | You want the site on `http://localhost` or on a LAN/lab machine you reach by IP. Requires an open inbound port. |
| [**Cloudflare Tunnel**](cloudflare-tunnel.md) | You want a public HTTPS URL on your own domain, from a machine behind NAT or a firewall, with no inbound port and no public IP. |

Both paths share the same prerequisites, model download, and `.env` file —
covered below. Read this page first, then follow one of the two guides.

**Quick deploy on a new host** (Docker + Compose, `curl`, `unzip`): clone the
repository, then

```bash
bash scripts/website/deploy.sh          # port 8643; `deploy.sh <port>` to override
```

It downloads the weights (skipped if present), records the port as `NGINX_PORT`
in `.env`, builds the images, starts the stack and waits until the model is
loaded. Re-run it after a `git pull` to rebuild and restart. Every other setting
keeps its default (CPU inference); see [Settings reference](#settings-reference).

---

## Architecture

```
                  ┌─────────────────────────────────────────────┐
                  │  proxy-net                    backend-net    │
  browser ──80──▶ │  nginx ──▶ frontend ─────────────▶ backend   │
                  │   (edge)    (SPA + own nginx)      :5000     │
                  │     └────────────────────────────▶ backend   │
                  └─────────────────────────────────────────────┘
```

- **nginx** (`nginx/`) is the only container with a host-published port. It
  serves the SPA via the frontend container and routes `/api/*` to the backend.
- **frontend** (`frontend/`) is a Vite/React SPA built into static files and
  served by its own nginx. It has no host port.
- **backend** (`backend/`) is FastAPI + uvicorn running MARINA inference. It has
  no host port and sits on `backend-net`. The edge proxy joins both networks, so
  it can reach the backend; nothing outside the stack can.

The SPA calls the API from the **browser** at the relative path `VITE_API_BASE`
(default `/api`). Those requests go back out to the edge nginx, which is why
`/api/*` is always routed there — see
[the API surface](#the-api-surface-and-what-expose_api_via_nginx-does) below.

---

## Prerequisites

- Docker Engine 20.10+ and the Compose v2 plugin (`docker compose version`).
  The Cloudflare overlay uses the `!reset` YAML tag, which needs Compose **v2.24+**.
- ~10 GB free disk for images (the backend image includes PyTorch).
- ~6 GB RAM per model copy. Total copies in memory =
  `UVICORN_WORKERS × (1 + MAX_COMPUTE_WORKERS)` per model, 3 at the stack defaults. The two served models
  measure ~17.5 GB together (the container cap defaults to 28 GB).
- GPU inference additionally needs the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

All commands below run from the repository root (the directory containing
`docker-compose.yml`).

---

## 1. Download the model

Needs only `curl` and `unzip` (`gdown` is used instead if installed). Run from
anywhere; the script resolves the repository root itself.

```bash
bash scripts/website/download_model.sh          # skips a model already on disk
bash scripts/website/download_model.sh --force  # re-download
```

The bundle, `marina_db_release_s0.zip` (2.86 GB), is published on Google Drive as file id
`132U2C8XleodH7e1PJvfCaJHq-GhQbSls`, the script's default. To fetch it from elsewhere, export `MARINA_BUNDLE_URL` with a
direct URL (e.g. a Zenodo record file); it takes precedence. `MARINA_RELEASE_ID` overrides the Drive id. The script
checks the download's sha256 (`fd0e86289480492986980133b32c4c25e321be89e20f793923c92e756c81c9a2`) and unpacks the two
model directories into `checkpoints/`. It then writes `checkpoints/models.json`:

```json
{
    "models": [
        {
            "id": "marina_flag_s0",
            "root": "marina_flag_s0",
            "type": "marina",
            "default": true,
            "display_name": "MARINA-DB flagship (unique-multiplicity, formula-capable)"
        },
        {
            "id": "marina_shifts_s0",
            "root": "marina_shifts_s0",
            "type": "marina",
            "role": "hsqc_unsigned",
            "display_name": "MARINA-DB HSQC shifts-only (no multiplicity sign)"
        }
    ]
}
```

- **`marina_flag_s0`** is the flagship `marina-db-flagship-uniqmult-s0` checkpoint (epoch 616). It was trained on the
  release MARINA-DB with `RankingEntropyUniqueMultiplicity`, radius 10, and ranks against the 531,927-structure
  MARINA-DB retrieval set. It answers every request, except one kind:
- **`marina_shifts_s0`** is `marina-db-flagship-hsqcshifts-s0` (epoch 638). It was trained on HSQC shifts without the
  multiplicity-edit sign. Its `"role": "hsqc_unsigned"` keeps it out of ordinary selection: it answers exactly the
  requests whose HSQC has **no** sign/intensity column (every third value of `raw.hsqc` null). A column missing for
  only some rows is rejected as incomplete. The response names the model that answered, and the UI says so above
  the results. On MARINA-Bench 456 test it trails the flagship (rank@1 HSQC-only 31.6 vs 44.0, NMR 46.7 vs 56.8, 3-seed
  means), but it does far better than feeding the flagship a made-up sign.

Seed s0 is the best of three on validation (MARINA-Bench 456, NMR + formula, rank@1) for both. These replaced
`marina_db_s1` (MARINA-DB-OPEN-OLD) in October 2026. Each model's `calibration.json` and `metrics.json` are only valid
for that checkpoint. Copying them across models would silently serve wrong confidences. The vocabulary must match
the checkpoint too, or retrieval ranks in the wrong space.

Expected layout afterwards:

```
checkpoints/
├── models.json
├── marina_flag_s0/
│   ├── params.json
│   ├── best.ckpt
│   ├── calibration.json
│   ├── metrics.json
│   ├── index.pkl
│   ├── retrieval.pkl
│   ├── metadata.json
│   ├── npclassifier.json
│   ├── mw_index.json
│   ├── formula_index.json
│   └── RankingEntropyUniqueMultiplicity/
│       ├── rankingset.pt
│       └── bitinfo_to_idx.pkl
└── marina_shifts_s0/
    ├── params.json, best.ckpt, calibration.json, metrics.json
    └── everything else: symlinks to ../marina_flag_s0/ (same release bank)
```

Two models need roughly twice the memory of one: about 17.5 GB with the stack defaults, so set
`BACKEND_MEM_LIMIT` above that (e.g. `28g`) in `.env`. The fallback loads on its first request.

`root` is resolved relative to the mounted data directory. A `root` that does
not exist is skipped with a warning at startup rather than crashing the backend,
so check `docker compose logs backend` for `models.json:` warnings if a model
you expect is missing from the selector.

### Molecular-weight index

The retrieval MW filter needs a monoisotopic mass per database entry.
`metadata.json` does not carry one, so the backend derives them from the
structures with RDKit and caches the result as `mw_index.json`. That pass runs
at roughly 2k structures/s — minutes for a full database — and happens on the
first MW-filtered request unless it is precomputed:

```bash
python scripts/website/build_mw_index.py checkpoints/marina_flag_s0
```

Precompute it whenever a model directory is first deployed. The download bundles
already ship `mw_index.json`, so this is only needed for a directory assembled
without one. If the mount is read-only the index is rebuilt on every start,
which is correct but slow; run the script and ship the file instead.

Hosting more than one model, or a SPECTRE checkpoint, is documented in
[`backend/README.md`](../../backend/README.md#data-layout).

---

## 2. Create `.env`

```bash
cp .env.example .env
```

Every setting is optional — the defaults give a working CPU deployment on port
80. A typical file:

```dotenv
MODEL_DATA_DIR=./checkpoints
NGINX_PORT=8080
DEVICE=cpu
```

### The API surface, and what `EXPOSE_API_VIA_NGINX` does

`/api/*` is routed to the backend in **both** settings. The toggle controls only
the interactive API documentation:

| Path | `false` (default) | `true` |
|---|---|---|
| `/` | SPA | SPA |
| `/api/health`, `/api/predict`, … | backend | backend |
| `/docs`, `/redoc`, `/openapi.json` | SPA catch-all (docs unreachable) | backend |
| "API Docs" link in site header | hidden | shown |

The toggle cannot hide the API itself, and no setting can. MARINA's frontend is
a client-side SPA: it fetches `/api/*` from the visitor's browser through this
same proxy, so any rule that blocks those routes blocks the site's own
predictions and search along with everything else. There is no server-side path
that could be treated differently.

If the API must not be public, put authentication in front of the whole site —
[Cloudflare Access](cloudflare-tunnel.md#locking-the-site-down) is the least
invasive option and needs no changes to this repo.

Because `EXPOSE_API_VIA_NGINX` is also baked into the frontend bundle as
`VITE_SHOW_API_DOCS`, changing it needs an image rebuild for the header link to
match: `docker compose up -d --build`.

### Settings reference

| Variable | Default | Notes |
|---|---|---|
| `MODEL_DATA_DIR` | `./checkpoints` | Mounted read-only at `/data`. Relative paths resolve from the repo root. |
| `NGINX_PORT` | `8643` | Host port for the edge proxy. Ignored in tunnel mode. |
| `EXPOSE_API_VIA_NGINX` | `false` | Routes the API docs endpoints; see above. |
| `DEVICE` | `cpu` | `cpu`, `cuda`, or `cuda:0`. |
| `TORCH_INDEX_URL` | CPU wheels | Build-time. `.../whl/cu124` or `.../whl/cu128` for GPU. |
| `LEGACY_NUMPY` | `false` | Build-time. Set `true` on older CPUs that fail with `NumPy was built with baseline optimizations (X86_V2)`. |
| `UVICORN_WORKERS` | `1` | Multiplies memory use, and the effective rate limit, by its value. `backend/docker-compose.yml` defaults to 2; this stack defaults to 1. |
| `MAX_COMPUTE_WORKERS` | `2` | Worker processes serving `/api/predict`. |
| `MAX_COMPUTE_QUEUE` | `8` | Pending jobs before the API returns 503. |
| `BACKEND_MEM_LIMIT` | `16g` | Memory cap for the backend container (~3 GB per model copy). |
| `PRELOAD_MODELS` | `default` | `default`, `all`, or comma-separated model IDs. |
| `HIGHLIGHT_ENABLED` | `true` | `false` serves plain depictions only and hides the UI's **Similarity map** toggle. Saves a per-atom fingerprint ablation on every result card, which is the bulk of card-building time on CPU. |
| `STATS_PATH` | `/var/lib/marina/stats.json` | On the writable `marina-state` volume; `/data` is read-only. |

Backend-only variables (`RATE_LIMIT_PREDICT`, `MAX_HEAVY_ATOMS`, `MAX_TOP_K`, `PREDICT_TIMEOUT_S`,
`CORS_ALLOW_ORIGINS`, …) are documented in
[`backend/README.md`](../../backend/README.md#environment-variables-reference).
To set them, add them under `backend.environment` in `docker-compose.yml`.

---

## 3. Deploy

Continue with [**local deployment**](local-deployment.md) or
[**Cloudflare Tunnel**](cloudflare-tunnel.md).

---

## Common operations

`scripts/website/start.sh` wraps the usual Compose commands and can be run from
anywhere — it resolves the repo root from its own location.

```bash
bash scripts/website/start.sh start      # start, no rebuild
bash scripts/website/start.sh restart    # rebuild images + recreate  [default]
bash scripts/website/start.sh build      # build images only
bash scripts/website/start.sh logs       # tail all logs
bash scripts/website/start.sh status     # container status
bash scripts/website/start.sh stop       # stop and remove
```

Flags: `--gpu` (CUDA 12.8 wheels, `DEVICE=cuda`) or `--cpu` (force CPU-only).
With neither flag the script falls back to `DEVICE=cpu` if `.env` does not set
it, and because it exports the value, that fallback takes precedence over the
`.env` file. Set `DEVICE` in `.env` *and* pass `--gpu`, or use plain Compose:

```bash
docker compose build
docker compose up -d
docker compose logs -f backend
docker compose down
```

The Cloudflare overlay is not understood by `start.sh` — use plain Compose in
tunnel mode.

### Applying changes

| You changed | Command |
|---|---|
| `DEVICE`, worker counts, model paths | `docker compose up -d --force-recreate` |
| `EXPOSE_API_VIA_NGINX` | `docker compose up -d --build` (rebuilds the frontend so the header link matches) |
| `NGINX_PORT` | `docker compose up -d` |
| `TORCH_INDEX_URL`, `LEGACY_NUMPY` | `docker compose build backend && docker compose up -d` |
| Files in `checkpoints/` | `docker compose restart backend` |
| Backend or frontend source, or `src/` (baked into the backend image) | `docker compose up -d --build` |

---

## Health checks

```bash
# Is the process alive?
curl -sf http://localhost/api/health/live && echo OK

# Are the models loaded?
curl -s http://localhost/api/health | python -m json.tool

# End-to-end prediction
curl -s -X POST http://localhost/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"raw": {"hsqc": [7.23, 128.5, 1.0, 3.81, 55.2, 0.8], "mw": 350.0}, "k": 5}' \
  | python -m json.tool | head -20
```

Replace `http://localhost` with `http://localhost:$NGINX_PORT` or your tunnel
hostname as appropriate. The full endpoint list is in
[`backend/README.md`](../../backend/README.md#endpoints).

---

## Troubleshooting

**Backend exits during startup, or restarts in a loop.**
`docker compose logs backend`. Common causes: `models.json` missing or listing
no usable model; an `out_dim` / fingerprint-size mismatch; a checkpoint whose
architecture does not match `params.json`. The backend deliberately fails loudly
on these rather than serving random weights — see
[`backend/README.md`](../../backend/README.md#data-layout).

**The model selector is empty, or a model is missing.**
Its `root` did not resolve. Look for `models.json:` warnings in the backend log —
a root that does not exist is skipped rather than fatal.

**`NumPy was built with baseline optimizations (X86_V2)`.** Set
`LEGACY_NUMPY=true` in `.env` and rebuild: `docker compose build backend`.

**Every result card reads `1.000`.** The rankingset rows are not L2-normalised.
See [`backend/README.md`](../../backend/README.md#hosting-a-spectre-model).

**503 from `/api/predict`.** The compute queue is full. Raise
`MAX_COMPUTE_QUEUE`, or `MAX_COMPUTE_WORKERS` if you have the RAM.

**429 from `/api/predict`.** Per-client rate limit (`30 per minute` by default).
If *all* users hit this at once, the backend is seeing one shared client IP —
expected behind a tunnel without the real-IP fix
([Cloudflare guide](cloudflare-tunnel.md#step-6-verify-client-ips)).

**Predictions run on CPU despite `DEVICE=cuda`.** Either the
`deploy.resources.reservations` block in `docker-compose.yml` is still commented
out, the image was built with CPU wheels, or `start.sh` was run without `--gpu`
(see [Common operations](#common-operations)). Check with
`docker compose exec backend python -c "import torch; print(torch.cuda.is_available())"`.

**Port 80 already in use.** Set `NGINX_PORT` to something free, or use the
tunnel path, which publishes no port at all.
