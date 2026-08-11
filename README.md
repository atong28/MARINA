# MARINA Repository

Multi-modal NMR/MS to molecular-fingerprint retrieval model, the web application that
serves it, and the analyses supporting the papers.

| Directory | What |
|---|---|
| [`src/`](src/) | Training and evaluation code — model, datasets, losses, CLI |
| [`analysis/`](analysis/README.md) | Standalone analyses, each with its own environment, README and report |
| [`scripts/`](scripts/) | Dataset pipeline, benchmarking, website deployment |
| [`backend/`](backend/), [`frontend/`](frontend/) | The web application |
| [`nautilus/`](nautilus/) | Kubernetes job specs for GPU training on NRP Nautilus |

## Analyses

[`analysis/`](analysis/README.md) holds eight self-contained analyses — fingerprint
redundancy, dataset domain comparisons, CLS-token attribution, and the MARINA2/3/4 dataset
builder among them. Each pins its own pixi environment and documents its own run order.
Four carry a LaTeX report with a compiled PDF.

Data living outside the repo is resolved through a single environment variable:

```bash
export MARINA_DATA_ROOT=/path/to/data-root   # holds Datasets/, Benchmark/, Checkpoints/
```

Bulk inputs and large derived artifacts are not committed; see
[`analysis/README.md`](analysis/README.md) for what ships and what has to be re-fetched.

## Website Setup

Full deployment documentation lives in [`docs/website/`](docs/website/README.md):

- [Deployment guide](docs/website/README.md) — architecture, prerequisites, model download, `.env` reference, troubleshooting
- [Local deployment](docs/website/local-deployment.md) — expose a host port on `localhost` or a LAN
- [Cloudflare Tunnel](docs/website/cloudflare-tunnel.md) — public HTTPS hostname with no inbound ports

Quick version: ensure `docker` and `docker-compose` are installed, and copy `.env.example` to `.env`, and configure the variables as you see fit. `MODEL_DATA_DIR` defaults to `./checkpoints` (or you can set an absolute path), and by default the website will run on cpu inference. Note that some legacy machines may be buggy with numpy, so if there is a repeated import error then set `LEGACY_NUMPY=true`.

Download the model (run it in a environment with `gdown` installed, it is installed if you download `pixi` and install the environment below)
```bash
bash scripts/website/download_model.sh
```

Start the docker containers:
```bash
bash scripts/website/start.sh
```

### Website tests

```bash
cd backend  && make test    # 231 pytest tests (~35 s); make test-fast for ~9 s
cd frontend && npm test     # 89 vitest tests (~2 s)
```

Neither suite needs a trained checkpoint or a running server. See
[`backend/README.md`](backend/README.md#tests) for what each module covers.

## Code Installation

Install pixi according to the following instructions:
```
https://pixi.sh/dev/installation/
```
If you are not running on linux-64, you can try adding your distro into `pixi.toml` and install anyways, but no guarantees for support. Running the following command should automatically boot you into the shell with the loaded environment:
```
pixi shell
```
To just install the environment, use
```
pixi i
```

