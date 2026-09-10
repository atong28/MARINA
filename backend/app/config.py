"""
Configuration loaded from environment variables.
All settings have sensible defaults for development.
"""
import os

# ── Inference device ──────────────────────────────────────────────────────────
# Set DEVICE=cuda (or cuda:0) to run on GPU; default is CPU-only.
DEVICE: str = os.getenv("DEVICE", "cpu")

# ── Server ────────────────────────────────────────────────────────────────────
HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
PORT: int = int(os.getenv("BACKEND_PORT", "5000"))

# ── Model paths ───────────────────────────────────────────────────────────────
# DATA_DIR is the root that holds per-model subdirectories and models.json.
DATA_DIR: str = os.getenv("DATA_DIR", "data")
MODELS_JSON_PATH: str = os.getenv("MODELS_JSON_PATH", os.path.join(DATA_DIR, "models.json"))
# Fallback single-model root (used when models.json is absent).
MODEL_ROOT: str = os.getenv("MODEL_ROOT", os.path.join(DATA_DIR, "marina_best"))
DEFAULT_MODEL_ID: str = os.getenv("DEFAULT_MODEL_ID", "marina_best")

# ── MARINA source root ────────────────────────────────────────────────────────
# Path to the directory that contains the MARINA `src/` package.
# The container mounts/copies this; the host path is also used in dev.
MARINA_ROOT: str = os.getenv("MARINA_ROOT", "/marina")

# ── Legacy dataset root (required by src/modules/core/const.py) ───────────────
# The backend does not use DATASET_ROOT directly (model data lives under MODEL_ROOT),
# but const.py imports it at module load time. Set it to DATA_DIR as a safe default.
# The env var is read directly by const.py; we set it here so the process inherits it.
if not os.environ.get("DATASET_ROOT"):
    os.environ.setdefault("DATASET_ROOT", DATA_DIR)

# ── Retrieval / prediction ────────────────────────────────────────────────────
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "10"))
MAX_TOP_K: int = int(os.getenv("MAX_TOP_K", "50"))
PREDICT_TIMEOUT_S: float = float(os.getenv("PREDICT_TIMEOUT_S", "60"))
SMILES_TIMEOUT_S: float = float(os.getenv("SMILES_TIMEOUT_S", "45"))

# ── Model loading ─────────────────────────────────────────────────────────────
# "default" → preload only the default model; "all" → preload every model;
# comma-separated list of model ids → preload those.
PRELOAD_MODELS: str = os.getenv("PRELOAD_MODELS", "default").strip()
# Maximum number of models to hold in memory simultaneously (0 = unlimited).
MAX_LOADED_MODELS: int = int(os.getenv("MAX_LOADED_MODELS", "0"))

# ── Compute pool ──────────────────────────────────────────────────────────────
# Worker processes that run heavy inference (predict) off the main event loop.
# Set to 0 to disable the pool entirely (dev mode): predictions run inline via
# asyncio.to_thread, which is simpler and compatible with uvicorn --reload.
MAX_COMPUTE_WORKERS: int = int(os.getenv("MAX_COMPUTE_WORKERS", "2"))
# Maximum number of jobs allowed to queue before returning 503.
MAX_COMPUTE_QUEUE: int = int(os.getenv("MAX_COMPUTE_QUEUE", "8"))

# ── Rendering ─────────────────────────────────────────────────────────────────
MOLECULE_IMG_SIZE: int = int(os.getenv("MOLECULE_IMG_SIZE", "400"))
RDKIT_ENABLED: bool = os.getenv("RDKIT_ENABLED", "true").lower() == "true"
# Similarity-map highlighting. Set to false to serve plain depictions only; the
# UI then hides its toggle.
HIGHLIGHT_ENABLED: bool = os.getenv("HIGHLIGHT_ENABLED", "true").lower() == "true"
# How the map's per-atom weights are computed: "attribution" projects each present
# substructure's predicted probability onto its atoms (one pass); "ablation" is the
# older SPECTRE-style leave-one-atom-out cosine difference (one re-score per atom),
# kept for comparison and as a revert path. See backend/README.md.
HIGHLIGHT_METHOD: str = os.getenv("HIGHLIGHT_METHOD", "attribution").lower()

# ── CORS ──────────────────────────────────────────────────────────────────────
# Comma-separated list of allowed origins. The app is normally served from the
# same origin as the API (via the nginx proxy), so nothing is needed by default.
# Use "*" only for a deliberately public, credential-free API.
CORS_ALLOW_ORIGINS: list = [
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()
]

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Format: "<count> per <second|minute|hour>". Set to "" to disable a limit.
# Enforced per client IP, per process — with UVICORN_WORKERS > 1 the effective
# limit is multiplied by the worker count.
RATE_LIMIT_PREDICT: str = os.getenv("RATE_LIMIT_PREDICT", "30 per minute")
RATE_LIMIT_SMILES: str = os.getenv("RATE_LIMIT_SMILES", "20 per minute")

# ── Usage stats ───────────────────────────────────────────────────────────────
# Where the query/visitor counters are persisted. Must be writable — DATA_DIR is
# mounted read-only in docker-compose, so this defaults elsewhere. Set to "" to
# keep counters in memory only (reset on every restart).
STATS_PATH: str = os.getenv("STATS_PATH", "/var/lib/marina/stats.json")

# ── Request size caps ─────────────────────────────────────────────────────────
# Upper bounds on spectral peak counts, mirrored in app.schemas. The frontend
# spreadsheet tops out at 400 rows; these guard the API against direct callers.
MAX_HSQC_PEAKS: int = int(os.getenv("MAX_HSQC_PEAKS", "2000"))
MAX_NMR_PEAKS: int = int(os.getenv("MAX_NMR_PEAKS", "2000"))
MAX_MS_PEAKS: int = int(os.getenv("MAX_MS_PEAKS", "5000"))
MAX_SMILES_LENGTH: int = int(os.getenv("MAX_SMILES_LENGTH", "1000"))
MAX_FP_LENGTH: int = int(os.getenv("MAX_FP_LENGTH", "65536"))
