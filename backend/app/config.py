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
# Optional cap on simultaneous heavy requests (0 = no cap).
MAX_CONCURRENT_HEAVY_OPS: int = int(os.getenv("MAX_CONCURRENT_HEAVY_OPS", "0"))

# ── Rendering ─────────────────────────────────────────────────────────────────
MOLECULE_IMG_SIZE: int = int(os.getenv("MOLECULE_IMG_SIZE", "400"))
RDKIT_ENABLED: bool = os.getenv("RDKIT_ENABLED", "true").lower() == "true"

# ── Rate limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_PREDICT: str = os.getenv("RATE_LIMIT_PREDICT", "30 per minute")
RATE_LIMIT_SMILES: str = os.getenv("RATE_LIMIT_SMILES", "20 per minute")
