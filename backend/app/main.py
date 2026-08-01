"""
FastAPI application entry point.

Startup sequence:
  1. Bootstrap MARINA sys.path (marina_import)
  2. Launch compute worker pool
  3. Preload configured models into the registry

All heavy configuration is read from environment variables via app.config.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import Request, status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Application lifespan ──────────────────────────────────────────────────────
#
# Worker processes are cleaned up by the lifespan shutdown below. Installing
# SIGINT/SIGTERM handlers here used to short-circuit that with os._exit(0),
# which skipped shutdown_pool() entirely and fought uvicorn's own supervisor
# when running with --workers.

@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.config import (
        MAX_COMPUTE_WORKERS, MAX_COMPUTE_QUEUE,
        PRELOAD_MODELS, MARINA_ROOT,
    )
    from app.compute_pool import get_pool, shutdown_pool
    from app.manifest import load_models_json, list_models, get_default_model_id
    from app.registry import ensure_loaded, pin

    # Ensure MARINA src is importable before anything else
    from app.marina_import import ensure_marina_importable
    ensure_marina_importable()

    logger.info("Starting MARINA backend (MARINA_ROOT=%s)", MARINA_ROOT)

    # Attach compute pool to app state
    pool = get_pool(MAX_COMPUTE_WORKERS, MAX_COMPUTE_QUEUE)
    app.state.compute_pool = pool

    # Preload models
    load_models_json()
    entries    = list_models()
    default_id = get_default_model_id()

    to_preload = _resolve_preload(PRELOAD_MODELS, entries, default_id)

    for entry in to_preload:
        try:
            logger.info("Preloading model %s (type=%s)…", entry.id, entry.type)
            await asyncio.to_thread(ensure_loaded, entry.id)
            pin(entry.id)
            logger.info("Model %s ready", entry.id)
        except Exception as exc:
            logger.error("Failed to preload model %s: %s", entry.id, exc, exc_info=True)
            raise

    logger.info("MARINA backend ready")
    yield

    logger.info("Shutting down MARINA backend…")
    from app.stats import get_stats
    get_stats().flush()
    await shutdown_pool()
    for child in mp.active_children():
        child.terminate()
        child.join(timeout=1.0)


def _resolve_preload(spec: str, entries, default_id: str):
    if not entries:
        return []
    spec = spec.lower()
    if spec in ("", "default"):
        return [e for e in entries if e.id == default_id] or [entries[0]]
    if spec == "all":
        return entries
    want = {s.strip() for s in spec.split(",") if s.strip()}
    return [e for e in entries if e.id in want]


# ── Create application ────────────────────────────────────────────────────────

app = FastAPI(
    title="MARINA API",
    description="Natural products structure annotation via spectral data",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS. Credentials are off because the API uses none: pairing them with a
# wildcard origin makes Starlette echo back whatever Origin it is given, which
# is an open door for credentialed cross-origin requests.
from app.config import CORS_ALLOW_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Per-client throttling for the expensive endpoints.
from app.rate_limit import RateLimitMiddleware

app.add_middleware(RateLimitMiddleware)

# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"error": "Validation error", "details": exc.errors()}),
    )

# General error handler
@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"},
    )

# ── Routers ───────────────────────────────────────────────────────────────────

from app.routes import (
    health, models, predict, smiles_search, fingerprints, custom_smiles, status as status_routes,
)

app.include_router(health.router,        prefix="/api", tags=["health"])
app.include_router(status_routes.router, prefix="/api", tags=["status"])
app.include_router(models.router,        prefix="/api", tags=["models"])
app.include_router(predict.router,       prefix="/api", tags=["prediction"])
app.include_router(smiles_search.router, prefix="/api", tags=["search"])
app.include_router(fingerprints.router,  prefix="/api", tags=["fingerprints"])
app.include_router(custom_smiles.router, prefix="/api", tags=["custom"])


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from app.config import HOST, PORT

    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False, log_level="info")
