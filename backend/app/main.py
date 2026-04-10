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
import os
import signal
import time

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


# ── Signal handling (clean shutdown of worker processes) ──────────────────────

def _kill_children(signum, frame) -> None:  # pragma: no cover
    logger.warning("Received signal %s – killing child processes", signum)
    for child in mp.active_children():
        try:
            child.terminate()
            child.join(timeout=0.5)
        except Exception:
            pass
        if child.is_alive() and child.pid:
            os.kill(child.pid, signal.SIGKILL)
    os._exit(0)


signal.signal(signal.SIGINT,  _kill_children)
signal.signal(signal.SIGTERM, _kill_children)


# ── Application lifespan ──────────────────────────────────────────────────────

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
    await shutdown_pool()


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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

from app.routes import health, models, predict, smiles_search, fingerprints

app.include_router(health.router,       prefix="/api", tags=["health"])
app.include_router(models.router,       prefix="/api", tags=["models"])
app.include_router(predict.router,      prefix="/api", tags=["prediction"])
app.include_router(smiles_search.router, prefix="/api", tags=["search"])
app.include_router(fingerprints.router, prefix="/api", tags=["fingerprints"])


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from app.config import HOST, PORT

    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False, log_level="info")
