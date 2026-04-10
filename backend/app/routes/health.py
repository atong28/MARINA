"""Health check endpoints."""
import time

from fastapi import APIRouter, status

router = APIRouter()
_start_time = time.time()


@router.get("/health/live", status_code=status.HTTP_200_OK, include_in_schema=False)
async def liveness():
    """Kubernetes liveness probe – always 200 while the process is alive."""
    return {"status": "alive"}


@router.get("/health", status_code=status.HTTP_200_OK)
async def health():
    """Readiness check that reports whether the default model is loaded."""
    import asyncio
    from app.registry import is_loaded
    from app.manifest import get_default_model_id

    default_id = get_default_model_id()
    ready = await asyncio.to_thread(is_loaded, default_id)

    return {
        "status":         "ok" if ready else "initializing",
        "model_loaded":   ready,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }
