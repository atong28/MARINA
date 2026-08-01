"""
Queue depth, per-request queue position, and usage counters.

These are polled by the browser while a prediction is in flight, so they must
stay cheap: no model access, no disk reads, no locks held across I/O.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request, status

logger = logging.getLogger(__name__)
router = APIRouter()


def _idle_snapshot() -> dict:
    """What to report when no worker pool is configured (dev / inline mode)."""
    return {"workers": 1, "capacity": 1, "running": 0, "queued": 0, "in_flight": 0}


@router.get("/queue", status_code=status.HTTP_200_OK)
async def queue_status(request: Request):
    """Pool-wide queue depth."""
    pool = getattr(request.app.state, "compute_pool", None)
    if pool is None:
        return {"pooled": False, **_idle_snapshot()}
    return {"pooled": True, **(await pool.snapshot())}


@router.get("/queue/{request_id}", status_code=status.HTTP_200_OK)
async def queue_position(request_id: str, request: Request):
    """
    Where the caller's in-flight prediction sits in line.

    state is "queued" while waiting behind others, "running" once a worker has
    picked it up, and "unknown" once it is no longer in flight — which normally
    means it finished and the POST is about to return.
    """
    pool = getattr(request.app.state, "compute_pool", None)
    if pool is None:
        return {"state": "running", "position": 0, "ahead": 0, **_idle_snapshot()}

    found = await pool.position_of(request_id)
    if found is None:
        snap = await pool.snapshot()
        return {"state": "unknown", "position": 0, "ahead": 0, **snap}
    return found


@router.get("/stats", status_code=status.HTTP_200_OK)
async def usage_stats():
    """Cumulative query and visitor counts."""
    from app.stats import get_stats
    return get_stats().snapshot()
