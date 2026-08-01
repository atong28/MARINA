"""
Per-client rate limiting for the expensive endpoints.

A fixed-window counter keyed by (client IP, endpoint group). State is in-process,
so with UVICORN_WORKERS > 1 the effective limit is the configured value times
the worker count — good enough to stop a single client monopolising inference,
not a substitute for a shared limiter if you need exact global caps.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Dict, Optional, Tuple

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

_PERIODS = {"second": 1.0, "minute": 60.0, "hour": 3600.0}
_SPEC_RE = re.compile(r"^\s*(\d+)\s*per\s*(second|minute|hour)\s*$", re.IGNORECASE)


def parse_limit(spec: str) -> Optional[Tuple[int, float]]:
    """Parse "30 per minute" into (30, 60.0). Returns None if unset/malformed."""
    if not spec or not spec.strip():
        return None
    match = _SPEC_RE.match(spec)
    if not match:
        logger.warning("Ignoring malformed rate limit spec %r", spec)
        return None
    return int(match.group(1)), _PERIODS[match.group(2).lower()]


def client_ip(request) -> str:
    """
    Best-effort client address.

    X-Forwarded-For's first entry is the original client, provided the edge
    proxy overwrites the header rather than appending to a client-supplied one
    (see nginx/nginx*.conf, which set it to $remote_addr).
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


class _FixedWindow:
    """Counts hits per key within a window, discarding expired windows."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: Dict[str, Tuple[float, int]] = {}

    def allow(self, key: str, limit: int, period: float, now: float) -> Tuple[bool, float]:
        with self._lock:
            window_start, count = self._hits.get(key, (now, 0))
            if now - window_start >= period:
                window_start, count = now, 0
            if count >= limit:
                return False, period - (now - window_start)
            self._hits[key] = (window_start, count + 1)

            # Opportunistic sweep so idle clients do not accumulate forever.
            if len(self._hits) > 4096:
                self._hits = {
                    k: v for k, v in self._hits.items() if now - v[0] < period
                }
            return True, 0.0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Throttles the inference endpoints; everything else passes through."""

    def __init__(self, app) -> None:
        super().__init__(app)
        from app.config import RATE_LIMIT_PREDICT, RATE_LIMIT_SMILES

        predict_limit = parse_limit(RATE_LIMIT_PREDICT)
        smiles_limit  = parse_limit(RATE_LIMIT_SMILES)

        # Longest-prefix wins, so /api/predict does not also match a broader rule.
        self._rules = [
            ("/api/predict", predict_limit),
            ("/api/smiles-search", smiles_limit),
            ("/api/custom-smiles-card", smiles_limit),
            ("/api/fingerprints/", smiles_limit),
        ]
        self._window = _FixedWindow()

    def _rule_for(self, path: str):
        for prefix, limit in self._rules:
            if limit and path.startswith(prefix):
                return prefix, limit
        return None, None

    async def dispatch(self, request, call_next):
        prefix, limit = self._rule_for(request.url.path)
        if limit is None:
            return await call_next(request)

        count, period = limit
        key = f"{client_ip(request)}|{prefix}"
        allowed, retry_after = self._window.allow(key, count, period, time.monotonic())
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down."},
                headers={"Retry-After": str(max(1, int(retry_after)))},
            )
        return await call_next(request)
