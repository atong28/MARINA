"""
Usage counters: how many queries have been served, and by how many distinct
clients.

Clients are identified by a salted hash of their IP address, never the address
itself — the salt is generated once and stored alongside the counters, so the
same visitor stays the same anonymous id across restarts but the set cannot be
walked back to a list of addresses.

State is per-process. With UVICORN_WORKERS=1 (the default) that is the whole
picture; raising it splits counts across workers.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import tempfile
import threading
import time
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

# Distinct client hashes retained. Past this the unique count keeps rising but
# becomes approximate, so memory stays bounded on a long-lived deployment.
MAX_TRACKED_CLIENTS = 200_000

_QUERY_KINDS = ("predict", "smiles_search", "custom_card")


class UsageStats:
    def __init__(self, path: Optional[str], flush_interval: float = 30.0) -> None:
        self._path = path
        self._flush_interval = flush_interval
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {k: 0 for k in _QUERY_KINDS}
        self._clients: Set[str] = set()
        self._overflow_clients = 0
        self._salt = secrets.token_hex(16)
        self._started_at = time.time()
        self._last_flush = 0.0
        self._dirty = False
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path or not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r") as fh:
                data = json.load(fh)
            self._counts.update({
                k: int(v) for k, v in (data.get("counts") or {}).items() if k in self._counts
            })
            self._clients = set(data.get("clients") or [])
            self._overflow_clients = int(data.get("overflow_clients") or 0)
            self._salt = data.get("salt") or self._salt
            self._started_at = float(data.get("started_at") or self._started_at)
            logger.info(
                "Loaded usage stats from %s (%d queries, %d clients)",
                self._path, sum(self._counts.values()), len(self._clients),
            )
        except Exception as exc:
            logger.warning("Could not read usage stats from %s: %s", self._path, exc)

    def _flush_locked(self) -> None:
        if not self._path:
            return
        payload = {
            "counts":            dict(self._counts),
            "clients":           sorted(self._clients),
            "overflow_clients":  self._overflow_clients,
            "salt":              self._salt,
            "started_at":        self._started_at,
        }
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            # Write-then-rename so a crash mid-write cannot truncate the file.
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self._path) or ".")
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self._path)
            self._dirty = False
        except Exception as exc:
            # A read-only mount is a normal deployment, not an error worth
            # failing a request over: counters simply stay in memory.
            logger.debug("Could not persist usage stats to %s: %s", self._path, exc)
            self._path = None

    def flush(self) -> None:
        with self._lock:
            if self._dirty:
                self._flush_locked()

    # ── Recording ────────────────────────────────────────────────────────────

    def record(self, kind: str, client: Optional[str]) -> None:
        if kind not in self._counts:
            return
        now = time.monotonic()
        with self._lock:
            self._counts[kind] += 1
            if client:
                digest = hashlib.sha256(f"{self._salt}:{client}".encode()).hexdigest()[:16]
                if digest not in self._clients:
                    if len(self._clients) < MAX_TRACKED_CLIENTS:
                        self._clients.add(digest)
                    else:
                        self._overflow_clients += 1
            self._dirty = True
            if now - self._last_flush >= self._flush_interval:
                self._last_flush = now
                self._flush_locked()

    # ── Reporting ────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            counts = dict(self._counts)
            unique = len(self._clients) + self._overflow_clients
            started = self._started_at
        return {
            "queries_total":  sum(counts.values()),
            "by_kind":        counts,
            "unique_clients": unique,
            "counting_since": started,
        }


_stats: Optional[UsageStats] = None
_init_lock = threading.Lock()


def get_stats() -> UsageStats:
    global _stats
    if _stats is None:
        with _init_lock:
            if _stats is None:
                from app.config import STATS_PATH
                _stats = UsageStats(STATS_PATH or None)
    return _stats


def record_query(kind: str, request) -> None:
    """Record one served query. Never raises — stats must not break a request."""
    try:
        from app.rate_limit import client_ip
        get_stats().record(kind, client_ip(request))
    except Exception as exc:
        logger.debug("stats.record_query failed: %s", exc)
