"""
Worker process pool for CPU/GPU-bound inference tasks.

Each worker process loads the default model once at startup, then handles
serialised requests from the main process.  Results are routed back via a
shared response queue and delivered to the awaiting asyncio Future.

Supported operations (op names):
    "predict"  – run predict_from_raw and return (scores, global_idxs, pred_fp)
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import signal
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ComputeOverloadedError(RuntimeError):
    """Raised when the compute queue is full."""


class ComputeTimeoutError(TimeoutError):
    """Raised when a compute job exceeds its timeout."""


@dataclass
class _Result:
    ok:    bool
    value: Any   = None
    error: Optional[str] = None
    trace: Optional[str] = None


# ── Worker entry point ────────────────────────────────────────────────────────

def _worker_loop(req_q: mp.Queue, res_q: mp.Queue, marina_root: str) -> None:
    """
    Worker process main loop.  Loads the model once, then dispatches jobs.
    marina_root is passed explicitly so the worker can bootstrap sys.path
    without depending on environment variables being inherited correctly.
    """
    import sys
    if marina_root not in sys.path:
        sys.path.insert(0, marina_root)

    # Bootstrap MARINA imports before anything else
    from app.marina_import import ensure_marina_importable
    ensure_marina_importable()

    from app.registry import ensure_loaded
    from app.manifest import get_default_model_id

    # Preload default model in this worker
    try:
        default_id = get_default_model_id()
        ensure_loaded(default_id)
        logger.info("Worker %d: preloaded model %s", os.getpid(), default_id)
    except Exception as exc:
        logger.error("Worker %d: failed to preload model: %s", os.getpid(), exc)

    while True:
        job = req_q.get()
        if job is None:                # sentinel → shutdown
            break
        job_id  = job["job_id"]
        op      = job["op"]
        payload = job["payload"]
        try:
            if op == "predict":
                from app.predictor import predict_from_raw
                value = predict_from_raw(
                    raw_inputs=payload["raw_inputs"],
                    k=payload["k"],
                    model_id=payload.get("model_id"),
                    mw_min=payload.get("mw_min"),
                    mw_max=payload.get("mw_max"),
                )
            else:
                raise ValueError(f"Unknown op {op!r}")
            res_q.put((job_id, _Result(ok=True, value=value)))
        except Exception as exc:
            res_q.put((
                job_id,
                _Result(ok=False, error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()),
            ))


# ── Pool ──────────────────────────────────────────────────────────────────────

class ComputePool:
    """
    Persistent pool of worker processes.  Provides a single async `run` coroutine
    that submits a job and waits for the result, with an optional timeout.
    """

    def __init__(self, max_workers: int, max_queue: int = 0) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be ≥ 1")

        from app.config import MARINA_ROOT

        self._marina_root = MARINA_ROOT
        self._max_queue   = max_queue
        self._pending     = 0
        self._pending_lock = asyncio.Lock()

        ctx = mp.get_context("spawn")
        self._req_q:  mp.Queue = ctx.Queue()
        self._res_q:  mp.Queue = ctx.Queue()
        self._workers: list[mp.Process] = []
        self._jobs:    dict[str, asyncio.Future] = {}
        self._jobs_lock = asyncio.Lock()
        self._response_task: Optional[asyncio.Task] = None
        self._restart_lock = asyncio.Lock()

        self._spawn(ctx, max_workers)

    def _spawn(self, ctx, count: int) -> None:
        for _ in range(count):
            p = ctx.Process(
                target=_worker_loop,
                args=(self._req_q, self._res_q, self._marina_root),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

    async def _ensure_response_task(self) -> None:
        if self._response_task is None or self._response_task.done():
            self._response_task = asyncio.create_task(self._response_loop())

    async def _response_loop(self) -> None:
        while True:
            try:
                job_id, result = await asyncio.to_thread(self._res_q.get)
            except Exception:
                continue
            async with self._jobs_lock:
                future = self._jobs.pop(job_id, None)
            if future is None or future.done():
                continue
            if result.ok:
                future.set_result(result.value)
            else:
                logger.error("Worker error: %s\n%s", result.error, result.trace or "")
                future.set_exception(RuntimeError(result.error or "Worker failed"))

    async def _acquire_slot(self) -> None:
        async with self._pending_lock:
            cap = self._max_queue if self._max_queue > 0 else len(self._workers)
            if self._pending >= cap:
                raise ComputeOverloadedError(
                    "Compute queue is full" if self._max_queue > 0 else "All workers are busy"
                )
            self._pending += 1

    async def _release_slot(self) -> None:
        async with self._pending_lock:
            self._pending = max(0, self._pending - 1)

    async def _restart_all(self) -> None:
        async with self._restart_lock:
            import signal as _sig
            for _ in self._workers:
                self._req_q.put(None)
            for p in self._workers:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1.0)
                if p.is_alive() and p.pid:
                    os.kill(p.pid, _sig.SIGKILL)
            ctx = mp.get_context("spawn")
            count = len(self._workers)
            self._workers = []
            self._spawn(ctx, count)

    async def run(self, op: str, payload: dict, timeout: Optional[float] = None) -> Any:
        """Submit a job to the pool and await its result."""
        await self._ensure_response_task()
        await self._acquire_slot()

        job_id = f"{time.monotonic_ns()}-{os.getpid()}-{id(payload)}"
        loop   = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        async with self._jobs_lock:
            self._jobs[job_id] = future

        self._req_q.put({"job_id": job_id, "op": op, "payload": payload})

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            async with self._jobs_lock:
                self._jobs.pop(job_id, None)
            await self._restart_all()
            raise ComputeTimeoutError("Compute job timed out") from exc
        finally:
            await self._release_slot()

    async def shutdown(self) -> None:
        for _ in self._workers:
            self._req_q.put(None)
        for p in self._workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()


# ── Singleton helpers ─────────────────────────────────────────────────────────

_pool: Optional[ComputePool] = None
_pool_lock = threading.Lock()


def get_pool(max_workers: int, max_queue: int = 0) -> Optional[ComputePool]:
    """
    Return a ComputePool singleton, or None when max_workers == 0.
    Passing max_workers=0 disables the pool entirely; predict routes will run
    inference inline via asyncio.to_thread (useful for dev / single-process mode).
    """
    if max_workers == 0:
        return None
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ComputePool(max_workers=max_workers, max_queue=max_queue)
        return _pool


async def shutdown_pool() -> None:
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        await pool.shutdown()
