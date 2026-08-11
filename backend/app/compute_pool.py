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
import itertools
import logging
import multiprocessing as mp
import os
import queue
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


@dataclass
class _QueueEntry:
    """Bookkeeping for one in-flight job, used to answer "where am I in line?"."""
    job_id:       str
    seq:          int
    request_id:   Optional[str]
    submitted_at: float


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

    # Spawned workers do not inherit the parent's logging config (app.main is
    # never imported here), so without this their logs are silently dropped.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s [worker] %(message)s",
    )

    # Bootstrap MARINA imports before anything else
    from app.marina_import import ensure_marina_importable
    ensure_marina_importable()

    from app.registry import ensure_loaded
    from app.manifest import get_default_model_id
    from app.session import disable_annotations

    # This process answers "predict" only; result cards are built in the API
    # process, so the annotation table would be ~270 MB of dead weight here.
    disable_annotations()

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
        self._entries: dict[str, _QueueEntry]    = {}
        self._jobs_lock = asyncio.Lock()
        self._response_task: Optional[asyncio.Task] = None
        self._restart_lock = asyncio.Lock()
        self._job_counter = itertools.count()

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
            self._response_task = asyncio.create_task(self._response_loop(self._res_q))

    async def _response_loop(self, res_q: "mp.Queue") -> None:
        """
        Drain results for one generation of the pool.

        The queue is bound at task creation: after a restart swaps in a fresh
        queue this loop exits instead of blocking forever on the dead one. The
        1 s poll is what lets it notice — a bare get() would park a thread that
        never returns.
        """
        while res_q is self._res_q:
            try:
                job_id, result = await asyncio.to_thread(res_q.get, True, 1.0)
            except queue.Empty:
                continue
            except (OSError, ValueError, EOFError) as exc:
                logger.warning("Response queue closed: %s", exc)
                return
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
        """
        Replace every worker after a timeout.

        The queues are rebuilt rather than reused. Terminating a process that is
        blocked in Queue.get() can leave the queue's internal lock held, and the
        queue then blocks every later reader forever — the replacement workers
        would come up dead. Any sentinel left over from the old generation would
        likewise be consumed by a fresh worker and shut it down immediately.
        """
        async with self._restart_lock:
            old_workers, self._workers = self._workers, []
            for p in old_workers:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1.0)
                if p.is_alive() and p.pid:
                    os.kill(p.pid, signal.SIGKILL)

            # The jobs those workers were running died with them.
            async with self._jobs_lock:
                orphaned, self._jobs = self._jobs, {}
                self._entries = {}
            for fut in orphaned.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("Compute worker pool was restarted"))

            ctx = mp.get_context("spawn")
            self._req_q = ctx.Queue()
            self._res_q = ctx.Queue()
            self._spawn(ctx, len(old_workers))

        # The old reader is bound to the old queue; it will exit on its own.
        self._response_task = None
        await self._ensure_response_task()

    async def run(
        self,
        op: str,
        payload: dict,
        timeout: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> Any:
        """
        Submit a job to the pool and await its result.

        request_id is a caller-supplied handle used only for queue reporting —
        pass the one the client generated so it can poll its own position while
        this request is still open.
        """
        await self._ensure_response_task()
        await self._acquire_slot()

        seq    = next(self._job_counter)
        job_id = f"{time.monotonic_ns()}-{os.getpid()}-{seq}"
        loop   = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        async with self._jobs_lock:
            self._jobs[job_id] = future
            self._entries[job_id] = _QueueEntry(
                job_id=job_id, seq=seq, request_id=request_id,
                submitted_at=time.monotonic(),
            )

        # Held so a concurrent _restart_all cannot swap the queue mid-submit.
        async with self._restart_lock:
            self._req_q.put({"job_id": job_id, "op": op, "payload": payload})

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            async with self._jobs_lock:
                self._jobs.pop(job_id, None)
            await self._restart_all()
            raise ComputeTimeoutError("Compute job timed out") from exc
        finally:
            async with self._jobs_lock:
                self._entries.pop(job_id, None)
            await self._release_slot()

    # ── Queue introspection ───────────────────────────────────────────────────

    def _ordered_entries(self) -> list:
        return sorted(self._entries.values(), key=lambda e: e.seq)

    async def snapshot(self) -> dict:
        """Pool-wide queue state, safe to poll frequently."""
        async with self._jobs_lock:
            entries = self._ordered_entries()
        workers = len(self._workers)
        in_flight = len(entries)
        running = min(in_flight, workers)
        return {
            "workers":  workers,
            "capacity": self._max_queue if self._max_queue > 0 else workers,
            "running":  running,
            "queued":   max(0, in_flight - running),
            "in_flight": in_flight,
        }

    async def position_of(self, request_id: str) -> Optional[dict]:
        """
        Where a caller's job sits, or None once it is no longer in flight.

        Workers pull from a FIFO queue, so job k is only picked up after every
        earlier job has been. The oldest `workers` in-flight jobs are therefore
        exactly the ones executing, and anything behind them is still waiting.
        """
        async with self._jobs_lock:
            entries = self._ordered_entries()
        workers = len(self._workers)
        for idx, entry in enumerate(entries):
            if entry.request_id == request_id:
                waiting = idx >= workers
                return {
                    "state":     "queued" if waiting else "running",
                    "position":  idx - workers + 1 if waiting else 0,
                    "ahead":     idx,
                    "queued":    max(0, len(entries) - workers),
                    "workers":   workers,
                    "waited_seconds": round(time.monotonic() - entry.submitted_at, 1),
                }
        return None

    async def shutdown(self) -> None:
        # Sentinels first so workers idle in get() can exit cleanly; only
        # terminate the ones that do not take the hint.
        for _ in self._workers:
            self._req_q.put(None)
        for p in self._workers:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        self._workers = []
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass


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
