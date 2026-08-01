"""
ComputePool: restart survival and queue-position reporting.

These spawn real worker processes, so they are marked slow. The workers here run
a stub loop rather than loading a model — the behaviour under test is the
request/response plumbing, not inference.
"""
import asyncio
import multiprocessing as mp
import os
import time

import pytest

import app.compute_pool as cp

pytestmark = pytest.mark.slow


def _stub_worker(req_q, res_q, marina_root, delay=0.0):
    """Echoes jobs back without touching a model."""
    from app.compute_pool import _Result
    while True:
        job = req_q.get()
        if job is None:
            break
        if delay:
            time.sleep(delay)
        res_q.put((job["job_id"], _Result(ok=True, value=job["op"])))


def _slow_worker(req_q, res_q, marina_root):
    _stub_worker(req_q, res_q, marina_root, delay=2.0)


@pytest.fixture
def stub_pool(monkeypatch):
    monkeypatch.setattr(cp, "_worker_loop", _stub_worker)
    pool = cp.ComputePool(max_workers=2, max_queue=16)
    yield pool
    asyncio.get_event_loop_policy()  # no-op; shutdown handled by the test


@pytest.fixture
def slow_pool(monkeypatch):
    monkeypatch.setattr(cp, "_worker_loop", _slow_worker)
    return cp.ComputePool(max_workers=2, max_queue=16)


async def test_a_job_round_trips(stub_pool):
    assert await stub_pool.run("ping", {}, timeout=30) == "ping"
    await stub_pool.shutdown()


async def test_pool_survives_a_restart(stub_pool):
    """
    Regression, and the most severe bug found in review: _restart_all used to
    reuse queues whose lock a terminated worker could still hold, so every
    replacement worker blocked forever and the service never recovered without
    a container restart.
    """
    assert await stub_pool.run("before", {}, timeout=30) == "before"

    await stub_pool._restart_all()

    assert await stub_pool.run("after", {}, timeout=30) == "after"
    assert await stub_pool.run("again", {}, timeout=30) == "again"
    assert sum(p.is_alive() for p in stub_pool._workers) == 2
    await stub_pool.shutdown()


async def test_restart_fails_in_flight_jobs_rather_than_hanging(slow_pool):
    task = asyncio.create_task(slow_pool.run("slow", {}, timeout=60))
    await asyncio.sleep(0.5)
    await slow_pool._restart_all()

    with pytest.raises(RuntimeError, match="restarted"):
        await asyncio.wait_for(task, timeout=15)
    await slow_pool.shutdown()


async def test_overload_is_reported_when_the_queue_is_full(monkeypatch):
    monkeypatch.setattr(cp, "_worker_loop", _slow_worker)
    pool = cp.ComputePool(max_workers=1, max_queue=2)
    tasks = [asyncio.create_task(pool.run("op", {}, timeout=60)) for _ in range(2)]
    await asyncio.sleep(0.3)

    with pytest.raises(cp.ComputeOverloadedError):
        await pool.run("overflow", {}, timeout=60)

    for t in tasks:
        t.cancel()
    await pool.shutdown()


async def test_queue_positions_reflect_fifo_order(slow_pool):
    """
    Workers pull FIFO, so the oldest `workers` in-flight jobs are the running
    ones and everything behind them is queued.
    """
    ids = [f"req-{i}" for i in range(6)]
    tasks = [asyncio.create_task(slow_pool.run("op", {}, timeout=90, request_id=r))
             for r in ids]
    await asyncio.sleep(0.6)

    snap = await slow_pool.snapshot()
    assert snap["workers"] == 2
    assert snap["running"] == 2
    assert snap["queued"] == 4

    states = [await slow_pool.position_of(r) for r in ids]
    assert [s["state"] for s in states] == ["running", "running"] + ["queued"] * 4
    assert [s["position"] for s in states] == [0, 0, 1, 2, 3, 4]
    assert [s["ahead"] for s in states] == [0, 1, 2, 3, 4, 5]

    await asyncio.gather(*tasks)
    await slow_pool.shutdown()


async def test_positions_advance_as_jobs_complete(slow_pool):
    ids = [f"req-{i}" for i in range(4)]
    tasks = [asyncio.create_task(slow_pool.run("op", {}, timeout=90, request_id=r))
             for r in ids]
    await asyncio.sleep(0.6)
    assert (await slow_pool.position_of("req-3"))["position"] == 2

    await asyncio.sleep(2.4)          # first pair completes
    later = await slow_pool.position_of("req-3")
    assert later is None or later["position"] < 2

    await asyncio.gather(*tasks)
    await slow_pool.shutdown()


async def test_entries_are_released_when_jobs_finish(stub_pool):
    await asyncio.gather(*[
        stub_pool.run("op", {}, timeout=30, request_id=f"r{i}") for i in range(5)
    ])
    assert stub_pool._entries == {}
    assert await stub_pool.position_of("r0") is None
    await stub_pool.shutdown()


async def test_position_of_an_unknown_request_is_none(stub_pool):
    assert await stub_pool.position_of("never-submitted") is None
    await stub_pool.shutdown()


async def test_snapshot_on_an_idle_pool(stub_pool):
    snap = await stub_pool.snapshot()
    assert snap["in_flight"] == 0 and snap["queued"] == 0
    assert snap["workers"] == 2
    await stub_pool.shutdown()
