"""
Classify every MARINA1 retrieval molecule with NPClassifier (npclassifier.gnps2.org).

    cd /home/user/atong/MARINA
    pixi run python3 analysis/np-classifier/scripts/01_fetch.py [--concurrency 32] [--limit N]

Append-only and resumable: every answered index is written to results/npclassifier.jsonl
as it arrives, and a re-run skips whatever is already in that file. Killing the job at any
point costs at most the in-flight requests.

Read-only on Datasets/. The retrieval md5 is written to results/provenance.json so the
index space these records are keyed on can be verified before anything trusts them.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pickle
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "results")
JSONL = os.path.join(OUT_DIR, "npclassifier.jsonl")
PROVENANCE = os.path.join(OUT_DIR, "provenance.json")

RETRIEVAL = os.path.join(
    os.environ.get("MARINA_DATA_ROOT", "/home/user/atong"),
    "Datasets", "MARINA1", "retrieval.pkl",
)
URL = "https://npclassifier.gnps2.org/classify"
# Identifies the traffic to the GNPS2 operators, so a 519k-request pass is attributable
# rather than anonymous.
UA = "MARINA-NPClassifier-annotation/1.0 (UCSD; atong28.usa@gmail.com)"

# A 500 from this service is how it reports a structure it cannot parse, so it is not
# retried indefinitely -- but transient 5xx look identical, hence a few attempts first.
MAX_ATTEMPTS = 4


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_done() -> set:
    """Indices already answered. Truncated final line (kill mid-write) is dropped."""
    done = set()
    if not os.path.exists(JSONL):
        return done
    with open(JSONL, "r") as f:
        for line in f:
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                continue
    return done


async def classify(client: httpx.AsyncClient, idx: int, smiles: str) -> dict:
    last = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = await client.get(URL, params={"smiles": smiles})
            if r.status_code == 200:
                d = r.json()
                return {
                    "idx": idx,
                    "smiles": smiles,
                    "class_results": d.get("class_results") or [],
                    "superclass_results": d.get("superclass_results") or [],
                    "pathway_results": d.get("pathway_results") or [],
                    "isglycoside": bool(d.get("isglycoside", False)),
                }
            last = f"HTTP {r.status_code}"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        if attempt < MAX_ATTEMPTS - 1:
            await asyncio.sleep(2 ** attempt)
    return {"idx": idx, "smiles": smiles, "error": last}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N pending (smoke test).")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(RETRIEVAL, "rb") as f:
        retrieval = pickle.load(f)
    todo = [(int(k), v["smiles"]) for k, v in retrieval.items()]
    todo.sort()
    total = len(todo)

    done = load_done()
    todo = [t for t in todo if t[0] not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"retrieval={total}  already done={len(done)}  this run={len(todo)}", flush=True)
    if not todo:
        print("nothing to do")
        return 0

    if not os.path.exists(PROVENANCE):
        with open(PROVENANCE, "w") as f:
            json.dump({
                "retrieval_path": RETRIEVAL,
                "retrieval_md5": md5(RETRIEVAL),
                "retrieval_n": total,
                "url": URL,
            }, f, indent=2)

    sem = asyncio.Semaphore(args.concurrency)
    out = open(JSONL, "a", buffering=1)          # line-buffered: crash-safe to the line
    lock = asyncio.Lock()
    t0 = time.time()
    n_done = 0
    n_err = 0

    limits = httpx.Limits(max_connections=args.concurrency,
                          max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits,
                                 headers={"User-Agent": UA}) as client:

        async def worker(idx: int, smiles: str):
            nonlocal n_done, n_err
            async with sem:
                rec = await classify(client, idx, smiles)
            async with lock:
                out.write(json.dumps(rec) + "\n")
                n_done += 1
                if "error" in rec:
                    n_err += 1
                if n_done % 5000 == 0:
                    el = time.time() - t0
                    rate = n_done / el
                    eta = (len(todo) - n_done) / rate / 60
                    print(f"{n_done}/{len(todo)}  {rate:.0f}/s  err={n_err}  "
                          f"eta {eta:.0f}min", flush=True)

        await asyncio.gather(*(worker(i, s) for i, s in todo))

    out.close()
    el = time.time() - t0
    print(f"\ndone {n_done} in {el/60:.1f}min ({n_done/el:.0f}/s), {n_err} errors", flush=True)
    # Errors are left for a re-run: they are absent from the JSONL only if they were
    # never attempted, so `load_done` treats a recorded error as done. 02_export reports
    # them; re-running after deleting those lines retries just those.
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
