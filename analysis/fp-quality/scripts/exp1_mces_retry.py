"""Exp 1 (retry) — recompute the NaN-MCES pairs from exp1_pairs.parquet.

The original run used returnEmptyMCES=False, so RASCAL returned an empty result — logged as
NaN — for BOTH (a) genuinely low-similarity pairs below the screen / under minFragSize, and
(b) true timeouts. This retry uses returnEmptyMCES=True to force a computed similarity for
every pair and reads `.timedOut` to distinguish the two:
  - timedOut=False  -> reliable similarity (recovers the wrongly-dropped dissimilar pairs, ~0)
  - timedOut=True   -> still intractable at the longer timeout; left NaN and counted

Writes an updated parquet (mces filled for recovered rows; a `retry_timedout` flag column).
Runs in MARINA's env (torch not needed; rdkit + pandas + pyarrow).

Usage: python exp1_mces_retry.py --retrieval <retrieval.pkl> --pairs <in.parquet>
         --out <out.parquet> [--timeout 300] [--workers 32]
"""
import argparse, os, pickle, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

_SMILES = None
_TIMEOUT = 300


def _init(smiles, timeout):
    global _SMILES, _TIMEOUT
    _SMILES, _TIMEOUT = smiles, timeout


def _mces(ij):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdRascalMCES
    RDLogger.DisableLog("rdApp.*")
    i, j = ij
    m1 = Chem.MolFromSmiles(_SMILES[i])
    m2 = Chem.MolFromSmiles(_SMILES[j])
    if m1 is None or m2 is None:
        return (i, j, np.nan, False)  # parse failure, not a timeout
    opts = rdRascalMCES.RascalOptions()
    opts.similarityThreshold = 0.05
    opts.minFragSize = 3
    opts.returnEmptyMCES = True
    opts.timeout = _TIMEOUT
    try:
        r = rdRascalMCES.FindMCES(m1, m2, opts)
        if not r:
            return (i, j, np.nan, False)
        res = r[0]
        return (i, j, float(res.similarity), bool(res.timedOut))
    except Exception:
        return (i, j, np.nan, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    df = pd.read_parquet(args.pairs)
    with open(args.retrieval, "rb") as f:
        R = pickle.load(f)
    n_all = len(R)
    smiles = [(R[k]["smiles"] if isinstance(R[k], dict) else R[k]) for k in range(n_all)]

    nan_mask = ~np.isfinite(df["mces"].to_numpy())
    todo = [(int(r.i), int(r.j)) for r in df[nan_mask].itertuples(index=False)]
    print(f"{len(df)} pairs total; {len(todo)} NaN pairs to recompute "
          f"(timeout={args.timeout}s, workers={args.workers})", flush=True)

    sim = {}
    timed = {}
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(smiles, args.timeout)) as pool:
        for k, (i, j, s, to) in enumerate(pool.imap_unordered(_mces, todo, chunksize=32)):
            sim[(i, j)] = s
            timed[(i, j)] = to
            if k % 2000 == 0:
                rec = sum(1 for v in timed.values() if not v)
                print(f"  {k}/{len(todo)}  {time.time()-t0:.0f}s  recovered≈{rec}", flush=True)

    mces = df["mces"].to_numpy().copy()
    retry_to = np.zeros(len(df), dtype=bool)
    idx_of = {(int(r.i), int(r.j)): n for n, r in enumerate(df.itertuples(index=False))}
    recovered = still_timed = parse_fail = 0
    for (i, j), s in sim.items():
        row = idx_of[(i, j)]
        if timed[(i, j)]:
            still_timed += 1
            retry_to[row] = True  # leave mces NaN
        elif np.isnan(s):
            parse_fail += 1
        else:
            mces[row] = s
            recovered += 1
    df["mces"] = mces
    df["retry_timedout"] = retry_to

    n_valid = int(np.isfinite(df["mces"].to_numpy()).sum())
    print(f"\nrecovered {recovered} pairs; still timed out {still_timed}; parse-fail {parse_fail}", flush=True)
    print(f"valid MCES pairs: {n_valid}/{len(df)} (was {len(df)-len(todo)})", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
