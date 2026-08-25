"""Exp 1 (augment) — add Tanimoto columns for extra fingerprints to an existing pairs parquet.

The MCES ground truth (the expensive RASCAL pass) is fixed per (i,j) pair, so evaluating a new
fingerprint means only recomputing Tanimoto on the SAME pairs — no re-run of MCES. Used to fold
the radius-10 arms into the corrected v2 pair set without recomputing structural similarity.

Drops any non-similarity bookkeeping columns (e.g. retry_timedout) so exp1_analyze.py — which
treats every non-(i,j,mces) column as a fingerprint — does not mistake them for an FP.

Usage:
    python exp1_add_fps.py --pairs exp1_pairs_v2.parquet --out exp1_pairs_v3.parquet \
        --fp NAME=/path/rankingset.pt [--fp ...]
"""
import argparse, os

import numpy as np
import pandas as pd

_KEEP_NONFP = ("i", "j", "mces")


def load_colsets(path, needed):
    import torch
    csr = torch.load(path, weights_only=True)
    crow = csr.crow_indices().numpy()
    col = csr.col_indices().numpy()
    N = int(csr.shape[0])
    return {i: set(col[crow[i]:crow[i + 1]].tolist()) for i in needed}, N


def tanimoto(a, b):
    if not a and not b:
        return 1.0
    inter = len(a & b)
    uni = len(a) + len(b) - inter
    return inter / uni if uni else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--fp", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.pairs)
    # drop bookkeeping / bool columns that are not fingerprint similarities
    drop = [c for c in df.columns
            if c not in _KEEP_NONFP and df[c].dtype == bool]
    if drop:
        print(f"dropping non-FP columns: {drop}", flush=True)
        df = df.drop(columns=drop)

    pairs = list(zip(df["i"].astype(int).tolist(), df["j"].astype(int).tolist()))
    needed = {i for p in pairs for i in p}
    print(f"{len(pairs)} pairs; {len(needed)} distinct rows needed", flush=True)

    for spec in args.fp:
        name, path = spec.split("=", 1)
        if name in df.columns:
            print(f"SKIP {name}: column already present", flush=True)
            continue
        if not os.path.exists(path):
            print(f"MISSING {name}: {path}", flush=True)
            continue
        colsets, N = load_colsets(path, needed)
        df[name] = np.array([tanimoto(colsets[i], colsets[j]) for i, j in pairs])
        print(f"  {name}: Tanimoto computed (D from {N} rows)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out)
    fp_cols = [c for c in df.columns if c not in _KEEP_NONFP]
    print(f"saved {len(df)} pairs -> {args.out}\nfingerprints now: {fp_cols}", flush=True)


if __name__ == "__main__":
    main()
