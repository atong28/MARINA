"""Exp 1 (compute) — MCES similarity vs fingerprint similarity, sampled pairs.

Precedent: Huber & Pollmann "Count your bits" (RascalMCES structural reference);
Riniker & Landrum benchmarking platform. Tests whether each fingerprint's similarity
tracks a graph-based ground-truth structural similarity (Metric A: Spearman correlation).

RascalMCES is the ground truth; FP similarity is Tanimoto over the binary selected-bit
column-sets (all five fingerprints are binary thermometer/selected bits, so plain Tanimoto
is the consistent metric — no metric tuning). Runs in MARINA's env (torch + rdkit).

Output: results/exp1_pairs.parquet with columns [i, j, mces, <fp Tanimoto per fingerprint>].
exp1_analyze.py consumes it (torch-free).

Usage:
    python exp1_mces.py --retrieval /path/retrieval.pkl --n-pool 5000 --n-pairs 100000 \
        --workers 16 --seed 0 --timeout 60 \
        --fp NAME=/path/rankingset.pt [--fp ...] --out results/exp1_pairs.parquet
"""
import argparse, os, pickle, time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd


def load_colsets(path, needed):
    """Column-sets only for the `needed` row indices (dict), to avoid materializing
    all ~519k sets per fingerprint."""
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


_SMILES = None
_TIMEOUT = 60


def _mces_pair(ij):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdRascalMCES
    RDLogger.DisableLog("rdApp.*")
    i, j = ij
    m1 = Chem.MolFromSmiles(_SMILES[i])
    m2 = Chem.MolFromSmiles(_SMILES[j])
    if m1 is None or m2 is None:
        return (i, j, np.nan)
    opts = rdRascalMCES.RascalOptions()
    opts.similarityThreshold = 0.05
    opts.timeout = _TIMEOUT
    try:
        res = rdRascalMCES.FindMCES(m1, m2, opts)
        sim = float(res[0].similarity) if res else np.nan
    except Exception:
        sim = np.nan
    return (i, j, sim)


def _init(smiles, timeout):
    global _SMILES, _TIMEOUT
    _SMILES, _TIMEOUT = smiles, timeout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--fp", action="append", default=[])
    ap.add_argument("--n-pool", type=int, default=5000)
    ap.add_argument("--n-pairs", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.retrieval, "rb") as f:
        R = pickle.load(f)
    n_all = len(R)
    smiles = [(R[i]["smiles"] if isinstance(R[i], dict) else R[i]) for i in range(n_all)]

    # mass-stratified pool sample: sort by mass, take evenly spaced strata
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    masses = np.array([Descriptors.ExactMolWt(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s)
                       else np.nan for s in smiles])
    valid = np.where(np.isfinite(masses))[0]
    order = valid[np.argsort(masses[valid])]
    idx = order[np.linspace(0, len(order) - 1, args.n_pool).astype(int)]
    idx = np.unique(idx)
    print(f"pool: {len(idx)} molecules spanning mass {masses[idx].min():.0f}-{masses[idx].max():.0f}", flush=True)

    # sample unordered pairs from the pool
    pairs = set()
    while len(pairs) < args.n_pairs:
        a, b = rng.choice(idx, 2, replace=False)
        pairs.add((int(min(a, b)), int(max(a, b))))
    pairs = list(pairs)
    print(f"sampled {len(pairs)} unique pairs", flush=True)

    # FP similarities (cheap, in-process) — only for rows touched by sampled pairs
    needed = {i for p in pairs for i in p}
    fp_sims = {}
    for spec in args.fp:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"MISSING {name}: {path}", flush=True)
            continue
        colsets, N = load_colsets(path, needed)
        fp_sims[name] = np.array([tanimoto(colsets[i], colsets[j]) for i, j in pairs])
        print(f"  {name}: FP Tanimoto computed (D from {N} rows)", flush=True)

    # MCES ground truth (parallel, the expensive part)
    t0 = time.time()
    pos = {p: k for k, p in enumerate(pairs)}
    results = np.full(len(pairs), np.nan)
    with Pool(args.workers, initializer=_init, initargs=(smiles, args.timeout)) as pool:
        for k, (i, j, sim) in enumerate(pool.imap_unordered(_mces_pair, pairs, chunksize=64)):
            results[pos[(i, j)]] = sim
            if k % 5000 == 0:
                print(f"  mces {k}/{len(pairs)}  {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame({"i": [p[0] for p in pairs], "j": [p[1] for p in pairs], "mces": results})
    for name, sims in fp_sims.items():
        df[name] = sims
    n_timeout = int(np.isnan(results).sum())
    print(f"MCES done in {time.time()-t0:.0f}s; {n_timeout} failed/timeout "
          f"({100*n_timeout/len(pairs):.1f}%)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out)
    print(f"saved {len(df)} pairs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
