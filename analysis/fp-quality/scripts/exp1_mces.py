"""Exp 1 (compute) — MCES similarity vs fingerprint similarity, sampled pairs.

Precedent: Huber & Pollmann "Count your bits" (RascalMCES structural reference);
Riniker & Landrum benchmarking platform. Tests whether each fingerprint's similarity
tracks a graph-based ground-truth structural similarity (Metric A: Spearman correlation).

RascalMCES is the ground truth; FP similarity is Tanimoto over the binary selected-bit
column-sets (all fingerprints are binary thermometer/selected bits, so plain Tanimoto is the
consistent metric — no metric tuning).

Single correct pass: RascalMCES runs with returnEmptyMCES=True (default minFragSize), so
screened-out dissimilar pairs get their real ~0 similarity instead of being dropped as NaN
(the v1 bias, which the old exp1_mces_retry.py patched after the fact — now folded in here).
Only true timeouts remain NaN and are flagged in the `timedout` column.

Do NOT set opts.minFragSize: on the MARINA-DB pool it makes RASCAL's search explode on
ordinary 250-900 Da pairs (real sim 0.05-0.2, above the screen so the full search runs) — ~35%
of pairs hit the 60s timeout, collapsing throughput ~3x. returnEmptyMCES alone is fast and
returns the same similarities; minFragSize=3 was the regression, verified pair-by-pair.

Output: <out>.parquet with columns [i, j, mces, timedout, <fp Tanimoto per fingerprint>].
exp1_analyze.py consumes it.

Deps: torch (read CSR) + rdkit (MCES) + pandas + pyarrow. Run under the ~/Workspace master
pixi env locally, or the Nautilus MARINA image on the cluster.

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
        return (i, j, np.nan, False)   # unparseable -> excluded, not a timeout
    opts = rdRascalMCES.RascalOptions()
    opts.similarityThreshold = 0.05
    # NB: do NOT set opts.minFragSize — it explodes RASCAL's search into 60s timeouts on
    # ordinary above-threshold pairs (see module docstring). Default minFragSize is correct.
    # returnEmptyMCES=True is essential: without it RASCAL returns an EMPTY result for pairs
    # it screens out below the 0.05 threshold (genuinely dissimilar, MCES ~ 0), which then get
    # logged NaN and dropped -- removing the hard-to-rank dissimilar tail and biasing every
    # Spearman rho high (the v1 bug this fold-in fixes; no separate retry pass needed).
    opts.returnEmptyMCES = True
    opts.timeout = _TIMEOUT
    try:
        r = rdRascalMCES.FindMCES(m1, m2, opts)
        if not r:
            return (i, j, np.nan, False)
        res = r[0]
        # timedOut=True => intractable at this timeout, left NaN (true timeout); else the
        # similarity is reliable, including the recovered ~0 screen-outs.
        return (i, j, float(res.similarity), bool(res.timedOut))
    except Exception:
        return (i, j, np.nan, True)


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
    timedout = np.zeros(len(pairs), dtype=bool)
    with Pool(args.workers, initializer=_init, initargs=(smiles, args.timeout)) as pool:
        for k, (i, j, sim, to) in enumerate(pool.imap_unordered(_mces_pair, pairs, chunksize=16)):
            results[pos[(i, j)]] = sim
            timedout[pos[(i, j)]] = to
            if k % 5000 == 0:
                print(f"  mces {k}/{len(pairs)}  {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame({"i": [p[0] for p in pairs], "j": [p[1] for p in pairs],
                       "mces": results, "timedout": timedout})
    for name, sims in fp_sims.items():
        df[name] = sims
    n_nan = int(np.isnan(results).sum())
    n_to = int(timedout.sum())
    print(f"MCES done in {time.time()-t0:.0f}s; {n_nan} NaN "
          f"({100*n_nan/len(pairs):.1f}%), of which {n_to} true timeouts; "
          f"{len(pairs)-n_nan} valid", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out)
    print(f"saved {len(df)} pairs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
