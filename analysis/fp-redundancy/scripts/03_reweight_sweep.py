#!/usr/bin/env python3
"""Does de-correlating the retrieval metric improve ranking?

Keeps the trained model and the 16384 selected bits fixed and changes only the
similarity used at retrieval time, from plain cosine to a weighted cosine

    sim_w(q, r) = <s*q, s*r> / (||s*q|| ||s*r||),   s = sqrt(w)

which is ordinary cosine after scaling every bit by sqrt(w_i). Weighting is
applied consistently to the rankingset rows, the query, and the truth vector that
sets the per-query threshold, exactly mirroring RankingSet.batched_rank:

    rank = #{rows with sim >= sim(query, truth)} - 1

    pixi run python scripts/03_reweight_sweep.py
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKINGSET = os.path.join(HERE, "results", "rankingset.npz")
PREDS = os.path.join(HERE, "results", "preds.npz")
BITMI = os.path.join(HERE, "results", "bit_mi.npz")
OUT = os.path.join(HERE, "results", "reweight_sweep.json")

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="queries to use (0 = all cached)")
ap.add_argument("--qchunk", type=int, default=64)
ap.add_argument("--tol", type=float, default=1e-6, help="tie tolerance at the threshold")
cli = ap.parse_args()


def log(m):
    print(m, flush=True)


z = np.load(RANKINGSET, allow_pickle=True)
N, B = (int(v) for v in z["shape"])
indices, indptr = z["indices"], z["indptr"]
nnz_row = np.diff(indptr)
row_of_nnz = np.repeat(np.arange(N, dtype=np.int32), nnz_row)
# binary copy, used only to get per-row weight sums (handles the empty rows that
# np.add.reduceat would silently get wrong)
Xbin = sp.csr_matrix((np.ones(len(indices), dtype=np.float64), indices, indptr), shape=(N, B))
log(f"empty rankingset rows: {int((nnz_row == 0).sum())}")

pz = np.load(PREDS, allow_pickle=True)
probs = pz["probs"].astype(np.float32)
fp_idx, fp_ptr = pz["fp_idx"], pz["fp_ptr"]
K = probs.shape[0] if cli.limit in (0, None) else min(cli.limit, probs.shape[0])
probs = probs[:K]
log(f"rankingset {N}x{B}, {len(indices)} nnz; {K} cached queries from {pz['ckpt']}")

mz = np.load(BITMI)
counts = mz["counts"]

SCHEMES = {
    "uniform":  np.ones(B),                              # baseline = current behavior
    "dup":      1.0 / mz["dup_group_size"],              # exact duplicate collapse
    "soft0.99": 1.0 / mz["n_partners_0.99"],
    "soft0.9":  1.0 / mz["n_partners_0.9"],
    "soft0.7":  1.0 / mz["n_partners_0.7"],
    "soft0.5":  1.0 / mz["n_partners_0.5"],
    "idf":      np.log(N / np.maximum(counts, 1)),       # control: frequency only
}


def evaluate(w):
    """Ranks of each cached query under weighted cosine with bit weights w."""
    w = np.asarray(w, dtype=np.float64)
    assert (w > 0).all(), "weights must be positive"
    s = np.sqrt(w).astype(np.float32)

    # weighted, row-normalized rankingset;  ||row||^2 = sum of w over the row support
    rownorm = np.sqrt(np.maximum(Xbin @ w, 1e-30)).astype(np.float32)
    data = s[indices] / rownorm[row_of_nnz]
    Xw = sp.csr_matrix((data, indices, indptr), shape=(N, B))

    # weighted, normalized queries and truths
    Q = probs * s[None, :]
    Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-30)
    thresh = np.zeros(K, dtype=np.float32)
    for k in range(K):
        t_ix = fp_idx[fp_ptr[k]:fp_ptr[k + 1]]
        tv = s[t_ix]
        thresh[k] = float(Q[k, t_ix] @ tv / max(np.linalg.norm(tv), 1e-30))

    ranks = np.zeros(K, dtype=np.int64)
    for start in range(0, K, cli.qchunk):
        end = min(start + cli.qchunk, K)
        sims = Xw @ Q[start:end].T                       # (N, nq)
        ranks[start:end] = (sims >= thresh[start:end][None, :] - cli.tol).sum(axis=0) - 1
    return np.maximum(ranks, 0)


results = {}
all_ranks = {}
baseline = None
for name, w in SCHEMES.items():
    t0 = time.time()
    r = evaluate(w)
    all_ranks[name] = r
    m = {
        "rank_1": float((r < 1).mean()),
        "rank_5": float((r < 5).mean()),
        "rank_10": float((r < 10).mean()),
        "mean_rank": float(r.mean()),
        "median_rank": float(np.median(r)),
    }
    if baseline is None:
        baseline = r
    else:
        m["better"] = int((r < baseline).sum())
        m["worse"] = int((r > baseline).sum())
    results[name] = m
    log(f"{name:>9s}  top1 {m['rank_1']:.4f}  top5 {m['rank_5']:.4f}  top10 {m['rank_10']:.4f}"
        f"  mean {m['mean_rank']:8.1f}  median {m['median_rank']:6.1f}"
        + (f"  (better {m['better']}, worse {m['worse']})" if "better" in m else "")
        + f"   [{time.time()-t0:.0f}s]")

with open(OUT, "w") as f:
    json.dump({"n_queries": K, "ckpt": str(pz["ckpt"]), "results": results}, f, indent=2)
log(f"wrote {OUT}")

RANKS = os.path.join(HERE, "results", "reweight_ranks.npz")
np.savez_compressed(RANKS, **all_ranks)
log(f"wrote {RANKS}")
