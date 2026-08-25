#!/usr/bin/env python3
"""Does crediting same-substructure bits actually improve retrieval?

`03_reweight_sweep.py` in ../fp-redundancy/ swept *diagonal* metrics -- rescale each bit
independently -- and found none beat plain cosine. The metric implied by "two columns
describing the same substructure should not be orthogonal" is not diagonal: it is
M = G G^T with off-diagonal entries on same-substructure pairs. For any PSD M = L L^T,

    sim_M(q, r) = cos(L^T q, L^T r)

so this is the identical protocol with the elementwise scaling replaced by the group
projection G^T. Ranks are counted exactly as RankingSet.batched_rank does:

    rank = #{rows with sim >= sim(query, truth)} - 1

Two ways to merge a group, both tested, because they are a real design fork:
  sum : merged value = number of on columns in the group (count-like)
  or  : merged value = 1 if any column is on   <- what the substructure FP branch does

    pixi run python3 scripts/02_merge_kernel.py
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REDUNDANCY = os.path.join(os.path.dirname(HERE), "fp-redundancy")

ap = argparse.ArgumentParser()
ap.add_argument("--rankingset", default=os.path.join(REDUNDANCY, "results", "rankingset.npz"))
ap.add_argument("--preds", default=os.path.join(REDUNDANCY, "results", "preds.npz"))
ap.add_argument("--groups", default=os.path.join(HERE, "results", "groups.npz"))
ap.add_argument("--out", default=os.path.join(HERE, "results", "merge_kernel.json"))
ap.add_argument("--limit", type=int, default=0, help="queries to use (0 = all cached)")
ap.add_argument("--qchunk", type=int, default=64)
ap.add_argument("--tol", type=float, default=1e-6, help="tie tolerance at the threshold")
ap.add_argument("--boot", type=int, default=10000)
cli = ap.parse_args()


def log(m):
    print(m, flush=True)


z = np.load(cli.rankingset, allow_pickle=True)
N, B = (int(v) for v in z["shape"])
indices, indptr = z["indices"], z["indptr"]
Xbin = sp.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr), shape=(N, B))

pz = np.load(cli.preds, allow_pickle=True)
probs = pz["probs"].astype(np.float32)
fp_idx, fp_ptr = pz["fp_idx"], pz["fp_ptr"]
K = probs.shape[0] if cli.limit in (0, None) else min(cli.limit, probs.shape[0])
probs = probs[:K]

gz = np.load(cli.groups, allow_pickle=True)
assert int(gz["n_bits"]) == B, "groups.npz built against a different vocabulary"
log(f"rankingset {N}x{B}, {len(indices)} nnz; {K} queries from {pz['ckpt']}")


def evaluate(gid, n_groups, binarize):
    """Ranks of each cached query under cosine after projecting onto substructure groups."""
    Gmat = sp.csr_matrix((np.ones(B, dtype=np.float32), (np.arange(B), gid)), shape=(B, n_groups))

    Z = (Xbin @ Gmat).tocsr()
    if binarize:
        Z.data[:] = 1.0
    rownorm = np.sqrt(np.maximum(Z.multiply(Z).sum(axis=1).A.ravel(), 1e-30)).astype(np.float32)
    Z = sp.diags(1.0 / rownorm) @ Z

    Q = probs @ Gmat                                     # (K, n_groups) dense
    Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-30)

    thresh = np.zeros(K, dtype=np.float32)
    for k in range(K):
        gu, cnts = np.unique(gid[fp_idx[fp_ptr[k]:fp_ptr[k + 1]]], return_counts=True)
        tv = np.ones(len(gu), dtype=np.float32) if binarize else cnts.astype(np.float32)
        thresh[k] = float(Q[k, gu] @ tv / max(np.linalg.norm(tv), 1e-30))

    ranks = np.zeros(K, dtype=np.int64)
    for start in range(0, K, cli.qchunk):
        end = min(start + cli.qchunk, K)
        sims = Z @ Q[start:end].T                        # (N, nq)
        ranks[start:end] = (sims >= thresh[start:end][None, :] - cli.tol).sum(axis=0) - 1
    return np.maximum(ranks, 0)


SCHEMES = [("baseline", np.arange(B, dtype=np.int32), B, False)]
for variant in ("strict", "atom"):
    gid, gn = gz[f"gid_{variant}"], int(gz[f"n_groups_{variant}"])
    SCHEMES.append((f"{variant}-sum", gid, gn, False))
    SCHEMES.append((f"{variant}-or", gid, gn, True))

rng = np.random.default_rng(0)
boot_idx = rng.integers(0, K, size=(cli.boot, K))
results, all_ranks, baseline = {}, {}, None

for name, gid, gn, binarize in SCHEMES:
    t0 = time.time()
    r = evaluate(gid, gn, binarize)
    all_ranks[name] = r
    m = {"n_groups": int(gn),
         "rank_1": float((r < 1).mean()), "rank_5": float((r < 5).mean()),
         "rank_10": float((r < 10).mean()), "mean_rank": float(r.mean()),
         "median_rank": float(np.median(r))}
    line = (f"{name:>12s}  top1 {m['rank_1']:.4f}  top5 {m['rank_5']:.4f}  "
            f"top10 {m['rank_10']:.4f}  mean {m['mean_rank']:8.1f}  median {m['median_rank']:5.1f}")
    if baseline is None:
        baseline = r
    else:
        fixed = int(((baseline > 0) & (r == 0)).sum())
        broke = int(((baseline == 0) & (r > 0)).sum())
        # paired bootstrap over queries on the top-1 delta
        b1 = (baseline < 1).astype(np.float32)
        r1 = (r < 1).astype(np.float32)
        d = (r1 - b1)[boot_idx].mean(axis=1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        m.update(fixed_top1=fixed, broke_top1=broke,
                 better=int((r < baseline).sum()), worse=int((r > baseline).sum()),
                 d_top1=float(r1.mean() - b1.mean()), ci95=[float(lo), float(hi)],
                 p_not_better=float((d <= 0).mean()))
        line += (f"\n{'':12s}  top1 delta {m['d_top1']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                 f"p(delta<=0) {m['p_not_better']:.3f}   fixed {fixed}, broke {broke}")
    results[name] = m
    log(line + f"   [{time.time()-t0:.0f}s]")

with open(cli.out, "w") as f:
    json.dump({"n_queries": K, "ckpt": str(pz["ckpt"]), "n_bootstrap": cli.boot,
               "results": results}, f, indent=2)
np.savez_compressed(os.path.join(HERE, "results", "merge_kernel_ranks.npz"), **all_ranks)
log(f"wrote {cli.out}")
