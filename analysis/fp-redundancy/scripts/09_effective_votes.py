#!/usr/bin/env python3
"""How distorted is a single molecule's fingerprint by redundancy?

A rankingset row contributes each of its set bits equally to the cosine dot
product. If those bits collapse into far fewer independent structural claims, the
molecule's "vote" is inflated by whichever substructure families happen to have
spawned the most correlated bits.

Counts, per molecule, the set bits vs the number of distinct connected components
of the tau-redundancy graph they touch.

    pixi run python scripts/09_effective_votes.py
"""
import os
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
z = np.load(os.path.join(HERE, "results", "rankingset.npz"), allow_pickle=True)
N, B = (int(v) for v in z["shape"])
indices, indptr = z["indices"], z["indptr"]
X = sp.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr), shape=(N, B))
counts = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
p = counts / N
with np.errstate(divide="ignore", invalid="ignore"):
    H = np.nan_to_num(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))

t0 = time.time()
C = np.zeros((B, B), dtype=np.float32)
for s in range(0, N, 50_000):
    Xc = X[s:s + 50_000]
    C += (Xc.T @ Xc).toarray()
print(f"co-occurrence ({time.time()-t0:.0f}s)", flush=True)

TAUS = (0.99, 0.9, 0.7)
edges = {tau: [[], []] for tau in TAUS}
for s in range(0, B, 512):
    rows = np.arange(s, min(s + 512, B))
    k = np.arange(len(rows))
    n11 = C[rows].astype(np.float64)
    ci, cj = counts[rows][:, None], counts[None, :]
    mi = np.zeros_like(n11)
    for n, mr, mc in ((n11, ci, cj), (ci - n11, ci, N - cj),
                      (cj - n11, N - ci, cj), (N - ci - cj + n11, N - ci, N - cj)):
        with np.errstate(divide="ignore", invalid="ignore"):
            mi += np.nan_to_num(n * np.log2((n * N) / (mr * mc)))
    ratio = (mi / N) / np.minimum(H[rows][:, None], H[None, :])
    ratio[k, rows] = 0.0
    ratio[:, :s] = 0.0
    for tau in TAUS:
        a, b = np.nonzero(ratio >= tau)
        edges[tau][0].append(rows[a]); edges[tau][1].append(b)

row_of_nnz = np.repeat(np.arange(N, dtype=np.int64), np.diff(indptr))
nbits = np.diff(indptr)
nz = nbits > 0

print(f"\nmolecules: {N} ({int((~nz).sum())} with no bits set)")
print(f"set bits per molecule: median {np.median(nbits[nz]):.0f}, mean {nbits[nz].mean():.1f}\n")
print(f"{'tau':>6s} {'edges':>8s} {'components':>11s} {'largest':>8s} "
      f"{'median votes':>13s} {'median inflation':>17s}")

for tau in TAUS:
    ei = np.concatenate(edges[tau][0]); ej = np.concatenate(edges[tau][1])
    g = sp.coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(B, B))
    ncomp, lab = sp.csgraph.connected_components(g, directed=False)
    sizes = np.bincount(lab)

    # distinct components touched by each molecule's set bits
    key = row_of_nnz * np.int64(ncomp + 1) + lab[indices]
    uniq = np.unique(key)
    votes = np.bincount(uniq // np.int64(ncomp + 1), minlength=N)
    infl = nbits[nz] / np.maximum(votes[nz], 1)
    print(f"{tau:>6.2f} {len(ei):>8d} {ncomp:>11d} {sizes.max():>8d} "
          f"{np.median(votes[nz]):>13.0f} {np.median(infl):>17.2f}")
    if tau == 0.9:
        pct = {q: np.percentile(infl, q) for q in (50, 75, 90, 99, 99.9, 100)}
        print("        inflation distribution: " +
              "  ".join(f"p{q}={v:.2f}" for q, v in pct.items()))
        for t in (1.25, 1.5, 2.0):
            print(f"        molecules with inflation > {t}: {int((infl > t).sum())} "
                  f"({100*(infl > t).mean():.2f}%)")

print("\n'votes' = distinct redundancy components a molecule touches; "
      "'inflation' = set bits / votes.")
print("A largest-component size near B means the graph has percolated and the "
      "component count is no longer meaningful.")
