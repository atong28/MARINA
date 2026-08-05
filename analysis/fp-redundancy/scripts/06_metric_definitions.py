#!/usr/bin/env python3
"""Pin down exactly what the three redundancy summaries mean, and verify the
directionality claims.

  (a) two normalizations of the per-bit redundancy ratio, which are NOT the same
  (b) is P(i|j)=1 one-way?  is exact duplication the two-way case?
  (c) are the tau-neighbourhoods ("soft groups") actually cliques?

    pixi run python scripts/06_metric_definitions.py
"""
import os
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
z = np.load(os.path.join(HERE, "results", "rankingset.npz"), allow_pickle=True)
N, B = (int(v) for v in z["shape"])
X = sp.csr_matrix((np.ones(len(z["indices"]), dtype=np.float32), z["indices"], z["indptr"]),
                  shape=(N, B))
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

red_H = np.zeros(B)       # max_j I(i;j) / H_i
red_min = np.zeros(B)     # max_j I(i;j) / min(H_i, H_j)
max_cond = np.zeros(B)    # max_j P(i | j)
arg_cond = np.zeros(B, dtype=np.int32)
deg = np.zeros(B, dtype=np.int32)   # tau=0.5 neighbours, min-normalized

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
    mi /= N
    mi[k, rows] = -1.0

    red_H[rows] = mi.max(axis=1) / H[rows]
    rmin = mi / np.minimum(H[rows][:, None], H[None, :])
    rmin[k, rows] = -1.0
    red_min[rows] = rmin.max(axis=1)
    deg[rows] = (rmin >= 0.5).sum(axis=1)

    cond = n11 / counts[None, :]
    cond[k, rows] = -1.0
    arg_cond[rows] = np.argmax(cond, axis=1)
    max_cond[rows] = cond[k, arg_cond[rows]]

print("\n(a) two normalizations of the per-bit redundancy ratio")
print("      max_j I(i;j)/H_i            max_j I(i;j)/min(H_i,H_j)")
for q in (25, 50, 75, 95):
    print(f"  p{q:<3d}  {np.percentile(red_H, q):.4f}                      "
          f"{np.percentile(red_min, q):.4f}")
for thr in (0.5, 0.9, 0.99):
    print(f"  bits above {thr}: {(red_H > thr).sum():>6d}                  "
          f"{(red_min > thr).sum():>6d}")

print("\n(b) directionality of P(i|j) = 1")
implied = max_cond >= 0.999999
j = arg_cond[implied]
i = np.nonzero(implied)[0]
back = C[i, j] / counts[i]                       # P(j | i) for the same pair
two_way = back >= 0.999999
print(f"  bits with some j s.t. P(i|j)=1        : {implied.sum()}")
print(f"    of those, P(j|i)=1 as well (two-way): {int(two_way.sum())}")
print(f"    strictly one-way                    : {int((~two_way).sum())}")
print(f"  for one-way pairs, median count_j/count_i = "
      f"{np.median(counts[j[~two_way]] / counts[i[~two_way]]):.3f}  "
      f"(j is the rarer, more specific bit; support(j) is a subset of support(i))")
eq = (C[i, j] == counts[i]) & (C[i, j] == counts[j])
print(f"  two-way implication == identical support columns: {bool(np.array_equal(two_way, eq))}")

print("\n(c) are tau=0.5 neighbourhoods cliques?")
rng = np.random.default_rng(0)
frac = []
for b in rng.choice(B, 200, replace=False):
    nb = np.nonzero(
        np.nan_to_num(np.array([0.0])) == 0)[0]  # placeholder replaced below
    row = C[b].astype(np.float64)
    mi_row = np.zeros(B)
    ci = counts[b]
    for n, mr, mc in ((row, ci, counts), (ci - row, ci, N - counts),
                      (counts - row, N - ci, counts), (N - ci - counts + row, N - ci, N - counts)):
        with np.errstate(divide="ignore", invalid="ignore"):
            mi_row += np.nan_to_num(n * np.log2((n * N) / (mr * mc)))
    mi_row /= N
    r = mi_row / np.minimum(H[b], H)
    r[b] = 0.0
    nb = np.nonzero(r >= 0.5)[0]
    if len(nb) < 2:
        continue
    sub = C[np.ix_(nb, nb)].astype(np.float64)
    cn = counts[nb]
    misub = np.zeros_like(sub)
    for n, mr, mc in ((sub, cn[:, None], cn[None, :]),
                      (cn[:, None] - sub, cn[:, None], N - cn[None, :]),
                      (cn[None, :] - sub, N - cn[:, None], cn[None, :]),
                      (N - cn[:, None] - cn[None, :] + sub, N - cn[:, None], N - cn[None, :])):
        with np.errstate(divide="ignore", invalid="ignore"):
            misub += np.nan_to_num(n * np.log2((n * N) / (mr * mc)))
    misub /= N
    rsub = misub / np.minimum(cn[:, None] * 0 + H[nb][:, None], H[nb][None, :])
    m = len(nb)
    off = (rsub >= 0.5).sum() - m
    frac.append(off / (m * (m - 1)))
frac = np.array(frac)
print(f"  sampled {len(frac)} bits with >=2 neighbours; mean degree {deg.mean():.1f}")
print(f"  fraction of neighbour-pairs that are themselves >=0.5 redundant: "
      f"mean {frac.mean():.3f}, median {np.median(frac):.3f}")
print("  (1.0 would mean the neighbourhood is a true clique / genuine 'group')")
