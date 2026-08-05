#!/usr/bin/env python3
"""Pairwise redundancy structure of the 16384-bit RankingEntropy fingerprint.

EntropyFPLoader.setup ranks candidate features by *marginal* binary entropy only.
This measures how much of that entropy is shared between bits, and emits the
per-bit "soft group size" used to build de-correlating retrieval weights in
03_reweight_sweep.py.

    pixi run python scripts/02_bit_mi.py
"""
import os
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKINGSET = os.path.join(HERE, "results", "rankingset.npz")
OUT = os.path.join(HERE, "results", "bit_mi.npz")
PAIRS = os.path.join(HERE, "results", "bit_mi_pairs.csv")

ROW_CHUNK = 50_000
BIT_BLOCK = 512
TAUS = (0.5, 0.7, 0.9, 0.99)


def log(m):
    print(m, flush=True)


z = np.load(RANKINGSET, allow_pickle=True)
N, B = (int(v) for v in z["shape"])
X = sp.csr_matrix((np.ones(len(z["indices"]), dtype=np.float32), z["indices"], z["indptr"]),
                  shape=(N, B))
frags, radii = z["frags"], z["radii"]
log(f"presence matrix: {N} x {B}, {X.nnz} nonzeros")

counts = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
p = counts / N
with np.errstate(divide="ignore", invalid="ignore"):
    H = np.nan_to_num(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
log(f"marginal entropy: sum {H.sum():.1f} bits, mean {H.mean():.6f}")

# ---- co-occurrence (counts <= N < 2^24, so float32 accumulation is exact)
t0 = time.time()
C = np.zeros((B, B), dtype=np.float32)
for s in range(0, N, ROW_CHUNK):
    Xc = X[s:s + ROW_CHUNK]
    C += (Xc.T @ Xc).toarray()
log(f"co-occurrence done ({time.time()-t0:.1f}s)")
assert np.array_equal(np.diag(C).astype(np.int64), counts.astype(np.int64))

# ---- pairwise MI, redundancy ratio, soft group sizes
t0 = time.time()
max_mi = np.zeros(B)
max_cond = np.zeros(B)
n_partners = {tau: np.ones(B, dtype=np.int32) for tau in TAUS}   # includes self
dup_i, dup_j, pr, pc, pm = [], [], [], [], []

for s in range(0, B, BIT_BLOCK):
    rows = np.arange(s, min(s + BIT_BLOCK, B))
    k = np.arange(len(rows))
    n11 = C[rows].astype(np.float64)
    ci, cj = counts[rows][:, None], counts[None, :]
    mi = np.zeros_like(n11)
    for n, mr, mc in ((n11, ci, cj), (ci - n11, ci, N - cj),
                      (cj - n11, N - ci, cj), (N - ci - cj + n11, N - ci, N - cj)):
        with np.errstate(divide="ignore", invalid="ignore"):
            mi += np.nan_to_num(n * np.log2((n * N) / (mr * mc)))
    mi /= N

    ratio = mi / np.minimum(H[rows][:, None], H[None, :])
    ratio[k, rows] = 0.0                       # ignore self
    for tau in TAUS:
        n_partners[tau][rows] += (ratio >= tau).sum(axis=1).astype(np.int32)

    mi[k, rows] = -1.0
    max_mi[rows] = mi.max(axis=1)
    cond = n11 / counts[None, :]
    cond[k, rows] = -1.0
    max_cond[rows] = cond.max(axis=1)

    di, dj = np.nonzero((n11 == counts[rows][:, None]) & (n11 == counts[None, :]))
    keep = rows[di] < dj
    dup_i.append(rows[di[keep]]); dup_j.append(dj[keep])

    up = ratio.copy(); up[:, :s] = 0.0
    ri, rj = np.nonzero(up > 0.5)
    if ri.size:
        pr.append(rows[ri]); pc.append(rj); pm.append(mi[ri, rj])
    log(f"  MI bits {s:>6d}-{rows[-1]:>6d} ({time.time()-t0:.1f}s)")

redundancy = max_mi / H

# ---- exact-duplicate groups (union-find)
parent = np.arange(B)


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for a, b in zip(np.concatenate(dup_i), np.concatenate(dup_j)):
    ra, rb = find(int(a)), find(int(b))
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)
root = np.array([find(i) for i in range(B)])
gsize = np.bincount(root)
dup_group_size = gsize[root]                    # per-bit exact duplicate multiplicity
log(f"\nexact duplicates: {B} bits -> {int((gsize>0).sum())} distinct columns "
    f"({B - int((gsize>0).sum())} wasted slots); largest group {gsize.max()}")

# note: normalized by H_i, NOT by min(H_i,H_j) -- the latter is what `mi_ratio`
# in bit_mi_pairs.csv and the n_partners/tau graph use. See 06_metric_definitions.py.
log("\n--- redundancy ratio  max_j I(i;j)/H_i ---")
for q in (0, 25, 50, 75, 95, 100):
    log(f"  p{q:<3d} {np.percentile(redundancy, q):.4f}")
log(f"  bits fully implied by another bit (max_j P(i|j)=1): {(max_cond >= 0.999999).sum()}")
for tau in TAUS:
    log(f"  mean soft group size @tau={tau}: {n_partners[tau].mean():.2f} "
        f"(max {n_partners[tau].max()})")

np.savez_compressed(
    OUT, counts=counts, H=H, redundancy=redundancy, max_cond=max_cond,
    dup_group_size=dup_group_size, dup_root=root,
    **{f"n_partners_{tau}": n_partners[tau] for tau in TAUS},
)
log(f"wrote {OUT}")

pi, pj, pmv = np.concatenate(pr), np.concatenate(pc), np.concatenate(pm)
rr = pmv / np.minimum(H[pi], H[pj])
top = np.argsort(-rr)[:2000]
with open(PAIRS, "w") as f:
    f.write("bit_i,bit_j,frag_i,radius_i,frag_j,radius_j,count_i,count_j,count_ij,"
            "mi_bits,mi_ratio,p_i_given_j,p_j_given_i\n")
    for kk in top:
        i, j = int(pi[kk]), int(pj[kk])
        f.write(f"{i},{j},{frags[i]},{radii[i]},{frags[j]},{radii[j]},"
                f"{counts[i]:.0f},{counts[j]:.0f},{C[i,j]:.0f},{pmv[kk]:.6f},{rr[kk]:.6f},"
                f"{C[i,j]/counts[j]:.6f},{C[i,j]/counts[i]:.6f}\n")
log(f"wrote {PAIRS} ({len(pi)} pairs above ratio 0.5)")
