#!/usr/bin/env python3
"""Regenerate the pair-level redundancy tables with corrected normalization labels.

Emits markdown to stdout and to results/pair_tables.md:
  A. pair-level summary distributions
  B. strongest two-way pairs (identical support = exact duplicates)
  C. strongest strictly one-way implications, P(i|j)=1 but P(j|i)<1
  D. the largest exact-duplicate groups, with fragments

    pixi run python scripts/07_pair_tables.py
"""
import os
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "pair_tables.md")

z = np.load(os.path.join(HERE, "results", "rankingset.npz"), allow_pickle=True)
N, B = (int(v) for v in z["shape"])
frags, atoms, radii = z["frags"], z["atoms"], z["radii"]
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
print(f"co-occurrence ({time.time()-t0:.0f}s)\n", flush=True)

mz = np.load(os.path.join(HERE, "results", "bit_mi.npz"))
root = mz["dup_root"]

EPS = 1e-6
two_i, two_j, two_mi = [], [], []
one_i, one_j, one_mi = [], [], []
n_pairs_ratio = []

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

    fwd = n11 / cj          # P(row | col)
    bwd = n11 / ci          # P(col | row)
    fwd[k, rows] = 0.0
    bwd[k, rows] = 0.0

    two = (fwd >= 1 - EPS) & (bwd >= 1 - EPS) & (rows[:, None] < np.arange(B)[None, :])
    ti, tj = np.nonzero(two)
    two_i.append(rows[ti]); two_j.append(tj); two_mi.append(mi[ti, tj])

    one = (fwd >= 1 - EPS) & (bwd < 0.9)
    oi, oj = np.nonzero(one)
    one_i.append(rows[oi]); one_j.append(oj); one_mi.append(mi[oi, oj])

    rmin = mi / np.minimum(H[rows][:, None], H[None, :])
    rmin[k, rows] = -1.0
    n_pairs_ratio.append(rmin[rmin > 0.0].astype(np.float32))

two_i, two_j, two_mi = (np.concatenate(a) for a in (two_i, two_j, two_mi))
one_i, one_j, one_mi = (np.concatenate(a) for a in (one_i, one_j, one_mi))
allr = np.concatenate(n_pairs_ratio)

L = []


def w(s=""):
    L.append(s)
    print(s, flush=True)


def frag(i, n=34):
    f = frags[i]
    if not f:                       # radius-0 bits: bare atom type, no fragment SMILES
        return f"[{atoms[i]}] atom"
    return f if len(f) <= n else f[:n - 1] + "…"


w("### A. Pair-level summary\n")
w(f"Over all {B*(B-1)//2:,} unordered pairs of the {B:,} bits "
  f"({len(allr):,} have I(i;j) > 0).\n")
w("| quantity | value |")
w("|---|---|")
w(f"| pairs with `I(i;j)/min(H_i,H_j)` > 0.5 | {int((allr>0.5).sum())//2:,} |")
w(f"| pairs > 0.9 | {int((allr>0.9).sum())//2:,} |")
w(f"| pairs > 0.99 | {int((allr>0.99).sum())//2:,} |")
w(f"| two-way implication pairs (identical support) | {len(two_i):,} |")
w(f"| strictly one-way pairs, P(i\\|j)=1 and P(j\\|i)<0.9 | {len(one_i):,} |")
w(f"| median `I(i;j)/min(H_i,H_j)` over nonzero pairs | {np.median(allr):.4f} |")

w("\n### B. Strongest two-way pairs (identical support = exact duplicates)\n")
w("`P(i|j) = P(j|i) = 1`, so `I(i;j) = H_i = H_j` and the ratio is 1.000 under either "
  "normalization. Ranked by MI, i.e. by how much entropy is duplicated.\n")
w("| frag i | r | frag j | r | mols | MI (bits) |")
w("|---|---|---|---|---|---|")
for kk in np.argsort(-two_mi)[:12]:
    i, j = int(two_i[kk]), int(two_j[kk])
    w(f"| `{frag(i)}` | {radii[i]} | `{frag(j)}` | {radii[j]} | "
      f"{counts[i]:.0f} | {two_mi[kk]:.4f} |")

w("\n### C. Strongest strictly one-way implications\n")
w("`P(i|j) = 1` (support(j) ⊆ support(i)) but `P(j|i) < 0.9`. Bit *j* is the rarer, "
  "more specific fragment; *i* still fires without it, so `H(i|j) > 0` and the slot "
  "is **not** free. Ranked by MI.\n")
w("| implied bit i | r | mols i | implying bit j | r | mols j | P(j\\|i) | MI (bits) | MI/H_i |")
w("|---|---|---|---|---|---|---|---|---|")
for kk in np.argsort(-one_mi)[:12]:
    i, j = int(one_i[kk]), int(one_j[kk])
    w(f"| `{frag(i, 26)}` | {radii[i]} | {counts[i]:.0f} | `{frag(j, 26)}` | {radii[j]} | "
      f"{counts[j]:.0f} | {C[i,j]/counts[i]:.3f} | {one_mi[kk]:.4f} | "
      f"{one_mi[kk]/H[i]:.3f} |")

w("\n### D. Largest exact-duplicate groups\n")
sizes = np.bincount(root)
w("| group size | mols | example fragments |")
w("|---|---|---|")
for r in np.argsort(-sizes)[:8]:
    if sizes[r] < 2:
        continue
    mem = np.nonzero(root == r)[0]
    ex = ", ".join(f"`{frag(int(m), 22)}`" for m in mem[:3])
    w(f"| {len(mem)} | {counts[mem[0]]:.0f} | {ex} |")

with open(OUT, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"\nwrote {OUT}")
