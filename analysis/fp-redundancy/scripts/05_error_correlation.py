#!/usr/bin/env python3
"""Is the redundancy acting as an error-correcting repetition code, or just as a weight?

Exact-duplicate bit groups share an identical ground-truth column, so the model
sees g copies of the same target. If its prediction errors across the group were
independent, averaging them would genuinely suppress noise (a useful repetition
code). If the errors are perfectly correlated, the group supplies no extra
evidence and its only effect on cosine similarity is a g-fold weight.

    pixi run python scripts/05_error_correlation.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pz = np.load(os.path.join(HERE, "results", "preds.npz"), allow_pickle=True)
mz = np.load(os.path.join(HERE, "results", "bit_mi.npz"))

probs = pz["probs"].astype(np.float32)
K, B = probs.shape
truth = np.zeros((K, B), dtype=np.float32)
fp_idx, fp_ptr = pz["fp_idx"], pz["fp_ptr"]
for k in range(K):
    truth[k, fp_idx[fp_ptr[k]:fp_ptr[k + 1]]] = 1.0

E = probs - truth
root, gsize = mz["dup_root"], mz["dup_group_size"]
rng = np.random.default_rng(0)


def mean_offdiag_corr(cols):
    """Mean pairwise Pearson correlation between error columns."""
    A = E[:, cols]
    A = A - A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0)
    ok = sd > 1e-8
    if ok.sum() < 2:
        return np.nan
    A = A[:, ok] / sd[ok]
    Cm = (A.T @ A) / len(A)
    n = Cm.shape[0]
    return float((Cm.sum() - np.trace(Cm)) / (n * (n - 1)))


groups = [np.nonzero(root == r)[0] for r in np.unique(root) if (root == r).sum() > 1]
within = np.array([mean_offdiag_corr(g) for g in groups])
within = within[~np.isnan(within)]

# control: random bit pairs matched on presence count to the grouped bits
grouped = np.concatenate(groups)
counts = mz["counts"]
ctrl = []
for _ in range(2000):
    i = int(rng.choice(grouped))
    cand = np.nonzero(np.abs(counts - counts[i]) < 0.05 * counts[i])[0]
    j = int(rng.choice(cand))
    if j != i:
        c = mean_offdiag_corr(np.array([i, j]))
        if not np.isnan(c):
            ctrl.append(c)
ctrl = np.array(ctrl)

print(f"{len(groups)} exact-duplicate groups (sizes {gsize[grouped].min()}-{gsize[grouped].max()}), "
      f"{len(grouped)} bits, over {K} test molecules\n")
print("mean pairwise correlation of prediction errors")
print(f"  within duplicate groups : {within.mean():.4f}   "
      f"(p10 {np.percentile(within,10):.3f}, median {np.median(within):.3f}, "
      f"p90 {np.percentile(within,90):.3f})")
print(f"  frequency-matched pairs : {ctrl.mean():.4f}   "
      f"(median {np.median(ctrl):.3f})")

# averaging g errors with pairwise correlation rho gives Var/sigma^2 = (1+(g-1)rho)/g
g = gsize[grouped].astype(np.float64)
rho = float(np.clip(within.mean(), 0, 1))
eff = (1.0 + (g - 1) * rho) / g
ideal = 1.0 / g
print(f"\nnoise remaining after averaging a duplicate group, Var/sigma^2 = (1+(g-1)*rho)/g")
print(f"  at the observed rho={rho:.3f} : {eff.mean():.4f}   (1.0 = no error correction)")
print(f"  if errors were independent  : {ideal.mean():.4f}")
