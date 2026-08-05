#!/usr/bin/env python3
"""Is MARINA's fingerprint prediction biased by bit index?

Bits are ordered by descending marginal entropy (fp_loader.py: topk_sorted =
sorted(..., key=-ent[i])), and since every selected bit has p << 0.5 that is also
descending frequency. So "later bits" = rarer bits. Question: does the model
systematically under-predict them, or is the error index-independent?

Splits every statistic by whether a bit is fully implied by another bit
(max_cond == 1 from 02_bit_mi.py) -- 10,608 of 16,384 are, and they concentrate in
the early indices, so an unstratified trend would be a mix artifact.

    pixi run python scripts/10_bit_calibration.py
"""
import os

import numpy as np
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDS = os.path.join(HERE, "results", "preds.npz")
BIT_MI = os.path.join(HERE, "results", "bit_mi.npz")
OUT_PNG = os.path.join(HERE, "results", "bit_calibration.png")

NBINS = 16
BOOT = 2000
SEED = 0

d = np.load(PREDS, allow_pickle=True)
P = d["probs"].astype(np.float32)
n, K = P.shape
Y = np.zeros((n, K), dtype=np.float32)
idx, ptr = d["fp_idx"], d["fp_ptr"]
for i in range(n):
    Y[i, idx[ptr[i]:ptr[i + 1]]] = 1.0
implied = np.load(BIT_MI)["max_cond"] >= 0.999999

print(f"ckpt {d['ckpt']}  n={n} molecules  K={K} bits")
print(f"true bits/mol {Y.sum(1).mean():.1f}   predicted mass/mol {P.sum(1).mean():.1f}")
print(f"implied-by-another-bit: {implied.sum()}/{K}\n")


def per_bit_auc(cols):
    """Mean over bits of the within-bit (across-molecule) ranking AUC."""
    aucs = []
    for j in cols:
        y = Y[:, j]
        k = int(y.sum())
        if k == 0 or k == n:
            continue
        r = rankdata(P[:, j])
        aucs.append((r[y == 1].sum() - k * (k + 1) / 2) / (k * (n - k)))
    return float(np.mean(aucs)) if aucs else float("nan")


bins = np.array_split(np.arange(K), NBINS)
rng = np.random.default_rng(SEED)

def prf(cols):
    """Micro-averaged precision / recall / F1 at 0.5, pooled over bits in `cols`.

    Macro (per-bit) F1 is not reported: in the tail most bits have ~1 positive
    across the 3000 test molecules, so per-bit F1 is 0-or-1 noise.
    """
    y, q = Y[:, cols], P[:, cols]
    pred = q >= 0.5
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else float("nan")
    return prec, rec, f1, tp, fp, fn


print("=== mass bias and P/R/F1 by bit index (q = predicted mass, p = true mass) ===")
print(f"{'bitrange':<13} {'true_rate':>10} {'q/p':>7} {'95% CI':>16} "
      f"{'prec':>6} {'rec':>6} {'F1':>6} {'FN':>6} {'FP':>6} {'AUC':>7}")
ratio_pt, ratio_lo, ratio_hi, true_rate, aucs_b, f1_b = [], [], [], [], [], []
for b in bins:
    y, q = Y[:, b], P[:, b]
    qs, ys = q.sum(1), y.sum(1)          # per-molecule sums -> cheap bootstrap
    s = rng.integers(0, n, (BOOT, n))
    r = qs[s].sum(1) / np.maximum(ys[s].sum(1), 1)
    pt = qs.sum() / ys.sum()
    lo, hi = np.percentile(r, [2.5, 97.5])
    prec, rec, f1, tp, fp, fn = prf(b)
    auc = per_bit_auc(b)
    ratio_pt.append(pt); ratio_lo.append(lo); ratio_hi.append(hi)
    true_rate.append(y.mean()); aucs_b.append(auc); f1_b.append(f1)
    print(f"{f'{b[0]}-{b[-1]}':<13} {y.mean():>10.5f} {pt:>7.4f} "
          f"[{lo:.4f},{hi:.4f}] {prec:>6.4f} {rec:>6.4f} {f1:>6.4f} "
          f"{fn:>6d} {fp:>6d} {auc:>7.4f}")

# Does the bias itself trend with index? Per-bit relative bias, bits with enough signal.
ok = Y.sum(0) >= 5
rel = (P.mean(0)[ok] - Y.mean(0)[ok]) / Y.mean(0)[ok]
rho, pv = spearmanr(np.arange(K)[ok], rel)
print(f"\nper-bit relative bias (q_j-p_j)/p_j vs bit index, {ok.sum()} bits with >=5 positives:")
print(f"  Spearman rho={rho:+.4f}  p={pv:.3g}   mean={rel.mean():+.4f}  median={np.median(rel):+.4f}")

print("\n=== micro P/R/F1 @0.5, stratified by redundancy ===")
print(f"{'bitrange':<13} {'%impl':>6} | {'P':>6} {'R':>6} {'F1':>6} (implied) "
      f"| {'P':>6} {'R':>6} {'F1':>6} {'npos':>6} (independent)")
f1_i, f1_n, rec_i, rec_n, centers = [], [], [], [], []
for b in np.array_split(np.arange(K), 8):
    out = []
    for cols in (b[implied[b]], b[~implied[b]]):
        out.append(prf(cols) if len(cols) else (float("nan"),) * 3 + (0, 0, 0))
    (pi, ri, fi, *_), (pn, rn, fn_, tpn, fpn, fnn) = out
    f1_i.append(fi); f1_n.append(fn_); rec_i.append(ri); rec_n.append(rn)
    centers.append(b.mean())
    print(f"{f'{b[0]}-{b[-1]}':<13} {100*implied[b].mean():>5.1f}% | "
          f"{pi:>6.4f} {ri:>6.4f} {fi:>6.4f}            | "
          f"{pn:>6.4f} {rn:>6.4f} {fn_:>6.4f} {tpn+fnn:>6d}")

print("\n=== reliability (is over-confidence index-dependent?) ===")
edges = [0.5, 0.7, 0.9, 0.99, 0.999, 1.0001]
groups = {
    "0-1023 (common)": np.arange(0, 1024),
    "1024-8191": np.arange(1024, 8192),
    "8192-16383 (rarest)": np.arange(8192, K),
    "8192-16383 indep": np.arange(8192, K)[~implied[8192:K]],
}
rel_curves = {}
for name, cols in groups.items():
    q, y = P[:, cols].ravel(), Y[:, cols].ravel()
    xs, ys_ = [], []
    print(f"\n{name}  ({len(cols)} bits)")
    for i in range(len(edges) - 1):
        s = (q >= edges[i]) & (q < edges[i + 1])
        if not s.sum():
            continue
        print(f"  pred [{edges[i]:.3g},{edges[i+1]:.3g}): n={s.sum():>7d} "
              f"mean_pred={q[s].mean():.4f} empirical={y[s].mean():.4f}")
        xs.append(q[s].mean()); ys_.append(y[s].mean())
    miss = ((y == 1) & (q < 0.5)).sum() / y.sum()
    print(f"  missed positives (y=1, q<0.5): {miss:.4f}")
    rel_curves[name] = (xs, ys_)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
c = [b.mean() for b in bins]
ax[0].axhline(1.0, color="k", lw=0.8, ls="--")
ax[0].errorbar(c, ratio_pt,
               yerr=[np.array(ratio_pt) - ratio_lo, np.array(ratio_hi) - ratio_pt],
               fmt="o-", ms=4, capsize=3)
ax[0].set_xlabel("bit index (rarer $\\rightarrow$)"); ax[0].set_ylabel("predicted mass / true mass")
ax[0].set_title("Mass bias is flat and slightly < 1")

ax[1].plot(centers, f1_i, "o-", color="C0", label="implied, F1")
ax[1].plot(centers, f1_n, "s-", color="C1", label="independent, F1")
ax[1].plot(centers, rec_i, "o--", color="C0", alpha=0.4, lw=1, label="implied, recall")
ax[1].plot(centers, rec_n, "s--", color="C1", alpha=0.4, lw=1, label="independent, recall")
ax[1].set_xlabel("bit index (rarer $\\rightarrow$)"); ax[1].set_ylabel("micro F1 / recall @ 0.5")
ax[1].set_ylim(0.8, 1.0); ax[1].legend(fontsize=7); ax[1].set_title("F1 flat within each stratum")

ax[2].plot([0.5, 1], [0.5, 1], "k--", lw=0.8)
for name, (xs, ys_) in rel_curves.items():
    ax[2].plot(xs, ys_, "o-", ms=4, label=name)
ax[2].set_xlabel("mean predicted prob"); ax[2].set_ylabel("empirical rate")
ax[2].legend(fontsize=7); ax[2].set_title("Same over-confidence at every index")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140)
print(f"\nwrote {OUT_PNG}")
