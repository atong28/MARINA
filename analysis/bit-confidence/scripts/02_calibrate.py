#!/usr/bin/env python3
"""Does a predicted bit probability mean what it says, and can we make it?

The website wants to tell a user "this substructure is 93% likely to be present".
The raw sigmoid does not support that claim: bit-calibration.tex measured
empirical presence of 0.87 in the [0.9, 0.99) bucket on seed-1. This script
re-measures on the deployed checkpoint and fits an isotonic recalibration.

Two things make this cheap and exact:

  * preds.npz stores probabilities as float16, so the predictions take at most
    65,536 distinct values. Aggregating (count, n_positive) per distinct value
    via bincount is a lossless summary of all n*16384 (molecule, bit) pairs --
    no dense 387M-element array, and isotonic then fits ~15k weighted points
    instead of 387M.
  * Isotonic on those weighted points is identical to isotonic on the full pool,
    because the fit only ever sees x-ties as aggregates anyway.

The fit/eval split is BY MOLECULE, not by (molecule, bit) pair. Bits within one
molecule are heavily dependent -- 10,608 of 16,384 are logically implied by
another bit -- so a pair-level split would leak the answer across the split and
report a calibration error that is far too optimistic.

    pixi run python scripts/02_calibrate.py --preds results/preds_seed2.npz
"""
import argparse
import json
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default=os.path.join(HERE, "results", "preds_seed2.npz"))
ap.add_argument("--tag", default="seed2")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

OUT_JSON = os.path.join(HERE, "results", f"calibration_{args.tag}.json")
OUT_NPZ = os.path.join(HERE, "results", f"calibration_{args.tag}.npz")

# float16 bit-pattern b -> the probability it encodes. Bin index IS the raw
# uint16 view of the stored value, so no quantisation is introduced here.
BINS = 1 << 16
VALS = np.arange(BINS, dtype=np.uint16).view(np.float16).astype(np.float64)
VALID = np.isfinite(VALS) & (VALS >= 0.0) & (VALS <= 1.0)

d = np.load(args.preds, allow_pickle=True)
P = d["probs"]                      # (n, K) float16
idx, ptr = d["fp_idx"], d["fp_ptr"]
n, K = P.shape
print(f"ckpt {d['ckpt']}   n={n} molecules   K={K} bits   ({n * K / 1e6:.0f}M pairs)")

rng = np.random.default_rng(args.seed)
perm = rng.permutation(n)
fit_mols, eval_mols = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
print(f"split by molecule: {len(fit_mols)} fit / {len(eval_mols)} eval")


def tabulate(mols):
    """Exact (count, n_positive) per distinct float16 prediction value."""
    cnt = np.zeros(BINS, dtype=np.int64)
    pos = np.zeros(BINS, dtype=np.int64)
    for i in mols:
        row = P[i].view(np.uint16)
        cnt += np.bincount(row, minlength=BINS)
        true_bits = idx[ptr[i]:ptr[i + 1]]
        if len(true_bits):
            pos += np.bincount(row[true_bits], minlength=BINS)
    return cnt, pos


cnt_fit, pos_fit = tabulate(fit_mols)
cnt_ev, pos_ev = tabulate(eval_mols)

# Sanity: the tabulation must account for every pair and every true bit.
assert cnt_fit.sum() == len(fit_mols) * K, "lost pairs in fit tabulation"
assert cnt_ev.sum() == len(eval_mols) * K, "lost pairs in eval tabulation"
print(f"true bits/molecule: {(pos_fit.sum() + pos_ev.sum()) / n:.1f}")

m = VALID & (cnt_fit > 0)
x_fit, w_fit, y_fit = VALS[m], cnt_fit[m].astype(np.float64), (pos_fit[m] / cnt_fit[m])
print(f"distinct predicted values in fit half: {m.sum()}")

iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
iso.fit(x_fit, y_fit, sample_weight=w_fit)


def metrics(cnt, pos, transform):
    """ECE / Brier / log-loss over the pairs summarised by (cnt, pos)."""
    s = VALID & (cnt > 0)
    q = transform(VALS[s])
    c, p = cnt[s].astype(np.float64), pos[s].astype(np.float64)
    total = c.sum()
    emp = p / c
    ece = float((c * np.abs(q - emp)).sum() / total)
    # sum over pairs of (q-y)^2 = c*q^2 - 2*q*p + p   (since y is 0/1)
    brier = float((c * q * q - 2 * q * p + p).sum() / total)
    qc = np.clip(q, 1e-7, 1 - 1e-7)
    nll = float(-(p * np.log(qc) + (c - p) * np.log1p(-qc)).sum() / total)
    return {"ece": ece, "brier": brier, "nll": nll}


raw = metrics(cnt_ev, pos_ev, lambda v: v)
cal = metrics(cnt_ev, pos_ev, iso.predict)
print("\n=== held-out eval half, all pairs ===")
for name, mm in (("raw sigmoid", raw), ("isotonic", cal)):
    print(f"  {name:<12} ECE={mm['ece']:.6f}  Brier={mm['brier']:.6f}  NLL={mm['nll']:.6f}")


def reliability(cnt, pos, transform, edges):
    """Empirical presence rate per confidence bucket, on transformed scores."""
    s = VALID & (cnt > 0)
    q, c, p = transform(VALS[s]), cnt[s].astype(np.float64), pos[s].astype(np.float64)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (q >= lo) & (q < hi)
        if not c[b].sum():
            continue
        rows.append({"lo": lo, "hi": hi, "n": int(c[b].sum()),
                     "mean_pred": float((c[b] * q[b]).sum() / c[b].sum()),
                     "empirical": float(p[b].sum() / c[b].sum())})
    return rows


EDGES = [0.5, 0.7, 0.9, 0.99, 0.999, 1.0001]
print("\n=== reliability on held-out half (the claim the UI would make) ===")
for name, tr in (("raw sigmoid", lambda v: v), ("isotonic", iso.predict)):
    print(f"\n{name}")
    print(f"  {'bucket':<16} {'n':>10} {'mean_pred':>10} {'empirical':>10} {'gap':>8}")
    for r in reliability(cnt_ev, pos_ev, tr, EDGES):
        gap = r["mean_pred"] - r["empirical"]
        print(f"  [{r['lo']:.3g},{r['hi']:.3g}){'':<5} {r['n']:>10d} "
              f"{r['mean_pred']:>10.4f} {r['empirical']:>10.4f} {gap:>+8.4f}")

# Sharpness: how many bits per molecule land in a band worth showing a user?
s = VALID & (cnt_ev > 0)
qe, ce = VALS[s], cnt_ev[s].astype(np.float64)
ne = len(eval_mols)
print("\n=== sharpness: predicted bits per molecule by confidence band ===")
for lo, hi in [(0.0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.0001)]:
    b = (qe >= lo) & (qe < hi)
    print(f"  [{lo:.2g},{hi:.2g}){'':<3} {ce[b].sum() / ne:>10.2f} bits/molecule")

# Export curve: resample isotonic on a grid dense where the UI actually reads it.
grid = np.unique(np.concatenate([
    np.linspace(0.0, 0.1, 101), np.linspace(0.1, 0.9, 81), np.linspace(0.9, 1.0, 201),
]))
curve = iso.predict(grid)
curve = np.maximum.accumulate(curve)        # guard float noise; must stay monotone
np.savez_compressed(OUT_NPZ, grid=grid, curve=curve, ckpt=str(d["ckpt"]))
with open(OUT_JSON, "w") as f:
    json.dump({
        "ckpt": str(d["ckpt"]),
        "n_molecules": int(n), "n_bits": int(K),
        "fit_molecules": int(len(fit_mols)), "eval_molecules": int(len(eval_mols)),
        "eval_raw": raw, "eval_isotonic": cal,
        "reliability_raw": reliability(cnt_ev, pos_ev, lambda v: v, EDGES),
        "reliability_isotonic": reliability(cnt_ev, pos_ev, iso.predict, EDGES),
        "grid": grid.tolist(), "curve": curve.tolist(),
    }, f, indent=2)
print(f"\nwrote {OUT_JSON}\nwrote {OUT_NPZ}")
