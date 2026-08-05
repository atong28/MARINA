#!/usr/bin/env python3
"""Paired significance tests on the reweighting sweep.

Top-1 differences get an exact McNemar test (paired, same queries); rank-
distribution shifts get a paired bootstrap over queries.

    pixi run python scripts/04_analyze_ranks.py
"""
import os

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ranks = np.load(os.path.join(HERE, "results", "reweight_ranks.npz"))
base = ranks["uniform"]
K = len(base)
rng = np.random.default_rng(0)

print(f"n = {K} test molecules, baseline = uniform cosine\n")
print(f"{'scheme':>9s} {'top1':>7s} {'dtop1':>8s} {'McNemar p':>10s} "
      f"{'mean':>7s} {'p90':>6s} {'p99':>7s}  {'dmean 95% CI':>22s}")

for name in ranks.files:
    r = ranks[name]
    top1 = (r < 1).mean()
    # McNemar: queries where exactly one of the two schemes is correct
    b = int(((base < 1) & (r >= 1)).sum())    # baseline right, scheme wrong
    c = int(((base >= 1) & (r < 1)).sum())    # scheme right, baseline wrong
    p = binomtest(c, b + c, 0.5).pvalue if (b + c) else 1.0

    d = r.astype(np.float64) - base
    boot = np.array([d[rng.integers(0, K, K)].mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"{name:>9s} {top1:7.4f} {top1-(base<1).mean():+8.4f} {p:10.3g} "
          f"{r.mean():7.1f} {np.percentile(r,90):6.0f} {np.percentile(r,99):7.0f}"
          f"  [{lo:+8.1f}, {hi:+8.1f}]")

print("\n--- where does the mean-rank gain come from? (soft0.9 vs uniform) ---")
r = ranks["soft0.9"]
for lo_, hi_ in [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 10**9)]:
    m = (base >= lo_) & (base < hi_)
    if m.sum():
        print(f"  baseline rank [{lo_:>5d},{hi_:>6d}): n={m.sum():>4d}  "
              f"mean rank {base[m].mean():9.1f} -> {r[m].mean():9.1f}  "
              f"({(r[m] < base[m]).sum()} better, {(r[m] > base[m]).sum()} worse)")
