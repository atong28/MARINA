#!/usr/bin/env python3
"""Figures for the bit-confidence write-up.

    pixi run python scripts/03_figures.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(HERE, "results", "calibration_seed2.json")))
OUT = os.path.join(HERE, "results", "bit_confidence.png")

grid, curve = np.array(d["grid"]), np.array(d["curve"])
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# (a) reliability before/after
for key, label, style in (("reliability_raw", "raw sigmoid", "o-"),
                          ("reliability_isotonic", "isotonic", "s-")):
    r = d[key]
    ax[0].plot([b["mean_pred"] for b in r], [b["empirical"] for b in r], style, ms=5, label=label)
ax[0].plot([0.5, 1], [0.5, 1], "k--", lw=0.8, label="perfect")
ax[0].set_xlabel("displayed probability")
ax[0].set_ylabel("fraction actually present")
ax[0].set_title("(a) Raw sigmoid overstates presence")
ax[0].legend(fontsize=8)

# (b) the correction curve itself
ax[1].plot(grid, curve, lw=1.8)
ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
ax[1].set_xlabel("raw sigmoid output")
ax[1].set_ylabel("calibrated probability")
ax[1].set_title("(b) Correction: down high, up low")
ax[1].annotate("over-confident", xy=(0.9, 0.77), xytext=(0.45, 0.9), fontsize=8,
               arrowprops=dict(arrowstyle="->", lw=0.8))
ax[1].annotate("under-confident", xy=(0.1, 0.21), xytext=(0.15, 0.5), fontsize=8,
               arrowprops=dict(arrowstyle="->", lw=0.8))

# (c) how many bits per molecule the UI can actually talk about
bands = [("<0.01", 16318.24), ("0.01-0.1", 5.10), ("0.1-0.5", 2.32),
         ("0.5-0.9", 1.98), (">=0.9", 56.36)]
names = [b[0] for b in bands]
vals = [b[1] for b in bands]
ax[2].bar(names, vals, color=["#bbb", "#7aa6c2", "#7aa6c2", "#e0a458", "#5b8c5a"])
ax[2].set_yscale("log")
ax[2].set_ylabel("bits per molecule (log)")
ax[2].set_xlabel("raw confidence band")
ax[2].set_title("(c) ~56 confident bits, ~10 uncertain")
for i, v in enumerate(vals):
    ax[2].text(i, v * 1.15, f"{v:.1f}" if v < 100 else f"{v:.0f}", ha="center", fontsize=8)

fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
