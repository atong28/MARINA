"""Plot the per-bit entropy distribution of the 16384-bit sherlock fingerprint.

Reads the npz produced by bit_entropy.py. Emits a 3-panel PNG plus a CSV
table-view twin.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))

# Design tokens (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
CTX = "#c3c2b7"  # de-emphasis gray for the context series

d = np.load(os.path.join(HERE, "bit_entropy.npz"))
N = int(d["n_retrieval"])
ent = d["sel_entropy"]
counts = d["sel_counts"]
cand_ent = d["cand_entropy"]

cutoff = ent.min()
median = np.median(ent)
n_bits = ent.size

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": AXIS,
})


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8, labelsize=9)


fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

# ---- Panel A: distribution of per-bit entropy (the direct answer) ----
ax = axes[0]
bins = np.logspace(np.log10(cutoff * 0.97), 0, 61)
ax.hist(ent, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.6, zorder=2)
ax.set_xscale("log")
ax.axvline(median, color=INK, linewidth=1.4, zorder=3)
ax.annotate(
    f"median  H = {median:.4f}\n(present in {int(np.median(counts)):,} / {N:,} mols)",
    xy=(median, ax.get_ylim()[1] * 0.92), xytext=(8, 0), textcoords="offset points",
    ha="left", va="top", fontsize=9, color=INK,
)
ax.set_xlabel("per-bit entropy  H(p)   [bits, log scale]")
ax.set_ylabel("number of bits")
ax.set_title("A.  Distribution over the 16,384 bits", fontsize=11, color=INK,
             loc="left", pad=10, weight="bold")
style(ax)

# ---- Panel B: entropy vs bit index (index IS the entropy rank) ----
ax = axes[1]
ax.plot(np.arange(n_bits), ent, color=BLUE, linewidth=2.0, zorder=2)
ax.set_yscale("log")
ax.set_xlim(0, n_bits)
ax.annotate(f"bit 0:  H = {ent[0]:.4f}", xy=(0, ent[0]), xytext=(14, -4),
            textcoords="offset points", fontsize=9, color=INK, va="top")
ax.annotate(f"bit {n_bits-1}:  H = {cutoff:.5f}\n(selection cutoff)",
            xy=(n_bits, cutoff), xytext=(-8, 14), textcoords="offset points",
            fontsize=9, color=INK, ha="right", va="bottom")
ax.set_xlabel("bit index  (= entropy rank)")
ax.set_ylabel("per-bit entropy  H(p)   [bits, log scale]")
ax.set_title("B.  Entropy decays ~2 orders over the fingerprint", fontsize=11,
             color=INK, loc="left", pad=10, weight="bold")
style(ax)

# ---- Panel C: where the cutoff falls in the full candidate pool ----
ax = axes[2]
cb = np.logspace(np.log10(cand_ent.min()), 0, 71)
cb = np.unique(np.concatenate([cb, [cutoff]]))  # exact bin edge at the threshold
h, edges = np.histogram(cand_ent, bins=cb)
sel_mask = edges[:-1] >= cutoff
ax.bar(edges[:-1][~sel_mask], h[~sel_mask], width=np.diff(edges)[~sel_mask],
       align="edge", color=CTX, edgecolor=SURFACE, linewidth=0.4, zorder=2,
       label=f"not selected ({cand_ent.size - n_bits:,})")
ax.bar(edges[:-1][sel_mask], h[sel_mask], width=np.diff(edges)[sel_mask],
       align="edge", color=BLUE, edgecolor=SURFACE, linewidth=0.4, zorder=3,
       label=f"selected ({n_bits:,})")
ax.axvline(cutoff, color=INK, linewidth=1.4, zorder=4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.annotate(f"cutoff H = {cutoff:.5f}\n(present in {counts.min():,} mols)",
            xy=(cutoff, 0.62), xycoords=("data", "axes fraction"),
            xytext=(-10, 0), textcoords="offset points", ha="right",
            va="center", fontsize=9, color=INK)
ax.set_xlabel("per-bit entropy  H(p)   [bits, log scale]")
ax.set_ylabel("number of candidate features   [log scale]")
ax.set_title("C.  The 16,384 are the top 0.2% of 7.5M candidates", fontsize=11,
             color=INK, loc="left", pad=10, weight="bold")
leg = ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK2)
style(ax)
for a in axes:
    a.xaxis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=20))
    a.xaxis.set_minor_formatter(NullFormatter())

fig.suptitle(
    "Per-bit entropy of the 16,384-bit sherlock fingerprint  ·  "
    f"binary entropy over the {N:,}-molecule retrieval set",
    fontsize=13, color=INK, x=0.008, ha="left", y=0.985, weight="bold",
)
fig.text(0.008, 0.925,
         f"Total marginal entropy {ent.sum():.0f} bits across 16,384 bits "
         f"({ent.sum()/n_bits*100:.1f}% of nominal capacity)  ·  "
         f"median bit {median:.4f} bits  ·  {int((ent > 0.5).sum())} bits above 0.5",
         fontsize=10, color=INK2, ha="left")

fig.tight_layout(rect=[0, 0, 1, 0.90])
out = os.path.join(HERE, "bit_entropy.png")
fig.savefig(out, dpi=200)
print(f"wrote {out}")

# ---- table-view twin ----
csv = os.path.join(HERE, "bit_entropy_summary.csv")
qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
with open(csv, "w") as f:
    f.write("quantile,entropy_bits,presence_freq,n_molecules\n")
    for q in qs:
        e_q = np.percentile(ent, q)
        c_q = np.percentile(counts, q)
        f.write(f"p{q},{e_q:.6f},{c_q/N:.6f},{int(c_q)}\n")
print(f"wrote {csv}")
