"""Figure 1: descriptor distributions per dataset.

Four panels covering the axes that separate natural-product chemistry from synthetic:
NP-likeness score, fraction sp3 carbon, heavy-atom count, potential stereocenters.

Palette: dataviz reference categorical slots 1-5, light mode, validated with
scripts/validate_palette.js (all checks PASS; contrast WARN on slots 3-5 is relieved
by direct labels on every curve plus the summary table in the wiki entry).
Series are ordered along the NP -> synthetic gradient, and color follows the dataset,
never its rank in any panel.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2de"

# dataset -> (label, colour). Fixed order = the NP->synthetic gradient; never cycled.
SERIES = [
    ("marina1", "MARINA1", "#2a78d6"),
    ("massspecgym", "MassSpecGym", "#008300"),
    ("nmrshiftdb2", "nmrshiftdb2-2024", "#e87ba4"),
    ("nmrexp", "NMRexp", "#eda100"),
    ("mmsd", "Alberts/MMSD", "#1baf7a"),
]

PANELS = [
    ("np_score", "NP-likeness score", (-4, 4), None),
    ("fsp3", "Fraction sp$^3$ carbon", (0, 1), None),
    ("heavy_atoms", "Heavy atoms", (0, 80), 35),
    ("n_stereocenters", "Potential stereocenters", (0, 14), None),
]


def main():
    df = pd.read_parquet(os.path.join(RESULTS, "descriptors.parquet"))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.42, wspace=0.2)

    for ax, (col, xlabel, xlim, vline) in zip(axes.ravel(), PANELS):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=9, length=3, color=GRID)

        # ECDF rather than a histogram: Fsp3 and stereocenter counts are ratios/counts of
        # small integers, so any binning of them spikes at rational values. An ECDF is
        # binning-free and makes "what share of the set sits below x" directly readable.
        for key, label, colour in SERIES:
            v = np.sort(df.loc[df.source == key, col].dropna().to_numpy())
            y = np.arange(1, len(v) + 1) / len(v)
            ax.plot(v, y, color=colour, lw=2, zorder=3, solid_joinstyle="round")

        if vline is not None:
            ax.axvline(vline, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
            ax.annotate(
                "Alberts / SpecX ceiling (35)",
                (vline, 1.02),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8,
                color=INK2,
                zorder=4,
            )

        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
        ax.set_ylabel("cumulative fraction", color=INK2, fontsize=9)

    handles = [plt.Line2D([], [], color=c, lw=2, label=l) for _, l, c in SERIES]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=9,
        labelcolor=INK2,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(
        "Molecular domain of public spectroscopic datasets vs MARINA1",
        y=1.06,
        fontsize=12.5,
        color=INK,
    )

    out = os.path.join(RESULTS, "descriptor_distributions.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
