"""Figure 2: UMAP of all five sets, faceted one panel per dataset.

Faceted rather than five colours in one scatter, for two reasons: 75k overplotted points
in one axes hides whichever set is drawn first, and the dataviz all-pairs CVD gate caps a
scatter at four categorical slots. Each panel draws the other four sets as grey context so
every facet is read against the same backdrop.

Rows are the two representations. ECFP4 is vocabulary-free; sFP is MARINA's own 16,384-bit
vocabulary, which silently drops substructures it has no bit for -- so divergence between
the rows is the signal, and the sFP row cannot be read as evidence of NP-likeness.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
CONTEXT = "#d8d7d2"

SERIES = [
    ("marina1", "MARINA1", "#2a78d6"),
    ("massspecgym", "MassSpecGym", "#008300"),
    ("nmrshiftdb2", "nmrshiftdb2-2024", "#e87ba4"),
    ("nmrexp", "NMRexp", "#eda100"),
    ("mmsd", "Alberts/MMSD", "#1baf7a"),
]

REPS = [("ecfp4", "ECFP4  (vocabulary-free)"), ("sfp", "Sherlock FP  (NP-mined vocabulary)")]


def main():
    reps = [(r, t) for r, t in REPS if os.path.exists(os.path.join(RESULTS, f"umap_{r}.parquet"))]
    if not reps:
        raise SystemExit("no umap parquet files yet")

    fig, axes = plt.subplots(
        len(reps), len(SERIES), figsize=(3.1 * len(SERIES), 3.3 * len(reps)), facecolor=SURFACE
    )
    axes = axes.reshape(len(reps), len(SERIES))

    for row, (rep, rep_title) in enumerate(reps):
        df = pd.read_parquet(os.path.join(RESULTS, f"umap_{rep}.parquet"))
        # shared limits from the central 99% -- a handful of far outliers otherwise squeeze
        # every panel's core structure into a few pixels
        xlim = (df.x.quantile(0.005), df.x.quantile(0.995))
        ylim = (df.y.quantile(0.005), df.y.quantile(0.995))
        for col, (key, label, colour) in enumerate(SERIES):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            for s in ax.spines.values():
                s.set_color(CONTEXT)
            ax.set_xticks([])
            ax.set_yticks([])

            other = df[df.source != key]
            focal = df[df.source == key]
            ax.scatter(other.x, other.y, s=1.0, c=CONTEXT, lw=0, rasterized=True, zorder=1)
            ax.scatter(focal.x, focal.y, s=1.3, c=colour, lw=0, rasterized=True, zorder=2)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            if row == 0:
                ax.set_title(label, fontsize=10, color=INK, pad=6)
            if col == 0:
                ax.set_ylabel(rep_title, fontsize=9.5, color=INK2, labelpad=8)

    fig.suptitle(
        "Chemical space of public spectroscopic datasets vs MARINA1  (UMAP, Tanimoto/Jaccard)",
        fontsize=12.5,
        color=INK,
        y=0.99,
    )
    fig.text(
        0.5,
        0.005,
        "Grey = the other four datasets. Axes are arbitrary: UMAP coordinates are not a metric space.",
        ha="center",
        fontsize=8.5,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    out = os.path.join(RESULTS, "umap_facets.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
