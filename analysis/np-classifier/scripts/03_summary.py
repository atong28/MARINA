"""
What the NPClassifier pass says about MARINA1's retrieval set.

    cd /home/user/atong/MARINA
    pixi run python3 analysis/np-classifier/scripts/03_summary.py

Writes results/summary.md and results/summary.json. The numbers that matter downstream
are the per-tier coverage and the class-tier size distribution: a tier where most
molecules are unlabelled, or where the median class has a handful of members, is not
usable as ground truth for scoring retrieval neighbours.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "results")
TIERS = ["pathway", "superclass", "class"]


def main() -> None:
    df = pd.read_parquet(os.path.join(OUT_DIR, "npclassifier.parquet"))
    n = len(df)

    lines = [f"# NPClassifier over the MARINA1 retrieval set\n",
             f"Molecules classified: **{n:,}**\n",
             "## Coverage\n",
             "| tier | distinct labels | molecules labelled | multi-label |",
             "|---|---|---|---|"]

    summary = {"n": n, "tiers": {}}
    for t in TIERS:
        counts = Counter(lab for labs in df[t] for lab in labs)
        labelled = int((df[t].map(len) > 0).sum())
        multi = int((df[t].map(len) > 1).sum())
        sizes = sorted(counts.values())
        summary["tiers"][t] = {
            "distinct": len(counts),
            "labelled": labelled,
            "labelled_pct": round(100 * labelled / n, 2),
            "multi_label": multi,
            "median_label_size": sizes[len(sizes) // 2] if sizes else 0,
            "singleton_labels": sum(1 for v in sizes if v == 1),
            "top": counts.most_common(15),
        }
        lines.append(f"| {t} | {len(counts):,} | {labelled:,} ({100*labelled/n:.1f}%) "
                     f"| {multi:,} |")

    unlabelled = int((df[TIERS].map(len).sum(axis=1) == 0).sum())
    glyco = int(df["isglycoside"].sum())
    summary["unlabelled_all_tiers"] = unlabelled
    summary["n_glycoside"] = glyco

    lines += [
        "",
        f"Unlabelled at every tier: **{unlabelled:,}** ({100*unlabelled/n:.2f}%) — these "
        "cannot serve as ground truth for neighbour scoring.",
        f"Glycosides: **{glyco:,}** ({100*glyco/n:.1f}%).",
        "",
        "## Label sizes\n",
        "| tier | median members | singleton labels |",
        "|---|---|---|",
    ]
    for t in TIERS:
        s = summary["tiers"][t]
        lines.append(f"| {t} | {s['median_label_size']:,} | {s['singleton_labels']:,} |")

    for t in TIERS:
        lines += [f"\n## Most common {t}\n", "| label | molecules | share |", "|---|---|---|"]
        for lab, c in summary["tiers"][t]["top"]:
            lines.append(f"| {lab} | {c:,} | {100*c/n:.2f}% |")

    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n".join(lines[:24]))
    print(f"\nwrote {OUT_DIR}/summary.md and summary.json")


if __name__ == "__main__":
    main()
