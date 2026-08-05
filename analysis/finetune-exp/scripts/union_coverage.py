"""Union of GNPS + MassSpecGym + MassBank against MARINA1.

The three libraries overlap heavily (MassSpecGym is largely built from GNPS and
MassBank), so their molecule counts cannot be added. This computes the real
union -- the total set of MARINA1 molecules whose ICEBERG-simulated MS/MS could
be replaced by, or augmented with, experimental spectra.

Writes results/union_coverage.json.
"""
import json
import os
import pickle
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
MARINA1_INDEX = os.path.join(DATA_ROOT, "Datasets/MARINA1/index.pkl")
SRC = {
    "gnps": RESULTS / "gnps_matched_molecules.parquet",
    "massspecgym": RESULTS / "msms_matched_molecules.parquet",
    "massbank": RESULTS / "massbank_matched_molecules.parquet",
}

m1 = pickle.load(open(MARINA1_INDEX, "rb"))
m1_split = {e["smiles"]: e["split"] for e in m1.values()}

sets, neg_sets, frames = {}, {}, {}
for name, path in SRC.items():
    df = pd.read_parquet(path)
    frames[name] = df
    sets[name] = set(df["smiles"])
    if "has_negative" in df.columns:
        neg_sets[name] = set(df[df["has_negative"]]["smiles"])
    else:
        neg_sets[name] = set()  # MassSpecGym is positive-only
    print(f"{name:12s} {len(sets[name]):>7,} molecules")

union = set().union(*sets.values())
union_neg = set().union(*neg_sets.values())

pairwise = {}
names = list(sets)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        inter = len(sets[a] & sets[b])
        pairwise[f"{a}&{b}"] = {
            "intersection": inter,
            "pct_of_" + a: round(100 * inter / len(sets[a]), 1),
            "pct_of_" + b: round(100 * inter / len(sets[b]), 1),
        }

# marginal contribution: molecules only this source provides
unique_to = {n: len(sets[n] - set().union(
    *[s for m, s in sets.items() if m != n])) for n in names}

split_counts = pd.Series([m1_split[s] for s in union]).value_counts().to_dict()
split_counts_neg = pd.Series(
    [m1_split[s] for s in union_neg]).value_counts().to_dict()

summary = {
    "per_source_molecules": {k: len(v) for k, v in sets.items()},
    "union_molecules": len(union),
    "union_pct_of_marina1": round(100 * len(union) / len(m1_split), 3),
    "union_by_marina1_split": {k: int(v) for k, v in split_counts.items()},
    "union_with_negative_mode": len(union_neg),
    "union_negative_by_marina1_split": {k: int(v)
                                        for k, v in split_counts_neg.items()},
    "pairwise_overlap": pairwise,
    "unique_contribution": unique_to,
    "naive_sum_would_be": sum(len(v) for v in sets.values()),
}

with open(RESULTS / "union_coverage.json", "w") as f:
    json.dump(summary, f, indent=2)

pd.DataFrame({"smiles": sorted(union)}).assign(
    split=lambda d: d.smiles.map(m1_split),
    has_negative=lambda d: d.smiles.isin(union_neg),
    in_gnps=lambda d: d.smiles.isin(sets["gnps"]),
    in_massspecgym=lambda d: d.smiles.isin(sets["massspecgym"]),
    in_massbank=lambda d: d.smiles.isin(sets["massbank"]),
).to_parquet(RESULTS / "union_matched_molecules.parquet", index=False)

print("\n" + json.dumps(summary, indent=2))
