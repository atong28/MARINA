"""Classify MARINA1's *stored* HSQC by provenance, directly.

hsqc_provenance.py joined MARINA1 back to SPECTRE, but a canonical SMILES can
carry both an experimental and an ACD entry upstream, and MARINA1 kept only one
of them -- so the join mislabels ~half of that group. MARINA1's own tensors are
the ground truth, and they are self-identifying:

  third column is a signed float volume      -> ACD/Labs simulation
  third column is exactly +/-1, shifts rounded -> JEOL/CH-NMR-NP experimental
  third column is exactly +/-1, full precision -> Mnova simulation

The rounding split is the discriminator established in discriminate_hsqc.py:
literature-transcribed peak lists are quoted to 1-2 dp, predictors emit full
float precision. "Rounded" here = every 13C shift in the molecule is exact at
<=2 dp and every 1H shift at <=3 dp.

Writes results/marina1_hsqc_classified.json + .parquet.
"""
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
MARINA1 = os.path.join(DATA_ROOT, "Datasets/MARINA1")
ACD, JEOL, MNOVA = "acd_simulated", "experimental_jeol", "mnova_simulated"


def rounded(vals, dp):
    """True if every value is exactly representable at <= dp decimals."""
    return bool(np.all(np.abs(vals - np.round(vals, dp)) < 1e-9))


def classify(arr):
    if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
        return "degenerate"
    if not np.all(np.isin(arr[:, 2], (-1.0, 1.0))):
        return ACD
    return JEOL if (rounded(arr[:, 0], 2) and rounded(arr[:, 1], 3)) else MNOVA


m1 = pickle.load(open(f"{MARINA1}/index.pkl", "rb"))
rows = []
for split in ("train", "val", "test"):
    tbl = pq.read_table(f"{MARINA1}/arrow/{split}/HSQC_NMR.parquet")
    idxs = tbl.column("idx").to_pylist()
    data = tbl.column("data")
    shapes = tbl.column("shape")
    for i in tqdm(range(len(idxs)), desc=f"classify/{split}"):
        arr = np.asarray(data[i].as_py(), dtype=np.float64)
        shp = tuple(shapes[i].as_py())
        if len(shp) != 2:
            continue
        arr = arr.reshape(shp)
        idx = idxs[i]
        e = m1.get(idx, {})
        rows.append({
            "idx": idx,
            "split": split,
            "smiles": e.get("smiles"),
            "mw": e.get("mw"),
            "n_peaks": int(arr.shape[0]),
            "provenance": classify(arr),
        })

df = pd.DataFrame(rows)
counts = df.provenance.value_counts()
print("\nMARINA1 stored HSQC by provenance:")
for k, v in counts.items():
    print(f"  {k:20s} {v:>8,}  ({100*v/len(df):5.2f}%)")

by_split = {p: {s: int(n) for s, n in g.split.value_counts().items()}
            for p, g in df.groupby("provenance")}
peaks = {p: {"mean": round(float(g.n_peaks.mean()), 2),
             "median": float(g.n_peaks.median())}
         for p, g in df.groupby("provenance")}

summary = {
    "total_hsqc_molecules": int(len(df)),
    "by_provenance": {k: int(v) for k, v in counts.items()},
    "by_provenance_pct": {k: round(100 * v / len(df), 2)
                          for k, v in counts.items()},
    "by_provenance_split": by_split,
    "peaks_per_molecule": peaks,
    "experimental_total": int(counts.get(JEOL, 0)),
    "simulated_total": int(counts.get(ACD, 0) + counts.get(MNOVA, 0)),
}
with open(RESULTS / "marina1_hsqc_classified.json", "w") as f:
    json.dump(summary, f, indent=2)
df.to_parquet(RESULTS / "marina1_hsqc_classified.parquet", index=False)
print("\n" + json.dumps(summary, indent=2))
