"""MARINA4 = MARINA1 with 13C/1H collapsed to symmetry-unique shifts.

MARINA1's Mnova NMR is per-atom: a shift appears once per nucleus, so symmetry-
equivalent carbons contribute duplicate entries. `benchmark.pkl` -- and any real
spectrum -- is per-peak, because equivalent nuclei are magnetically degenerate and
produce one line. MARINA1 therefore trains on one convention and is evaluated on
another, worth ~9.4% excess 13C entries dataset-wide.

MARINA4 removes that mismatch by merging shifts that fall within a tolerance, which
is what a spectrometer does. Everything else -- molecules, idx, splits, HSQC, MS/MS,
FragIdx, retrieval set, fingerprint vocabulary -- is inherited unchanged from
MARINA1, so MARINA1-vs-MARINA4 isolates the peak convention and nothing else.

HSQC is deliberately untouched: its rows are (13C, 1H, sign) triples, and collapsing
them would merge distinct correlations rather than degenerate lines.
"""
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_ROOT = Path(os.environ.get("MARINA_DATA_ROOT", "/home/user/atong"))
DATASETS = DATA_ROOT / "Datasets"
M1, M4 = DATASETS / "MARINA1", DATASETS / "MARINA4"
RESULTS = Path(__file__).resolve().parent.parent / "results"
SPLITS = ("train", "val", "test")
COPY_FILES = ["retrieval.pkl", "metadata.json", "count_hashes_under_radius_6.pkl"]
COPY_DIRS = ["RankingEntropy"]
PASS_THROUGH = ["HSQC_NMR.parquet", "MassSpec.parquet", "FragIdx.parquet"]

# Collapse EXACT duplicates only. Mnova emits bit-identical shifts for symmetry-
# equivalent nuclei (e.g. 45.007256, 45.007256), so exact matching captures the
# degeneracy precisely. A physically-motivated tolerance was tried first and
# rejected: benchmark.pkl resolves 13C peaks as close as 0.01 ppm, with 1.0% of its
# adjacent spacings under 0.02 ppm, so a 0.02 ppm merge destroys peaks that real
# spectra do resolve. This value only absorbs float32 round-trip noise (~7.6e-6 at
# 100 ppm) and sits two orders of magnitude below real resolution, keeping the
# transformation a pure symmetry collapse and the experiment single-variable.
TOL = {"C_NMR": 1e-4, "H_NMR": 1e-4}


def collapse(values, tol):
    out = []
    for v in sorted(float(x) for x in values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


if M4.exists():
    shutil.rmtree(M4)
M4.mkdir(parents=True)

stats = {}
for split in SPLITS:
    (M4 / "arrow" / split).mkdir(parents=True, exist_ok=True)
    for fname in PASS_THROUGH:
        shutil.copy2(M1 / "arrow" / split / fname, M4 / "arrow" / split / fname)

    for mod, tol in TOL.items():
        src = M1 / "arrow" / split / f"{mod}.parquet"
        schema = pq.read_schema(src)
        t = pq.read_table(src).to_pydict()
        idxs, datas, shapes = [], [], []
        before = after = 0
        for i, d, s in zip(t["idx"], t["data"], t["shape"]):
            merged = collapse(d, tol)
            before += len(d)
            after += len(merged)
            idxs.append(i)
            datas.append(merged)
            shapes.append([len(merged)])
        pq.write_table(
            pa.table({"idx": pa.array(idxs, type=schema.field("idx").type),
                      "data": pa.array(datas, type=schema.field("data").type),
                      "shape": pa.array(shapes, type=schema.field("shape").type)},
                     schema=schema),
            M4 / "arrow" / split / f"{mod}.parquet")
        key = f"{split}/{mod}"
        stats[key] = {"rows": len(idxs), "entries_before": before, "entries_after": after,
                      "retained": round(after / before, 4)}
        print(f"  {key}: {before:,} -> {after:,} entries ({after/before:.1%})", flush=True)

shutil.copy2(M1 / "index.pkl", M4 / "index.pkl")   # flags unchanged: nothing became empty
for f in COPY_FILES:
    shutil.copy2(M1 / f, M4 / f)
for d in COPY_DIRS:
    shutil.copytree(M1 / d, M4 / d)

RESULTS.mkdir(exist_ok=True)
(RESULTS / "marina4_build.json").write_text(json.dumps(stats, indent=2))
print("\n" + json.dumps(stats, indent=2))
