#!/usr/bin/env python3
"""Per-peak 1D-NMR collapse, applied dataset-wide.

MARINA's Mnova NMR is per-atom: a shift appears once per nucleus, so symmetry-
equivalent carbons contribute duplicate entries. Real spectra -- and the journal
benchmark -- are per-peak, because magnetically degenerate nuclei produce one
line. This stage removes that convention mismatch by merging shifts within
PEAK_COLLAPSE_TOL, on BOTH the training arrow dataset and the journal benchmark
pkl, using one collapse() function so the two sides stay identical.

13C (C_NMR) and 1H (H_NMR) only. HSQC is (13C,1H,sign) correlations, MassSpec is
m/z peaks, FragIdx is structural -- none are degenerate lines, so all are left
untouched.

Sources: analysis/marina23-build/scripts/09_build_marina4.py (arrow rewrite) and
scripts/benchmark/collapse_journal.py (journal pkl). Difference from 09: this
collapses the arrow dataset IN PLACE (config.DATA_DATASET) rather than materialising
a new dataset directory; everything else besides C_NMR/H_NMR is left as-is on disk.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[3]))
from config import DATA_DATASET, BENCH_JOURNAL, BENCH_JOURNAL_PREPARED, PEAK_COLLAPSE_TOL

SPLITS = ("train", "val", "test")
MODS = ("C_NMR", "H_NMR")  # 1D nuclei only; HSQC/MassSpec/FragIdx untouched


def collapse(values, tol):
    """Drop shifts within `tol` of a kept value. Verbatim from 09_build_marina4.py."""
    out = []
    for v in sorted(float(x) for x in values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def collapse_tensor(t):
    """(N,1) float32 shift tensor -> (M,1) with near-duplicates removed."""
    merged = collapse(t.view(-1).tolist(), PEAK_COLLAPSE_TOL)
    return torch.tensor([[v] for v in merged], dtype=torch.float32)


def collapse_dataset(root):
    """Collapse C_NMR/H_NMR parquet in place under {root}/arrow/{split}/."""
    root = Path(root)
    for split in SPLITS:
        for mod in MODS:
            src = root / "arrow" / split / f"{mod}.parquet"
            schema = pq.read_schema(src)
            t = pq.read_table(src).to_pydict()
            idxs, datas, shapes = [], [], []
            before = after = 0
            for i, d in zip(t["idx"], t["data"]):
                merged = collapse(d, PEAK_COLLAPSE_TOL)
                before += len(d)
                after += len(merged)
                idxs.append(i)
                datas.append(merged)
                shapes.append([len(merged)])
            tmp = src.with_suffix(".parquet.tmp")
            pq.write_table(
                pa.table({"idx": pa.array(idxs, type=schema.field("idx").type),
                          "data": pa.array(datas, type=schema.field("data").type),
                          "shape": pa.array(shapes, type=schema.field("shape").type)},
                         schema=schema),
                tmp)
            os.replace(tmp, src)  # atomic in-place swap
            print(f"  {split}/{mod}: {before:,} -> {after:,} entries "
                  f"({after / before:.1%})", flush=True)


def collapse_journal_pkl(path):
    """Collapse input['c_nmr']/input['h_nmr'] tensors; write *-collapsed.pkl."""
    path = Path(path)
    data = pickle.load(open(path, "rb"))
    before = {"c_nmr": 0, "h_nmr": 0}
    after = {"c_nmr": 0, "h_nmr": 0}
    shortened = 0
    for entry in data.values():
        inp = entry["input"]
        changed = False
        for mod in ("c_nmr", "h_nmr"):  # hsqc/mw/split/smiles untouched
            if mod not in inp or inp[mod].numel() == 0:
                continue
            n0 = inp[mod].shape[0]
            inp[mod] = collapse_tensor(inp[mod])
            n1 = inp[mod].shape[0]
            before[mod] += n0
            after[mod] += n1
            changed = changed or (n1 < n0)
        shortened += int(changed)

    out = path.with_name(path.stem + "-collapsed.pkl")
    with open(out, "wb") as f:
        pickle.dump(data, f)
    print(f"  {path.name}: {len(data)} entries, {shortened} shortened", flush=True)
    for mod in ("c_nmr", "h_nmr"):
        b, af = before[mod], after[mod]
        pct = 100.0 * af / b if b else float("nan")
        print(f"    {mod}: {b} -> {af} ({pct:.1f}% retained, {b - af} removed)")
    print(f"    saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=["dataset", "journal", "all"], default="all",
                    help="which peak set(s) to collapse")
    a = ap.parse_args()

    if a.target in ("dataset", "all"):
        print(f"[dataset] collapsing 1D NMR in {DATA_DATASET}")
        collapse_dataset(DATA_DATASET)

    if a.target in ("journal", "all"):
        for p in (BENCH_JOURNAL, BENCH_JOURNAL_PREPARED):
            if Path(p).exists():
                print(f"[journal] collapsing {p}")
                collapse_journal_pkl(p)
            else:
                print(f"[journal] skip (missing): {p}")


if __name__ == "__main__":
    main()
