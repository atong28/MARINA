#!/usr/bin/env python3
"""5_spectre_splits.py -- derive the SPECTRE train/val/test partition.

The SPECTRE partition is the `split` field of the SPECTRE corpus index
(`data/raw/index.pkl`, unpacked from spectre_data.zip in stage 1): every legacy
molecule already carries its published SPECTRE split there, and the per-split
`{train,val,test}.jsonl` are keyed by that same idx. This stage groups the corpus
SMILES by that field and writes:

  config.SPECTRE_SPLITS_PKL   {'train','val','test' -> set(canonical 2D SMILES)}

consumed by 7_splits.py (SPECTRE membership -> split) and 10_build_journal.py
(per-family leakage flags). SMILES are fixed-point canonicalized (stereo stripped)
via the one shared canonicalizer so both sides compare on the same keys.

This replaces the never-committed analysis/spectre-split deriver referenced by the
old 30_splits.py; it reads only data/raw, so it is part of the reproducible build.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # scripts/marina_db -> config
sys.path.insert(0, str(_HERE.parents[3]))   # repo root -> src
from config import DATA_RAW, SPECTRE_SPLITS_PKL
from src.modules.data.smiles import canonicalize_smiles

import pickle
from collections import Counter
from multiprocessing import Pool, cpu_count

SPLITS = ("train", "val", "test")


def main():
    index = pickle.load(open(DATA_RAW / "index.pkl", "rb"))
    print(f"Loaded {len(index)} SPECTRE corpus entries from {DATA_RAW / 'index.pkl'}")

    by_split = {s: [] for s in SPLITS}
    unknown = Counter()
    for entry in index.values():
        sp = entry.get("split")
        if sp in by_split:
            by_split[sp].append(entry["smiles"])
        else:
            unknown[sp] += 1
    if unknown:
        print(f"WARNING: {sum(unknown.values())} entries with non-train/val/test split: {dict(unknown)}")

    with Pool(cpu_count()) as pool:
        splits = {s: {c for c in pool.map(canonicalize_smiles, by_split[s], chunksize=500) if c}
                  for s in SPLITS}

    SPECTRE_SPLITS_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(SPECTRE_SPLITS_PKL, "wb") as f:
        pickle.dump(splits, f)
    print(f"Wrote {SPECTRE_SPLITS_PKL}")
    for s in SPLITS:
        print(f"  {s:5s} {len(splits[s]):>8d} canonical SMILES")


if __name__ == "__main__":
    main()
