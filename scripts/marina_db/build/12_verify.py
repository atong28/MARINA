#!/usr/bin/env python3
"""Verify the built MARINA-DB arrow dataset. Exits nonzero on any failure.

One checker over config.DATA_DATASET:

  1. index <-> arrow split consistency: each modality shard holds exactly the index
     idx for that split (gated by the has_* flags; FragIdx covers every molecule),
     with no duplicate idx within a shard and disjoint shards across splits. The
     arrow shards are what the dataloader reads, so consistency is checked against
     them rather than against JSONL (this dataset ships none).
  2. retrieval superset: every index molecule AND every Journal-benchmark molecule is
     present in the retrieval bank by canonical SMILES (the journal is folded into
     retrieval at stage 3, so all 467 must be present).
  3. no duplicate canonical SMILES in the index.
  4. split-distribution sanity: labels are exactly train/val/test, none empty.

Prints a PASS/FAIL line per check and a final report. Index and retrieval are read
from the dataset directory (the copies shipped with, and consistent with, the arrow
shards); --dataset_root overrides that root.
"""
import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[3]))
from config import DATA_DATASET, BENCH_JOURNAL, SPLIT_WEIGHTS
from src.modules.data.smiles import canonicalize_smiles

SPLITS = ("train", "val", "test")
MODALITIES = ("HSQC_NMR", "C_NMR", "H_NMR", "MassSpec", "FragIdx")
HAS_FLAG = {"HSQC_NMR": "has_hsqc", "C_NMR": "has_c_nmr", "H_NMR": "has_h_nmr",
            "MassSpec": "has_mass_spec"}

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}",
          flush=True)
    if not ok:
        failures.append(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset_root", default=str(DATA_DATASET))
    a = ap.parse_args()
    root = Path(a.dataset_root)

    index = pickle.load(open(root / "index.pkl", "rb"))
    retrieval = pickle.load(open(root / "retrieval.pkl", "rb"))
    retrieval_smiles = {entry["smiles"] for entry in retrieval.values()}

    # ---- 1. index <-> arrow split consistency ----
    for mod in MODALITIES:
        shard_idx = {}
        for sp in SPLITS:
            t = pq.read_table(root / "arrow" / sp / f"{mod}.parquet", columns=["idx"])
            shard_idx[sp] = set(t.column("idx").to_pylist())
            check(f"{sp}/{mod}: no duplicate idx in shard",
                  len(shard_idx[sp]) == t.num_rows)
            expect = ({i for i in index if index[i]["split"] == sp}
                      if mod == "FragIdx" else
                      {i for i in index if index[i]["split"] == sp and index[i][HAS_FLAG[mod]]})
            check(f"{sp}/{mod}: shard idx set == index has_* flags",
                  shard_idx[sp] == expect,
                  f"shard {len(shard_idx[sp])} vs index {len(expect)}")
        union = set().union(*shard_idx.values())
        check(f"{mod}: shards are disjoint across splits",
              sum(len(v) for v in shard_idx.values()) == len(union))

    # ---- 2. retrieval superset (index molecules; canonical field vs bank) ----
    miss_index = [(i, index[i]["smiles"]) for i in index
                  if index[i]["smiles"] not in retrieval_smiles]
    check("retrieval superset: every index molecule present", not miss_index,
          f"{len(retrieval_smiles)} bank, {len(index)} index, {len(miss_index)} missing")

    # ---- 2b. retrieval superset (benchmark molecules; canonicalized) ----
    def bench_missing(path, label):
        if not Path(path).exists():
            print(f"[skip] {label}: not present ({path})")
            return
        data = pickle.load(open(path, "rb"))
        miss = []
        for k, entry in data.items():
            canon = canonicalize_smiles(entry["smiles"])
            if canon is None or canon not in retrieval_smiles:
                miss.append(k)
        check(f"retrieval superset: every {label} molecule present ({len(data)})",
              not miss, f"{len(miss)} missing: {miss[:10]}")

    bench_missing(BENCH_JOURNAL, "journal")

    # ---- 3. no duplicate canonical SMILES in the index ----
    counts = Counter(v["smiles"] for v in index.values())
    dups = {s: n for s, n in counts.items() if n > 1}
    check("no duplicate canonical SMILES in index", not dups,
          f"{len(dups)} structures appear >1x")

    # ---- 4. split-distribution sanity ----
    split_counts = Counter(v["split"] for v in index.values())
    check("split labels are exactly train/val/test", set(split_counts) == set(SPLITS))
    check("no split is empty", all(split_counts.get(sp, 0) > 0 for sp in SPLITS))
    total = sum(split_counts.values())
    print("  split distribution (target weights "
          f"{SPLIT_WEIGHTS}):")
    for sp, w in zip(SPLITS, SPLIT_WEIGHTS):
        n = split_counts.get(sp, 0)
        print(f"    {sp}: {n:,} ({n / total:.3%}, target {w:.1%})")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILURES: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
