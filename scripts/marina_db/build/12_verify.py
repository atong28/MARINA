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
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[3]))
from config import DATA_DATASET, BENCH_JOURNAL, SPLIT_WEIGHTS
from src.modules.data.smiles import canonicalize_smiles

SPLITS = ("train", "val", "test")
MODALITIES = ("HSQC_NMR", "C_NMR", "H_NMR", "MassSpec", "MassSpecNeg", "FragIdx")
HAS_FLAG = {"HSQC_NMR": "has_hsqc", "C_NMR": "has_c_nmr", "H_NMR": "has_h_nmr",
            "MassSpec": "has_mass_spec", "MassSpecNeg": "has_mass_spec_neg"}
# Canonical last-dim per spectral modality in an entry's input dict.
# collate expects 2D tensors of shape (N, JOURNAL_MOD_D[mod]); a 1D or
# wrong-width tensor crashes forward with StopIteration (which the eval loop
# silently swallows, killing val/mean_cos). Lint catches that upfront.
JOURNAL_MOD_D = {"hsqc": 3, "c_nmr": 1, "h_nmr": 1, "mass_spec": 2, "mass_spec_neg": 2}

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
    global MODALITIES
    if 'has_hmbc' in next(iter(index.values())):        # MARINA2.0 ceiling 2D shards (build_2d.py)
        MODALITIES = MODALITIES + ("HMBC_NMR", "COSY_NMR")
        HAS_FLAG.update({"HMBC_NMR": "has_hmbc", "COSY_NMR": "has_cosy"})
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

    # ---- 2c. journal input tensors are well-formed AND non-empty ----
    # Every modality tensor in a journal entry's input dict must be 2D, have the
    # canonical last-dim, and hold at least one peak. An empty tensor (numel==0)
    # or a 1D one is the fingerprint of a header-only curation CSV that got
    # serialized as `torch.tensor([])` -- "we claim this modality but have no
    # data". At runtime this slips past collate's "all None -> skip" check;
    # a 1D empty case crashes on next(x.shape[1] ...) with StopIteration, which
    # Lightning's eval_loop silently absorbs (evaluation_loop.py:147:
    # `except StopIteration`), losing module.on_validation_epoch_end and any
    # EarlyStopping metric monitored there. The right response is to drop the
    # entry entirely (see Benchmark/dropped/README.txt), not to silently
    # in-fill; this lint enforces that policy at build time.
    def check_journal_input_tensors(path, label):
        if not Path(path).exists():
            print(f"[skip] {label} input-tensor lint: not present ({path})")
            return
        data = pickle.load(open(path, "rb"))
        bad = []  # (npid, split, mod, shape_or_type, reason)
        for k, entry in data.items():
            inp = entry.get("input", {}) or {}
            sp = entry.get("split", "?")
            for mod, v in inp.items():
                if mod == "mw":
                    continue  # scalar; format_inference_data reshapes to (1, 1)
                if mod == "formula":
                    continue  # not stored in input dict for journal
                if v is None:
                    continue
                if not isinstance(v, torch.Tensor):
                    bad.append((k, sp, mod, type(v).__name__, "not-tensor"))
                    continue
                expected_d = JOURNAL_MOD_D.get(mod)
                if v.ndim != 2:
                    bad.append((k, sp, mod, tuple(v.shape), "ndim!=2"))
                elif expected_d is not None and v.shape[1] != expected_d:
                    bad.append((k, sp, mod, tuple(v.shape), f"shape[1]!={expected_d}"))
                elif v.numel() == 0:
                    bad.append((k, sp, mod, tuple(v.shape), "empty (numel==0)"))
        check(f"{label} input tensors non-empty and well-formed ({len(data)} entries)",
              not bad,
              f"{len(bad)} bad tensors; first: {bad[:5]}")

    check_journal_input_tensors(BENCH_JOURNAL, "journal")

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
