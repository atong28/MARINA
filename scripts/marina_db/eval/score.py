#!/usr/bin/env python3
"""
The one MARINA checkpoint scorer for the marina_db build.

Consolidates the legacy scorers onto lib.eval_loop. The Journal is the sole benchmark
going forward. Scores each checkpoint on:

  - journal        (BENCH_JOURNAL_PREPARED, leakage-filtered) val / test / all,
                   reported for the full set AND the marina_clean / both_clean
                   subsets (199 of 467 journal compounds sit in MARINA1's train
                   split, so the full set is confounded -- keep both in view)
  - test_nmr_mw    the test split restricted to molecules carrying all of
                   hsqc/c_nmr/h_nmr/mw, batched through the dataset

Metrics per split: rank@1/5/10 (tie-aware, D6) + dereplication top-1/5/10
(cosine of sparse FP > 0.99) + mean cosine. --strict also emits the pessimistic
rank@k for reproducing published numbers.

rank@k uses DATA_DATASET/<fp_type>/rankingset.pt. The journal compounds are folded
into the retrieval bank at build stage 3, so the gold row is always present and
rank@k is well-defined for all 467 -- there is no separate augmented bank.

Model list file: one row per checkpoint  ->  name|params.json|checkpoint.pt

    DATASET_ROOT=/workspace PYTHONPATH=. pixi run python3 \
        scripts/marina_db/eval/score.py --models models.txt --out results.json
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # scripts/marina_db  (config, lib)
sys.path.insert(0, str(_HERE.parents[3]))   # repo root          (src)

import argparse
import json
import os
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# fp vocab (fp_type/out_dim) comes per-model from params.json, so a model is always
# scored on its own vocabulary; config FP_TYPE/FP_OUT_DIM are the build-time default.
from config import DATA_DATASET, RETRIEVAL_PKL, BENCH_JOURNAL_PREPARED
from lib.eval_loop import to_device, build_args, eval_split, summarise

from src.modules import MARINA, MARINADataModule
from src.modules.marina.dataset import MARINADataset, collate
from src.modules.data.fp_loader import make_fp_loader
from src.modules.core.ranker import RankingSet

NMR_MW = ["hsqc", "c_nmr", "h_nmr", "mw"]


def bucket_journal(records, strict):
    """val / test / all summaries; 'all' is val+test scored once."""
    return {split: summarise(
        records if split == "all" else [r for r in records if r["split"] == split], strict)
        for split in ("val", "test", "all")}


@torch.no_grad()
def eval_test_subset(args, model, fp_loader, device, strict,
                     batch_size=32, num_workers=4):
    """Test split restricted to molecules carrying all of hsqc/c_nmr/h_nmr/mw.

    Batched through MARINADataset (truth FP comes from the dataset, so no derep);
    kept from eval_comprehensive.py.
    """
    ds = MARINADataset(args, fp_loader, split="test", override_input_types=NMR_MW)
    # MARINADataset has no .collate_fn attribute -- the padding collate is the
    # module-level `collate` in marina/dataset.py, which MARINADataModule wraps as
    # _collate_fn. A hasattr() guard here silently falls back to default_collate,
    # which torch.stacks ragged peak lists and dies on the first mixed batch.
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=collate)
    recs = []
    for batch in tqdm(loader, desc="test/nmr+mw", leave=False):
        inputs, truth = to_device(batch[0], device), to_device(batch[1], device)
        pred = torch.sigmoid(model(batch=inputs))
        pred = pred[0] if isinstance(pred, (tuple, list)) else pred
        truth_n = truth / truth.norm(dim=1, keepdim=True).clamp_min(1e-12)
        pred_n = pred / pred.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cosv = (pred_n * truth_n).sum(dim=1).tolist()
        rank = model.ranker.batched_rank(pred, truth_n, tie_aware=True).tolist()
        rank_s = (model.ranker.batched_rank(pred, truth_n, tie_aware=False).tolist()
                  if strict else [None] * len(cosv))
        for c, r, rs in zip(cosv, rank, rank_s):
            rec = {"split": "test", "cos": c, "rank": r, "derep": None}
            if strict:
                rec["rank_strict"] = rs
            recs.append(rec)
    return summarise(recs, strict)


def eval_model(params_path, ckpt_path, journal, device, strict, skip_test):
    params = json.load(open(params_path))
    args = build_args(params, ckpt_path)
    fp_loader = make_fp_loader(args.fp_type, entropy_out_dim=args.out_dim,
                               retrieval_path=str(RETRIEVAL_PKL))
    model = MARINA(args, fp_loader)
    data_module = MARINADataModule(args, fp_loader)
    # Not benchmark.load_model: it torch.loads without map_location (cannot run on a
    # CPU box) and calls setup_ranker(), which reads a rankingset we override below.
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    model.eval()
    model.ranker = RankingSet(
        store=torch.load(os.path.join(DATA_DATASET, args.fp_type, "rankingset.pt"),
                         map_location="cpu"))
    model = model.to(device)
    restrictions = args.input_types if args.restrictions is None else args.restrictions

    res = {"fp_type": args.fp_type, "ckpt": ckpt_path}

    # Journal: score every entry with a canonical structure once, then subset.
    jentries = [v for v in journal.values() if v.get("canonical_2d_smiles")]
    jrecs = eval_split(jentries, model, data_module, fp_loader, restrictions,
                       device, smiles_key="canonical_2d_smiles", strict=strict)
    res["journal_full"] = bucket_journal(jrecs, strict)
    res["journal_marina_clean"] = bucket_journal(
        [r for r in jrecs if r["entry"].get("marina_clean")], strict)
    res["journal_both_clean"] = bucket_journal(
        [r for r in jrecs if r["entry"].get("both_clean")], strict)

    if not skip_test:
        res["test_nmr_mw"] = eval_test_subset(args, model, fp_loader, device, strict)

    del model, data_module, fp_loader
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="name|params.json|ckpt per line")
    ap.add_argument("--out", required=True, help="results json (resumed if it exists)")
    ap.add_argument("--strict", action="store_true",
                    help="also emit pessimistic (tie_aware=False) rank@k")
    ap.add_argument("--skip_test", action="store_true", help="skip the NMR+MW test subset")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    journal = pickle.load(open(BENCH_JOURNAL_PREPARED, "rb"))

    rows = [ln.strip() for ln in open(a.models) if ln.strip() and not ln.startswith("#")]
    results = json.load(open(a.out)) if os.path.exists(a.out) else {}

    for row in tqdm(rows, desc="scoring"):
        name, params_path, ckpt_path = row.split("|")
        if name in results:
            print(f"skip {name} (done)", flush=True)
            continue
        print(f"=== {name}", flush=True)
        try:
            results[name] = eval_model(params_path, ckpt_path, journal,
                                       device, a.strict, a.skip_test)
        except Exception as e:
            results[name] = {"error": repr(e)}
            print(f"ERROR {name}: {e!r}", flush=True)
        json.dump(results, open(a.out, "w"), indent=2)  # checkpoint after each

    print(f"done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
