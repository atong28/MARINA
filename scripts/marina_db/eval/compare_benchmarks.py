#!/usr/bin/env python3
"""
Per-compound benchmark comparison for a single MARINA/SPECTRE checkpoint.

For each of benchmark.pkl (Annotated) and benchmark-journal.pkl (Journal), and
every compound in it, records the retrieval outcome so an error-analysis
spreadsheet can be built offline:
  - dereplication_topk (k=1..10 hit booleans; hit == a retrieved structure has
    cosine of sparse FP > 0.99 with the query, matching src/modules/benchmark.py)
  - dereplication_rank (smallest k with a hit, else -1)
  - cosine_sim (predicted vs ground-truth sparse FP)
  - top-1 retrieved SMILES and its sFP cosine to the query
  - raw peak lists (h_nmr, c_nmr, hsqc) so the input differences are visible

Writes one JSON: {benchmark: {split: {npid: record}}}.
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

from lib.eval_loop import build_args, eval_split

from src.modules import (
    MARINA, MARINADataModule,
    SPECTRE, SPECTREDataModule,
)
from src.modules.data.fp_loader import make_fp_loader
from src.modules.benchmark import load_model
from src.modules.core.const import BENCHMARK_ROOT, DATASET_ROOT

MODEL_CLASSES = {"MARINA": MARINA, "SPECTRE": SPECTRE}
DM_CLASSES = {"MARINA": MARINADataModule, "SPECTRE": SPECTREDataModule}


def peaks(entry):
    inp = entry["input"]
    h = [round(float(r[0]), 3) for r in inp["h_nmr"].tolist()] if "h_nmr" in inp else []
    c = [round(float(r[0]), 3) for r in inp["c_nmr"].tolist()] if "c_nmr" in inp else []
    q = [[round(float(r[1]), 3), round(float(r[0]), 3), int(r[2])] for r in inp["hsqc"].tolist()] if "hsqc" in inp else []
    return h, c, q


@torch.no_grad()
def run(bench_data, data_module, model, fp_loader, restrictions, metadata, device):
    items = list(bench_data.items())
    recs = eval_split([entry for _, entry in items], model, data_module, fp_loader,
                      restrictions, device, smiles_key="smiles", return_details=True)
    out = {}
    for (key, entry), rec in zip(items, recs):
        near = rec["derep"]
        top_cos = [round(c, 4) for c in rec["derep_cos"]]
        top_smiles = [metadata[str(i)]["canonical_2d_smiles"] for i in rec["idxs"]]
        topk = {k: bool(any(near[:k])) for k in range(1, 11)}
        rank = -1 if not any(near) else next(k for k in range(1, 11) if topk[k])
        h, c_, q = peaks(entry)
        out[entry.get("npid", key)] = {
            "npid": entry.get("npid", key),
            "smiles": entry["smiles"],
            "mw": round(float(entry["input"]["mw"]), 5) if "mw" in entry["input"] else None,
            "cosine_sim": round(rec["cos"], 4),
            "dereplication_rank": rank,
            "topk": topk,
            "top1_smiles": top_smiles[0],
            "top1_cos_sfp": top_cos[0],
            "top_smiles": top_smiles,
            "top_cos_sfp": top_cos,
            "n_hsqc": len(q), "n_c_nmr": len(c_), "n_h_nmr": len(h),
            "h_nmr": h, "c_nmr": c_, "hsqc": q,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = json.load(open(a.params))
    args = build_args(params, a.ckpt)
    fp_loader = make_fp_loader(args.fp_type, entropy_out_dim=args.out_dim,
                               retrieval_path=os.path.join(DATASET_ROOT, "retrieval.pkl"))
    model = MODEL_CLASSES[args.project_name](args, fp_loader)
    data_module = DM_CLASSES[args.project_name](args, fp_loader)
    load_model(args, model)
    model = model.to(device)
    restrictions = args.input_types if args.restrictions is None else args.restrictions
    metadata = json.load(open(os.path.join(DATASET_ROOT, "metadata.json")))

    result = {}
    for label, fname in (("annotated", "benchmark.pkl"), ("journal", "benchmark-journal.pkl")):
        data = pickle.load(open(os.path.join(BENCHMARK_ROOT, fname), "rb"))
        by_split = {}
        for split in ("val", "test"):
            sub = {k: v for k, v in data.items() if v.get("split") == split}
            print(f"[{label}/{split}] {len(sub)} compounds", flush=True)
            by_split[split] = run(sub, data_module, model, fp_loader, restrictions, metadata, device)
        result[label] = by_split

    json.dump(result, open(a.out, "w"))
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
