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
import argparse
import json
import os
import pickle
from dataclasses import fields as dc_fields

import torch

from src.modules import (
    MARINA, MARINAArgs, MARINADataModule,
    SPECTRE, SPECTREArgs, SPECTREDataModule,
)
from src.modules.data.fp_loader import make_fp_loader
from src.modules.benchmark import load_model, filter_data
from src.modules.core.const import BENCHMARK_ROOT, DATASET_ROOT

MODEL_CLASSES = {"MARINA": MARINA, "SPECTRE": SPECTRE}
ARGS_CLASSES = {"MARINA": MARINAArgs, "SPECTRE": SPECTREArgs}
DM_CLASSES = {"MARINA": MARINADataModule, "SPECTRE": SPECTREDataModule}


def cos(a, b):
    return (torch.dot(a, b) / (torch.norm(a) * torch.norm(b))).item()


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def build_args(params, ckpt):
    ArgsCls = ARGS_CLASSES[params["project_name"]]
    valid = {f.name for f in dc_fields(ArgsCls)}
    kw = {k: v for k, v in params.items() if k in valid}
    kw.update(train=False, test=False, benchmark=True, load_from_checkpoint=ckpt)
    return ArgsCls(**kw)


def peaks(entry):
    inp = entry["input"]
    h = [round(float(r[0]), 3) for r in inp["h_nmr"].tolist()] if "h_nmr" in inp else []
    c = [round(float(r[0]), 3) for r in inp["c_nmr"].tolist()] if "c_nmr" in inp else []
    q = [[round(float(r[1]), 3), round(float(r[0]), 3), int(r[2])] for r in inp["hsqc"].tolist()] if "hsqc" in inp else []
    return h, c, q


@torch.no_grad()
def run(bench_data, data_module, model, fp_loader, restrictions, metadata, device):
    out = {}
    for key, entry in bench_data.items():
        inputs = to_device(data_module.format_inference_data(filter_data(entry["input"], restrictions)), device)
        pred = torch.sigmoid(model(**inputs)[0])
        idxs = model.ranker.retrieve_idx(pred, 10).tolist()
        sfp = fp_loader.build_mfp_for_smiles(entry["smiles"]).to(device)
        sfp = sfp / torch.norm(sfp)
        near, top_smiles, top_cos = [], [], []
        for k in range(10):
            rsfp = model.ranker.data[idxs[k]].to_dense().float().to(device)
            c = cos(sfp, rsfp)
            near.append(c > 0.99)
            top_smiles.append(metadata[str(idxs[k])]["canonical_2d_smiles"])
            top_cos.append(round(c, 4))
        topk = {k: bool(any(near[:k])) for k in range(1, 11)}
        rank = -1 if not any(near) else next(k for k in range(1, 11) if topk[k])
        h, c_, q = peaks(entry)
        out[entry.get("npid", key)] = {
            "npid": entry.get("npid", key),
            "smiles": entry["smiles"],
            "mw": round(float(entry["input"]["mw"]), 5) if "mw" in entry["input"] else None,
            "cosine_sim": round(cos(pred, sfp), 4),
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
