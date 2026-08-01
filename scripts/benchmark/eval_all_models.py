#!/usr/bin/env python3
"""
Evaluate a set of MARINA / SPECTRE checkpoints on both benchmark.pkl (Annotated,
NP-MRD peak-picked) and benchmark-journal.pkl (Journal, paper-reported shifts),
on each of the val and test splits.

For every (model, benchmark, split) it reports:
  - top-1 / top-5 / top-10 dereplication rate (%)   (a near-identical structure,
    cosine(sFP) > 0.99, appears within the top-k retrievals)
  - mean cosine similarity between predicted and true sparse fingerprint

Metric definitions mirror src/modules/benchmark.py exactly, but the whole loop
runs on a single GPU (device-consistent) for speed.

Model list is read from a text file, one row per model:
    <name>|<params.json path>|<checkpoint path>

Sharding: pass --shard i --nshards N to process models[i::N]. Each shard writes
<out_dir>/eval_shard<i>.json. Run one process per GPU with CUDA_VISIBLE_DEVICES.
"""
import argparse
import json
import os
import pickle
from dataclasses import fields as dc_fields

import torch
from tqdm import tqdm

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
    """Recursively move tensors inside dicts/lists/tuples to device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def build_args(params: dict, ckpt: str):
    proj = params["project_name"]
    ArgsCls = ARGS_CLASSES[proj]
    valid = {f.name for f in dc_fields(ArgsCls)}
    kw = {k: v for k, v in params.items() if k in valid}
    kw.update(train=False, test=False, benchmark=True, load_from_checkpoint=ckpt)
    return ArgsCls(**kw)


@torch.no_grad()
def eval_split(bench_data, split, data_module, model, fp_loader, restrictions, device):
    """Return dict with top1/top5/top10 (%) and mean_cos for one split of one pkl."""
    entries = [v for v in bench_data.values() if v.get("split") == split]
    cosines = []
    derep = {k: [] for k in (1, 5, 10)}
    for entry in entries:
        inputs = data_module.format_inference_data(filter_data(entry["input"], restrictions))
        inputs = to_device(inputs, device)  # works for MARINA {'batch':..} and SPECTRE {'inputs':..,'type_indicator':..}
        output = model(**inputs)
        pred = torch.sigmoid(output[0])
        idxs = model.ranker.retrieve_idx(pred, 10).tolist()
        sfp = fp_loader.build_mfp_for_smiles(entry["smiles"]).to(device)
        sfp = sfp / torch.norm(sfp)
        cosines.append(cos(pred, sfp))
        near = []
        for k in range(10):
            rsfp = model.ranker.data[idxs[k]].to_dense().float().to(device)
            near.append(cos(sfp, rsfp) > 0.99)
        for k in (1, 5, 10):
            derep[k].append(any(near[:k]))
    n = len(entries)
    return {
        "n": n,
        "mean_cos": sum(cosines) / n if n else float("nan"),
        "top1": 100.0 * sum(derep[1]) / n if n else float("nan"),
        "top5": 100.0 * sum(derep[5]) / n if n else float("nan"),
        "top10": 100.0 * sum(derep[10]) / n if n else float("nan"),
    }


def eval_model(name, params_path, ckpt_path, benchmarks, device):
    params = json.load(open(params_path))
    args = build_args(params, ckpt_path)
    fp_loader = make_fp_loader(
        args.fp_type,
        entropy_out_dim=args.out_dim,
        retrieval_path=os.path.join(DATASET_ROOT, "retrieval.pkl"),
    )
    model = MODEL_CLASSES[args.project_name](args, fp_loader)
    data_module = DM_CLASSES[args.project_name](args, fp_loader)
    load_model(args, model)          # load_state_dict + setup_ranker + eval (on CPU)
    model = model.to(device)          # moves params AND ranker.data buffer to GPU
    restrictions = args.input_types if args.restrictions is None else args.restrictions

    result = {"project": args.project_name, "ckpt": ckpt_path}
    for label, data in benchmarks.items():
        for split in ("val", "test"):
            result[f"{label}/{split}"] = eval_split(
                data, split, data_module, model, fp_loader, restrictions, device)
    del model, data_module, fp_loader
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="text file: name|params|ckpt per line")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)

    rows = [ln.strip() for ln in open(a.models) if ln.strip() and not ln.startswith("#")]
    rows = rows[a.shard::a.nshards]

    benchmarks = {
        "annotated": pickle.load(open(os.path.join(BENCHMARK_ROOT, "benchmark.pkl"), "rb")),
        "journal": pickle.load(open(os.path.join(BENCHMARK_ROOT, "benchmark-journal.pkl"), "rb")),
        "simulated": pickle.load(open(os.path.join(BENCHMARK_ROOT, "benchmark-sim.pkl"), "rb")),
    }

    out_path = os.path.join(a.out_dir, f"eval_shard{a.shard}.json")
    results = {}
    if os.path.exists(out_path):
        results = json.load(open(out_path))  # resume

    for row in tqdm(rows, desc=f"shard{a.shard}"):
        name, params_path, ckpt_path = row.split("|")
        if name in results:
            continue
        print(f"[shard{a.shard}] evaluating {name}", flush=True)
        try:
            results[name] = eval_model(name, params_path, ckpt_path, benchmarks, device)
        except Exception as e:
            results[name] = {"error": repr(e)}
            print(f"[shard{a.shard}] ERROR {name}: {e!r}", flush=True)
        json.dump(results, open(out_path, "w"), indent=2)  # checkpoint after each

    print(f"[shard{a.shard}] done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
