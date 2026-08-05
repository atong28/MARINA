#!/usr/bin/env python3
"""Cache MARINA test-split fingerprint predictions so the retrieval metric can be
swept offline without re-running the model.

This is the one script here that needs MARINA's own code and env; run it from the
MARINA repo root:

    pixi run python3 analysis/fp-redundancy/scripts/01_predict.py --limit 1500

Writes results/preds.npz: sigmoid(logits) per test molecule (float16), the true
fingerprint as a sparse index list, and the dataset idx of each molecule.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import fields as dc_fields

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARINA_ROOT = os.path.dirname(os.path.dirname(HERE))     # HERE = analysis/<dir>, up 2
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
os.environ.setdefault("DATASET_ROOT", os.path.join(DATA_ROOT, "Datasets/MARINA1"))
sys.path.insert(0, MARINA_ROOT)

import numpy as np
import torch

from src.modules import MARINA, MARINAArgs
from src.modules.marina.dataset import MARINADataset, collate
from src.modules.data.fp_loader import make_fp_loader
from src.modules.core.const import DATASET_ROOT

OUT = os.path.join(HERE, "results", "preds.npz")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt_dir", default=os.path.join(DATA_ROOT, "Checkpoints/MARINA/final1"))
ap.add_argument("--limit", type=int, default=1500, help="molecules to evaluate (0 = all)")
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
args_cli = ap.parse_args()

torch.set_num_threads(os.cpu_count() or 4)
torch.manual_seed(args_cli.seed)

with open(os.path.join(args_cli.ckpt_dir, "params.json")) as f:
    params = json.load(f)

valid = {f.name for f in dc_fields(MARINAArgs)}
kw = {k: v for k, v in params.items() if k in valid}
kw.update(train=False, test=True, benchmark=False,
          load_from_checkpoint=os.path.join(args_cli.ckpt_dir, "best.ckpt"),
          additional_test_types=[], num_workers=0)
args = MARINAArgs(**kw)

fp_loader = make_fp_loader(
    args.fp_type,
    entropy_out_dim=args.out_dim,
    retrieval_path=os.path.join(DATASET_ROOT, "retrieval.pkl"),
)

model = MARINA(args, fp_loader)
state = torch.load(args.load_from_checkpoint, map_location="cpu")["state_dict"]
model.load_state_dict(state)
model.eval()
# NOTE: deliberately skipping model.setup_ranker() -- ranking happens offline in
# 03_reweight_sweep.py, and the ranker would load a 400MB rankingset we don't need.
print(f"loaded {args.load_from_checkpoint}", flush=True)

ds = MARINADataset(args, fp_loader, split="test")
n_total = len(ds)
rng = np.random.default_rng(args_cli.seed)
sel = np.arange(n_total) if args_cli.limit in (0, None) or args_cli.limit >= n_total \
    else np.sort(rng.choice(n_total, size=args_cli.limit, replace=False))
print(f"test split: {n_total} molecules with all input types; evaluating {len(sel)}", flush=True)

probs = np.zeros((len(sel), args.out_dim), dtype=np.float16)
fp_idx, fp_ptr = [], [0]
mol_idx = np.array([ds.data[int(i)][0] for i in sel], dtype=np.int64)

t0 = time.time()
with torch.no_grad():
    for start in range(0, len(sel), args_cli.batch_size):
        chunk = sel[start:start + args_cli.batch_size]
        batch_inputs, fps = collate([ds[int(i)] for i in chunk])
        logits = model.forward(batch_inputs)
        probs[start:start + len(chunk)] = torch.sigmoid(logits).numpy().astype(np.float16)
        for row in fps:
            nz = torch.nonzero(row, as_tuple=False).flatten().numpy().astype(np.int32)
            fp_idx.append(nz)
            fp_ptr.append(fp_ptr[-1] + len(nz))
        done = start + len(chunk)
        rate = done / (time.time() - t0)
        print(f"  {done}/{len(sel)}  ({rate:.1f} mol/s, eta {(len(sel)-done)/rate/60:.1f} min)",
              flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(
    OUT,
    probs=probs,
    fp_idx=np.concatenate(fp_idx),
    fp_ptr=np.array(fp_ptr, dtype=np.int64),
    mol_idx=mol_idx,
    out_dim=args.out_dim,
    ckpt=args_cli.ckpt_dir,
)
print(f"wrote {OUT}  ({time.time()-t0:.1f}s total)", flush=True)
