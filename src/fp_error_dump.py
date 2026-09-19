"""
FP error study (Moonshot robustness): dump MARINA predicted-vs-true fingerprints.

One-off analysis. Rebuilds the flagship MARINA model from a checkpoint's params.json,
runs forward over a subsample of the *all_inputs* test split, and dumps, per molecule:
  - true fingerprint active column indices (sparse int32)
  - predicted per-bit probabilities (sigmoid(logits), dense float16, width = out_dim)

Downstream (analyze_fp_error.py) maps columns -> (fragment, multiplicity bucket) via
bitinfo_to_idx.pkl to characterize MARINA's error distribution in fragment space, which
calibrates the Moonshot corruption augmenter.

Run (from MARINA repo root):
  DATASET_ROOT=/home/atong/Workspace/MARINA/data/dataset \
  pixi run python -m src.fp_error_dump \
      --ckpt results/marina-db-uniqmult-formula-s0/2026-09-07_03-29-23/epoch_epoch=620.ckpt \
      --n 6000 --out /home/atong/Workspace/MARINA/scripts/fp_error_study/dump_s0
"""
import os
import json
import argparse
import pickle

import numpy as np
import torch

from .modules.marina import MARINA, MARINAArgs, MARINADataModule
from .modules.data.fp_loader import make_fp_loader
from .modules.core.const import DATASET_ROOT, DO_NOT_OVERRIDE


def build_args(ckpt_path: str) -> MARINAArgs:
    params_path = os.path.join(os.path.dirname(ckpt_path), "params.json")
    with open(params_path) as f:
        params = json.load(f)
    args = MARINAArgs()
    for k, v in params.items():
        if k in DO_NOT_OVERRIDE:
            continue
        if hasattr(args, k):
            setattr(args, k, v)
    args.debug = False
    args.train = False
    if getattr(args, "scheduler", None) == "none":
        args.scheduler = None
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=6000,
                    help="max molecules to dump from the all_inputs test loader")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", required=True, help="output directory")
    args_cli = ap.parse_args()

    print(f"[dump] DATASET_ROOT = {DATASET_ROOT}")
    torch.set_num_threads(os.cpu_count() or 8)
    os.makedirs(args_cli.out, exist_ok=True)

    def flush(pred_rows, true_cols, out_dim):
        pred = np.stack(pred_rows).astype(np.float16)
        np.save(os.path.join(args_cli.out, "pred_probs.npy"), pred)
        with open(os.path.join(args_cli.out, "true_cols.pkl"), "wb") as f:
            pickle.dump(true_cols, f)
        with open(os.path.join(args_cli.out, "meta.json"), "w") as f:
            json.dump({"ckpt": args_cli.ckpt, "fp_type": args.fp_type,
                       "out_dim": out_dim, "n": len(true_cols)}, f, indent=2)

    args = build_args(args_cli.ckpt)
    args.batch_size = args_cli.batch_size
    args.num_workers = 4
    args.persistent_workers = False
    print(f"[dump] fp_type={args.fp_type} input_types={args.input_types}")

    fp_loader = make_fp_loader(
        args.fp_type,
        entropy_out_dim=getattr(args, "out_dim", 16384),
        retrieval_path=os.path.join(DATASET_ROOT, "retrieval.pkl"),
    )
    out_dim = fp_loader.out_dim
    print(f"[dump] out_dim = {out_dim}")

    model = MARINA(args, fp_loader)
    state = torch.load(args_cli.ckpt, map_location="cpu")
    state_dict = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[dump] loaded ckpt (missing={len(missing)} unexpected={len(unexpected)})")
    model.eval()

    dm = MARINADataModule(args, fp_loader)
    dm.setup("test")
    loaders = dm.test_dataloader()
    loader = loaders[0]  # index 0 == 'all_inputs'
    print(f"[dump] test loaders: {len(loaders)}; using all_inputs")

    pred_rows = []
    true_cols = []
    n_done = 0
    with torch.no_grad():
        for batch in loader:
            batch_inputs, fps = batch
            logits = model(batch_inputs)
            probs = torch.sigmoid(logits).to(torch.float16).cpu().numpy()
            fps_np = fps.cpu().numpy()
            for i in range(probs.shape[0]):
                pred_rows.append(probs[i])
                true_cols.append(np.nonzero(fps_np[i])[0].astype(np.int32))
                n_done += 1
            if n_done >= args_cli.n:
                break
            if n_done % 512 == 0:
                print(f"[dump] {n_done} molecules...", flush=True)
                flush(pred_rows, true_cols, out_dim)

    pred_rows = pred_rows[: args_cli.n]
    true_cols = true_cols[: args_cli.n]
    flush(pred_rows, true_cols, out_dim)
    print(f"[dump] wrote {len(true_cols)} molecules to {args_cli.out}")


if __name__ == "__main__":
    main()
