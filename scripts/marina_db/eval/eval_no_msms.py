#!/usr/bin/env python3
"""
Evaluate MARINA checkpoints on the simulated test split with MS/MS withheld.

The stored test_result.pkl for a run covers `all_inputs` plus the single-modality
conditions, but never "everything except MS/MS". This script adds that condition:
inputs = hsqc + c_nmr + h_nmr + mw, i.e. the full input set minus `mass_spec`.

`all_inputs` is evaluated alongside it (it is always the first test dataloader)
and is used as a harness check -- it should reproduce the number already stored
in the run's test_result.pkl.

Note: this cannot be done through src.main. `additional_test_types` is not in
DO_NOT_OVERRIDE, so a CLI value is silently replaced by the checkpoint's
params.json, and dropping `mass_spec` from `input_types` instead would remove
the MS encoder from the model and break the strict load_state_dict.

Usage:
    pixi run python scripts/benchmark/eval_no_msms.py \
        --results_root /root/gurusmart/Moonshot/results \
        --seeds 0 1 2 \
        --out /root/gurusmart/marina_no_msms_eval.json
"""
import argparse
import glob
import json
import os
import pickle
from dataclasses import fields as dc_fields

import pytorch_lightning as pl
import torch

from src.modules import MARINA, MARINAArgs, MARINADataModule
from src.modules.data.fp_loader import make_fp_loader
from src.modules.benchmark import load_model
from src.modules.core.const import DATASET_ROOT

# full input set minus `mass_spec`; `mw` is kept
NO_MSMS = ['hsqc', 'c_nmr', 'h_nmr', 'mw']

# metrics worth reporting out of the full logged set
REPORT = ['rank_1', 'rank_5', 'rank_10', 'mean_rank', 'cos', 'jaccard', 'f1']


def find_run(results_root: str, seed: int) -> tuple[str, str]:
    """Return (params_path, ckpt_path) for the one run dir of this seed holding a ckpt."""
    exp_dir = os.path.join(results_root, f'marina-final-run-seed-{seed}')
    ckpts = sorted(glob.glob(os.path.join(exp_dir, '*', '*.ckpt')))
    if len(ckpts) != 1:
        raise RuntimeError(f'expected exactly 1 ckpt under {exp_dir}, found {ckpts}')
    run_dir = os.path.dirname(ckpts[0])
    return os.path.join(run_dir, 'params.json'), ckpts[0]


def build_args(params: dict, ckpt: str, num_workers: int) -> MARINAArgs:
    valid = {f.name for f in dc_fields(MARINAArgs)}
    kw = {k: v for k, v in params.items() if k in valid}
    # input_types stays at the trained value so the architecture matches the ckpt;
    # the MS/MS ablation happens at the dataloader level via additional_test_types.
    kw.update(
        train=False,
        test=True,
        benchmark=False,
        load_from_checkpoint=ckpt,
        additional_test_types=[NO_MSMS],
        num_workers=num_workers,
    )
    return MARINAArgs(**kw)


def eval_seed(seed: int, results_root: str, num_workers: int) -> dict:
    params_path, ckpt_path = find_run(results_root, seed)
    with open(params_path) as f:
        params = json.load(f)

    args = build_args(params, ckpt_path, num_workers)
    assert 'mass_spec' in args.input_types, 'expected a ckpt trained with mass_spec'

    fp_loader = make_fp_loader(
        args.fp_type,
        entropy_out_dim=args.out_dim,
        retrieval_path=os.path.join(DATASET_ROOT, 'retrieval.pkl'),
    )
    model = MARINA(args, fp_loader)
    data_module = MARINADataModule(args, fp_loader)
    load_model(args, model)  # load_state_dict(strict) + setup_ranker + eval

    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    raw = trainer.test(model, data_module)[0]

    no_ms_key = '_'.join(NO_MSMS)
    out = {
        'seed': seed,
        'ckpt': ckpt_path,
        'input_types': list(args.input_types),
        'no_msms_inputs': NO_MSMS,
        'test_types': [list(t) for t in data_module.test_types],
        'spectral_types': list(model.spectral_types),
        'all_inputs': {m: raw.get(f'test/mean_{m}/all_inputs') for m in REPORT},
        'no_msms': {m: raw.get(f'test/mean_{m}/{no_ms_key}') for m in REPORT},
    }

    # harness check: the recomputed all_inputs numbers should match the stored run
    stored_path = os.path.join(os.path.dirname(ckpt_path), 'test_result.pkl')
    if os.path.exists(stored_path):
        with open(stored_path, 'rb') as f:
            stored = pickle.load(f)
        stored = stored[0] if isinstance(stored, list) else stored
        out['stored_all_inputs'] = {
            m: stored.get(f'test/mean_{m}/all_inputs') for m in REPORT
        }

    del model, data_module, fp_loader
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', default='/root/gurusmart/Moonshot/results')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--out', default='/root/gurusmart/marina_no_msms_eval.json')
    a = ap.parse_args()

    results = []
    for seed in a.seeds:
        print(f'\n===== seed {seed} =====', flush=True)
        r = eval_seed(seed, a.results_root, a.num_workers)
        results.append(r)
        print(json.dumps(r, indent=2), flush=True)
        with open(a.out, 'w') as f:
            json.dump(results, f, indent=2)

    print(f'\nwrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
