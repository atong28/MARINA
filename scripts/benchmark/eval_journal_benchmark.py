#!/usr/bin/env python3
"""Run the journal benchmark (exp-rank@k strict + exp-mean-cos) for MARINA or SPECTRE.

For each run it rebuilds the args from the run's params.json (so the architecture matches
the checkpoint) and calls benchmark_marina, which loops val+test x {all,nmr,msms} and writes
  BENCHMARK_ROOT/benchmarks/<experiment_name>_benchmark_journal_results.pkl
with strict + tie rank@{1,5,10} and mean_cos per split/subset.

Two input modes:
  A) --results_root R --experiments E1 E2 ...   (MARINA: each E is a dir R/E/<ts>/*.ckpt)
  B) --ckpt CKPT --params PARAMS --name NAME     (single run, e.g. the SPECTRE bundle)

Project is chosen with --project_name {MARINA,SPECTRE} (default MARINA); --fp_type overrides
the fingerprint (SPECTRE bundle = RankingEntropy). DATASET_ROOT (rankingset+retrieval) and
BENCHMARK_ROOT (journal pkl) come from the environment / const.py.

Arg construction follows the (now-removed) eval_no_msms pattern: filter params.json to
valid dataclass fields, then set benchmark flags.
"""
import argparse
import glob
import json
import os
from dataclasses import fields as dc_fields

import torch

from src.modules import (
    MARINA, MARINAArgs, MARINADataModule,
    SPECTRE, SPECTREArgs, SPECTREDataModule,
)
from src.modules.data.fp_loader import make_fp_loader
from src.modules.benchmark import benchmark_marina, load_model
from src.modules.core.const import DATASET_ROOT, BENCHMARK_ROOT

CLASSES = {
    "MARINA": (MARINAArgs, MARINA, MARINADataModule),
    "SPECTRE": (SPECTREArgs, SPECTRE, SPECTREDataModule),
}

# Per-combo conditions for the SIMULATED test (MARINA-DB test split, via trainer.test).
# 'all_inputs' comes for free (test_types[0] = input_types); the datamodule drops any combo
# not fully contained in the model's trained input_types.
SIM_COMBOS = [
    ['hsqc'], ['c_nmr'], ['h_nmr'], ['mass_spec'], ['mass_spec_neg'],
    ['hsqc', 'c_nmr', 'h_nmr'],
    ['hsqc', 'c_nmr'], ['hsqc', 'h_nmr'], ['c_nmr', 'h_nmr'],
    ['mass_spec', 'mass_spec_neg'],
]


def find_run(results_root: str, experiment: str) -> tuple[str, str]:
    exp_dir = os.path.join(results_root, experiment)
    ckpts = sorted(glob.glob(os.path.join(exp_dir, "*", "*.ckpt")))
    if len(ckpts) != 1:
        raise RuntimeError(f"expected exactly 1 ckpt under {exp_dir}, found {ckpts}")
    return os.path.join(os.path.dirname(ckpts[0]), "params.json"), ckpts[0]


def build_args(argcls, params, ckpt, name, fp_type, num_workers, legacy_spectre=False, sim=False):
    valid = {f.name for f in dc_fields(argcls)}
    kw = {k: v for k, v in params.items() if k in valid}
    if sim:
        # simulated test: trainer.test() over the MARINA-DB test split, per SIM_COMBOS
        kw.update(train=False, test=True, benchmark=False,
                  load_from_checkpoint=ckpt, experiment_name=name, num_workers=num_workers)
        if 'additional_test_types' in valid:
            kw['additional_test_types'] = SIM_COMBOS
    else:
        kw.update(train=False, test=False, benchmark=True,
                  load_from_checkpoint=ckpt, experiment_name=name, num_workers=num_workers)
    if fp_type:
        kw["fp_type"] = fp_type
    if legacy_spectre and "legacy_type_embedding" in valid:
        kw["legacy_type_embedding"] = True
    return argcls(**kw)


def sim_test(project, ckpt, params_path, name, fp_type, num_workers, legacy_spectre=False):
    """Simulated test: trainer.test() over the MARINA-DB test split → per-combo
    test/mean_rank_{1,5,10} + test/mean_cos, dumped to <name>_sim_results.json.
    Requires the test-split arrow shards + index.pkl staged under DATASET_ROOT."""
    import pytorch_lightning as pl
    argcls, modelcls, dmcls = CLASSES[project]
    with open(params_path) as f:
        params = json.load(f)
    args = build_args(argcls, params, ckpt, name, fp_type, num_workers, legacy_spectre, sim=True)
    print(f"[{name}] SIM test fp_type={args.fp_type} ckpt={ckpt}", flush=True)
    fp_loader = make_fp_loader(
        args.fp_type, entropy_out_dim=args.out_dim,
        retrieval_path=os.path.join(DATASET_ROOT, "retrieval.pkl"),
    )
    model = modelcls(args, fp_loader)
    data_module = dmcls(args, fp_loader)
    load_model(args, model)  # load_state_dict(strict) + setup_ranker + eval
    trainer = pl.Trainer(accelerator='auto', devices=1, logger=False, enable_checkpointing=False)
    raw = trainer.test(model, data_module)[0]
    metrics = {k: v for k, v in raw.items() if k.startswith("test/mean_")}
    out = os.path.join(BENCHMARK_ROOT, "benchmarks", f"{name}_sim_results.json")
    with open(out, "w") as f:
        json.dump({"ckpt": ckpt, "spectral_types": list(getattr(model, 'spectral_types', [])),
                   "metrics": metrics}, f, indent=2)
    del model, data_module, fp_loader
    torch.cuda.empty_cache()
    return out


def eval_one(project, ckpt, params_path, name, fp_type, num_workers, legacy_spectre=False, deltas=False):
    argcls, modelcls, dmcls = CLASSES[project]
    with open(params_path) as f:
        params = json.load(f)
    args = build_args(argcls, params, ckpt, name, fp_type, num_workers, legacy_spectre)
    print(f"[{name}] project={project} fp_type={args.fp_type} ckpt={ckpt}", flush=True)

    fp_loader = make_fp_loader(
        args.fp_type, entropy_out_dim=args.out_dim,
        retrieval_path=os.path.join(DATASET_ROOT, "retrieval.pkl"),
    )
    model = modelcls(args, fp_loader)
    data_module = dmcls(args, fp_loader)
    benchmark_marina(args, data_module, model, fp_loader, load_from_checkpoint=ckpt, deltas=deltas)
    out = os.path.join(BENCHMARK_ROOT, "benchmarks",
                       f"{name}_benchmark_journal_results.pkl")
    del model, data_module, fp_loader
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_name", default="MARINA", choices=list(CLASSES))
    ap.add_argument("--fp_type", default=None)
    ap.add_argument("--legacy_spectre", action="store_true",
                    help="build the released SPECTRE 4-row NMR_type_embedding")
    ap.add_argument("--sim", action="store_true",
                    help="also run the simulated MARINA-DB test (trainer.test, per-combo)")
    ap.add_argument("--no-journal", dest="no_journal", action="store_true",
                    help="skip the journal benchmark (e.g. sim-only)")
    ap.add_argument("--deltas", action="store_true",
                    help="also compute per-combo +Formula/+MW subsets (flagship Tables 1 & S1)")
    ap.add_argument("--num_workers", type=int, default=2)
    # mode A
    ap.add_argument("--results_root")
    ap.add_argument("--experiments", nargs="+")
    # mode B
    ap.add_argument("--ckpt")
    ap.add_argument("--params")
    ap.add_argument("--name")
    a = ap.parse_args()

    def run(name, ckpt, params_path):
        if not a.no_journal:
            out = eval_one(a.project_name, ckpt, params_path, name, a.fp_type,
                           a.num_workers, a.legacy_spectre, a.deltas)
            print(f"[{name}] wrote {out}", flush=True)
        if a.sim:
            out = sim_test(a.project_name, ckpt, params_path, name, a.fp_type,
                           a.num_workers, a.legacy_spectre)
            print(f"[{name}] wrote {out}", flush=True)

    if a.ckpt:
        run(a.name, a.ckpt, a.params)
        return
    for exp in a.experiments:
        print(f"\n===== {exp} =====", flush=True)
        params_path, ckpt = find_run(a.results_root, exp)
        run(exp, ckpt, params_path)


if __name__ == "__main__":
    main()
