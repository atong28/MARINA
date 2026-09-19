"""Journal/benchmark RETRIEVAL for the 3 flagship seeds + their 3-seed ENSEMBLE CONSENSUS.

Reproduces MARINA's exact benchmark-retrieval recipe (soft sigmoid FP, L2-cosine against the
531k uniqmult rankingset, strict/tie rank@{1,5,10}; radius 10; MS/MS not re-normalized) — see
src/modules/benchmark.py::_run_benchmark_loop — but runs all three flagship checkpoints on each
entry and additionally scores the CONSENSUS = elementwise mean of the three sigmoid vectors (NO
threshold; retrieval wants the soft vector). Emits per-seed rank@k, the 3-seed mean±std, and the
consensus, per split/subset, plus paper-ready \\pmm cells.

Run (on a box with the flagship ckpts + uniqmult bank + benchmark set staged), e.g. grapefruit:
  DATASET_ROOT=~/MARINA/data/dataset \
  BENCHMARK_ROOT=~/MARINA/data/benchmark-spectreclean \
  pixi run python -m scripts.benchmark.eval_consensus_journal \
      --results_root ~/MARINA/results \
      --experiments marina-db-uniqmult-formula-s0 marina-db-uniqmult-formula-s1 marina-db-uniqmult-formula-s2 \
      --out ~/MARINA/results/consensus_journal_results.json
"""
import os
import re
import json
import glob
import argparse

import numpy as np
import torch

from src.modules.marina import MARINA, MARINAArgs, MARINADataModule
from src.modules.data.fp_loader import make_fp_loader
from src.modules.core.const import DATASET_ROOT, BENCHMARK_ROOT, DO_NOT_OVERRIDE
from src.modules.benchmark import (
    _journal_subsets, _rank_conventions, filter_data, _to_device, cos_sim,
    formula_vec_from_smiles,
)
import pickle


def build_args(params_path):
    with open(params_path) as f:
        params = json.load(f)
    args = MARINAArgs()
    for k, v in params.items():
        if k not in DO_NOT_OVERRIDE and hasattr(args, k):
            setattr(args, k, v)
    args.train = False
    if getattr(args, "scheduler", None) == "none":
        args.scheduler = None
    return args


def find_ckpt(results_root, exp):
    hits = sorted(glob.glob(os.path.join(results_root, exp, "**", "epoch_epoch=*.ckpt"), recursive=True))
    if not hits:
        hits = sorted(glob.glob(os.path.join(results_root, exp, "**", "*.ckpt"), recursive=True))
    if not hits:
        raise FileNotFoundError(f"no ckpt under {os.path.join(results_root, exp)}")
    return hits[-1]


def summarise(recs):
    n = len(recs)
    if n == 0:
        return None
    out = {"n": n, "mean_cos": float(np.mean([r["cos"] for r in recs]))}
    for k in (1, 5, 10):
        out[f"strict_top{k}"] = 100.0 * sum(r["rank_strict"] < k for r in recs) / n
        out[f"tie_top{k}"] = 100.0 * sum(r["rank_tie"] < k for r in recs) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--experiments", nargs=3, required=True, help="the 3 flagship seed exp names")
    ap.add_argument("--deltas", action="store_true", help="also emit +Formula/+MW combo subsets (Table 1/S1)")
    ap.add_argument("--benchmark", default=None,
                    help="benchmark set pkl (default BENCHMARK_ROOT/benchmark-journal.pkl); "
                         "point at the 466 journal (Table 1) or the 205 SPECTRE-clean (Table 3)")
    ap.add_argument("--limit", type=int, default=None, help="cap entries per split (smoke test)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[consensus] device={dev} DATASET_ROOT={DATASET_ROOT} BENCHMARK_ROOT={BENCHMARK_ROOT}")

    ckpts = [find_ckpt(a.results_root, e) for e in a.experiments]
    args = build_args(os.path.join(os.path.dirname(ckpts[0]), "params.json"))
    print(f"[consensus] fp_type={args.fp_type} input_types={args.input_types}")

    fp_loader = make_fp_loader(args.fp_type, entropy_out_dim=getattr(args, "out_dim", 16384),
                               retrieval_path=os.path.join(DATASET_ROOT, "retrieval.pkl"))
    data_module = MARINADataModule(args, fp_loader)

    models = []
    for ck in ckpts:
        m = MARINA(args, fp_loader)
        state = torch.load(ck, map_location=dev)
        m.load_state_dict(state.get("state_dict", state))  # strict=True, matches benchmark recipe
        m.to(dev).eval()
        models.append(m)
        print(f"[consensus] loaded {ck}")
    models[0].setup_ranker()          # 531k CSR bank; seed-independent -> shared for ranking
    ranker = models[0].ranker

    bench_path = a.benchmark or os.path.join(BENCHMARK_ROOT, "benchmark-journal.pkl")
    with open(bench_path, "rb") as f:
        benchmark_data = pickle.load(f)
    print(f"[consensus] benchmark set: {bench_path}")
    n_entries = len(benchmark_data)
    splits = sorted({e.get("split", "test") for e in benchmark_data.values()})
    subsets = _journal_subsets(args.input_types, deltas=a.deltas)
    print(f"[consensus] {n_entries} entries; splits={splits}; {len(subsets)} subsets")

    names = list(a.experiments) + ["consensus"]
    # records[name][f"{split}/{subset}"] = [rec,...]
    records = {nm: {} for nm in names}
    mfp_cache = {}

    for subset, restrictions in subsets.items():
        for split in splits:
            key = f"{split}/{subset}"
            for nm in names:
                records[nm].setdefault(key, [])
            seen = 0
            for entry in benchmark_data.values():
                if entry.get("split", "test") != split:
                    continue
                if a.limit and seen >= a.limit:
                    break
                raw_input = entry["input"]
                if "formula" in restrictions:
                    raw_input = {**raw_input, "formula": formula_vec_from_smiles(entry["smiles"])}
                clean = filter_data(raw_input, restrictions)
                if not any(kk not in ("mw", "formula") for kk in clean):
                    continue
                inputs = _to_device(data_module.format_inference_data(clean), dev)
                preds = []
                with torch.no_grad():
                    for m in models:
                        preds.append(torch.sigmoid(m(**inputs)[0]))
                cons = torch.stack(preds, 0).mean(0)     # consensus soft vector (no threshold)
                smi = entry["smiles"]
                if smi not in mfp_cache:
                    mfp_cache[smi] = fp_loader.build_mfp_for_smiles(smi)
                sfp = mfp_cache[smi]
                sfp = (sfp / torch.norm(sfp)).to(dev)
                for nm, pred in zip(names, preds + [cons]):
                    rs, rt = _rank_conventions(pred, sfp, ranker)
                    records[nm][key].append(
                        {"cos": cos_sim(pred, sfp).item(), "rank_strict": rs, "rank_tie": rt})
                seen += 1

    # summarise + 3-seed mean±std vs consensus
    summ = {nm: {k: summarise(v) for k, v in records[nm].items()} for nm in names}
    seeds = list(a.experiments)
    report = {}
    all_keys = sorted(summ["consensus"].keys())
    print(f"\n{'split/subset':>28} {'metric':>12} {'seed mean±std':>16} {'consensus':>10}")
    for key in all_keys:
        if summ["consensus"][key] is None:
            continue
        report[key] = {"n": summ["consensus"][key]["n"]}
        for metric in ("tie_top1", "tie_top5", "tie_top10", "strict_top1", "mean_cos"):
            vals = [summ[s][key][metric] for s in seeds if summ[s][key]]
            mean, std = float(np.mean(vals)), float(np.std(vals))
            cons = summ["consensus"][key][metric]
            fmt = (lambda x: f"{x:.3f}") if metric == "mean_cos" else (lambda x: f"{x:.2f}")
            report[key][metric] = {"seed_mean": mean, "seed_std": std, "consensus": cons,
                                   "pmm": f"{fmt(mean)}\\pmm{{{fmt(std)}}}", "consensus_cell": fmt(cons)}
            print(f"{key:>28} {metric:>12} {fmt(mean)+'±'+fmt(std):>16} {fmt(cons):>10}")

    with open(a.out, "w") as f:
        json.dump({"experiments": a.experiments, "n_entries": n_entries,
                   "summary": summ, "report": report}, f, indent=2)
    print(f"\n[consensus] wrote {a.out}")


if __name__ == "__main__":
    main()
