#!/usr/bin/env python3
"""
Grapefruit evaluation pipeline: iterate a directory of (Delta) runs and, for each
checkpoint, report

  (1) SIMULATED TEST SCORES on the test split, on a FIXED all-7-modality population
      (the same molecules for every combo -- inputs are masked down per combo, so the
      rows are directly comparable and isolate the marginal value of each modality,
      including +Formula / +MW). For each input combo: mean_cos and rank@{1,5,10} in
      BOTH the strict and tie-aware conventions.

  (2) BENCHMARK (journal) derep scores under the real available inputs (NMR+MW), plus
      an NMR+MW+Formula run with the formula computed on the fly from SMILES (the
      benchmark-journal.pkl on disk is never modified). Same metrics: mean_cos and
      strict/tie rank@{1,5,10}, per split.

Ranking uses the model's own RankingSet against rankingset.pt; strict vs tie-aware are
the two conventions defined in src/modules/core/ranker.py (dot_prod_rank) and
src/modules/benchmark.py (_rank_conventions) -- rank@k = fraction with 0-based rank < k.

Run from the MARINA repo root, e.g.:
    python scripts/benchmark/eval_grapefruit.py --runs_dir /path/to/delta_runs --out_dir eval_out
    python scripts/benchmark/eval_grapefruit.py --runs_dir /path/to/delta_runs --dry_run   # list discovery only

The input-combo matrix below is intentionally a plain data structure -- edit COMBO
definitions to add/remove combinations.
"""
import argparse
import csv
import glob
import json
import os
import pickle
from collections import OrderedDict
from dataclasses import fields as dc_fields

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.modules import MARINA, MARINAArgs, MARINADataModule
from src.modules.marina.dataset import MARINADataset, collate
from src.modules.data.fp_loader import make_fp_loader
from src.modules.benchmark import (
    load_model, filter_data, formula_vec_from_smiles, _rank_conventions, cos_sim,
)
from src.modules.core.const import BENCHMARK_ROOT, DATASET_ROOT

# --------------------------------------------------------------------------- #
# Combo matrix (edit here). Names map to the exact modality keys fed to the model.
# --------------------------------------------------------------------------- #
NMR = ['hsqc', 'c_nmr', 'h_nmr']
MSMS = ['mass_spec', 'mass_spec_neg']
ALL_INPUTS = ['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mass_spec_neg', 'mw', 'formula']

# Spectral-only base combos (no mw / formula). all_inputs is handled separately: it is
# the only combo that carries mw + formula and it gets no +Formula / +MW variants.
BASE_SPECTRAL_COMBOS = OrderedDict([
    ('nmr', NMR),
    ('msms', MSMS),
    ('nmr+msms', NMR + MSMS),
    ('hsqc', ['hsqc']),
    ('c_nmr', ['c_nmr']),
    ('h_nmr', ['h_nmr']),
    ('mass_spec', ['mass_spec']),
    ('mass_spec_neg', ['mass_spec_neg']),
])


def build_test_combos() -> "OrderedDict[str, list]":
    """all_inputs + every spectral base as {base, base+Formula, base+MW} -> 25 combos."""
    combos = OrderedDict()
    combos['all_inputs'] = list(ALL_INPUTS)
    for name, mods in BASE_SPECTRAL_COMBOS.items():
        combos[name] = list(mods)
        combos[f'{name}+formula'] = list(mods) + ['formula']
        combos[f'{name}+mw'] = list(mods) + ['mw']
    return combos


# Benchmark (journal) input combos: the real available inputs are NMR+MW; +formula is
# computed on the fly inside _run_benchmark_loop (journal pkl untouched).
BENCHMARK_COMBOS = OrderedDict([
    ('nmr+mw', NMR + ['mw']),
    ('nmr+mw+formula', NMR + ['mw', 'formula']),
])

METRIC_FIELDS = ['n', 'mean_cos',
                 'strict_top1', 'strict_top5', 'strict_top10',
                 'tie_top1', 'tie_top5', 'tie_top10']


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _pick_checkpoint(candidates, ckpt_name):
    """Choose one checkpoint from a run dir by priority: explicit name > 'best' >
    'last.ckpt' > newest by mtime."""
    if not candidates:
        return None
    if ckpt_name:
        named = [c for c in candidates if os.path.basename(c) == ckpt_name]
        if named:
            return named[0]
    best = [c for c in candidates if 'best' in os.path.basename(c).lower()]
    if best:
        return max(best, key=os.path.getmtime)
    last = [c for c in candidates if os.path.basename(c) == 'last.ckpt']
    if last:
        return last[0]
    return max(candidates, key=os.path.getmtime)


def discover_runs(runs_dir, ckpt_name):
    """Each immediate subdirectory of runs_dir is a run. Locate its params.json and a
    checkpoint (params.json must sit in the same dir as the ckpt -- MARINA auto-loads it
    from there). Returns list of (run_name, params_path, ckpt_path)."""
    runs = []
    for sub in sorted(glob.glob(os.path.join(runs_dir, '*'))):
        if not os.path.isdir(sub):
            continue
        params = sorted(glob.glob(os.path.join(sub, '**', 'params.json'), recursive=True))
        ckpts = sorted(glob.glob(os.path.join(sub, '**', '*.ckpt'), recursive=True))
        if not params or not ckpts:
            print(f"[discover] skip {os.path.basename(sub)}: "
                  f"params.json={'yes' if params else 'NO'} ckpt={'yes' if ckpts else 'NO'}")
            continue
        params_path = params[0]
        # Prefer checkpoints beside the chosen params.json, else any under the run.
        same_dir = [c for c in ckpts if os.path.dirname(c) == os.path.dirname(params_path)]
        ckpt = _pick_checkpoint(same_dir or ckpts, ckpt_name)
        runs.append((os.path.basename(sub), params_path, ckpt))
    return runs


# --------------------------------------------------------------------------- #
# Metric aggregation
# --------------------------------------------------------------------------- #
def _agg(rank_strict, rank_tie, cos):
    """rank_* : 1D long tensors of 0-based ranks; cos : 1D float tensor."""
    n = cos.numel()
    out = {'n': int(n), 'mean_cos': float(cos.mean()) if n else float('nan')}
    for k in (1, 5, 10):
        out[f'strict_top{k}'] = 100.0 * float((rank_strict < k).float().mean()) if n else float('nan')
        out[f'tie_top{k}'] = 100.0 * float((rank_tie < k).float().mean()) if n else float('nan')
    return out


def _agg_recs(recs):
    """recs : list of {cos, rank_strict, rank_tie} from _run_benchmark_loop."""
    n = len(recs)
    if n == 0:
        return {f: (0 if f == 'n' else float('nan')) for f in METRIC_FIELDS}
    cos = torch.tensor([r['cos'] for r in recs])
    rs = torch.tensor([r['rank_strict'] for r in recs])
    rt = torch.tensor([r['rank_tie'] for r in recs])
    return _agg(rs, rt, cos)


# --------------------------------------------------------------------------- #
# Part 1: simulated test scores on the fixed all-7 population
# --------------------------------------------------------------------------- #
def _to_device(batch_inputs, device):
    return {k: v.to(device) for k, v in batch_inputs.items()}


def load_all7_population(args, fp_loader):
    """Test-split molecules with all 7 modalities present, each item carrying every
    modality (inputs are masked down per combo later). Returns list of (input_dict, fp)."""
    ds = MARINADataset(args, fp_loader, split='test', override_input_types=list(ALL_INPUTS))
    return [ds[i] for i in range(len(ds))]


@torch.no_grad()
def score_test_combo(cache, keys, model, device, batch_size):
    keys = set(keys)
    rs_all, rt_all, cos_all = [], [], []
    for start in range(0, len(cache), batch_size):
        chunk = cache[start:start + batch_size]
        batch = [(filter_data(d, keys), fp) for d, fp in chunk]
        batch_inputs, batch_fps = collate(batch)
        batch_inputs = _to_device(batch_inputs, device)
        batch_fps = batch_fps.to(device)
        queries = torch.sigmoid(model(batch=batch_inputs))          # (B, out_dim)
        rs = model.ranker.batched_rank(queries, batch_fps, tie_aware=False)
        rt = model.ranker.batched_rank(queries, batch_fps, tie_aware=True)
        qn = F.normalize(queries, dim=1, p=2.0)
        tn = F.normalize(batch_fps.float(), dim=1, p=2.0)
        cos = (qn * tn).sum(dim=1)
        rs_all.append(rs.cpu()); rt_all.append(rt.cpu()); cos_all.append(cos.cpu())
    return (torch.cat(rs_all), torch.cat(rt_all), torch.cat(cos_all))


# --------------------------------------------------------------------------- #
# Part 2: journal benchmark. Device-aware port of benchmark._run_benchmark_loop
# (that helper leaves format_inference_data tensors on CPU; here the model lives on
# GPU, so inputs must be moved to match). Formula is computed on the fly when in the
# restriction; benchmark-journal.pkl is never modified.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_benchmark_loop(bench_data, data_module, model, fp_loader, restrictions, device, desc):
    restr = list(restrictions)
    recs = []
    for entry in tqdm(bench_data.values(), desc=desc):
        raw_input = entry['input']
        if 'formula' in restr:
            raw_input = {**raw_input, 'formula': formula_vec_from_smiles(entry['smiles'])}
        clean = filter_data(raw_input, restr)
        if not any(k not in ('mw', 'formula') for k in clean):
            continue  # nothing spectral to feed for this subset
        inputs = data_module.format_inference_data(clean)
        inputs = {'batch': _to_device(inputs['batch'], device)}
        output = model(**inputs)
        pred = torch.sigmoid(output[0])
        sfp = fp_loader.build_mfp_for_smiles(entry['smiles'])
        sfp = (sfp / torch.norm(sfp)).to(pred.device)
        rs, rt = _rank_conventions(pred, sfp, model.ranker)
        recs.append({'cos': cos_sim(pred, sfp).item(), 'rank_strict': rs, 'rank_tie': rt})
    return recs


# --------------------------------------------------------------------------- #
# Per-run driver
# --------------------------------------------------------------------------- #
def build_args(params, ckpt):
    valid = {f.name for f in dc_fields(MARINAArgs)}
    kw = {k: v for k, v in params.items() if k in valid}
    kw.update(train=False, test=False, benchmark=True, load_from_checkpoint=ckpt)
    return MARINAArgs(**kw)


def eval_run(run_name, params_path, ckpt_path, device, batch_size, bench_splits, parts):
    params = json.load(open(params_path))
    if params.get('project_name', 'MARINA') != 'MARINA':
        raise ValueError(f"{run_name}: only MARINA runs are supported (got {params.get('project_name')})")
    args = build_args(params, ckpt_path)

    # Population is fixed to all-7-DATA molecules (same molecules for every model, so
    # cross-model numbers are comparable). The MODEL, however, may lack some modalities
    # (e.g. a formula-free or MS+-only run) -- run only the combos it supports, skip the
    # rest, rather than erroring the whole run.
    model_mods = set(args.input_types)
    all_test = build_test_combos()
    test_combos = OrderedDict((n, k) for n, k in all_test.items() if set(k) <= model_mods)
    bench_combos = OrderedDict((n, k) for n, k in BENCHMARK_COMBOS.items() if set(k) <= model_mods)
    skipped = [n for n in all_test if n not in test_combos] + \
              [f"bench:{n}" for n in BENCHMARK_COMBOS if n not in bench_combos]
    if skipped:
        print(f"[{run_name}] model input_types={sorted(model_mods)}; "
              f"skipping combos needing absent modalities: {skipped}")

    fp_loader = make_fp_loader(
        args.fp_type, entropy_out_dim=args.out_dim,
        retrieval_path=os.path.join(DATASET_ROOT, 'retrieval.pkl'),
    )
    model = MARINA(args, fp_loader)
    data_module = MARINADataModule(args, fp_loader)
    load_model(args, model)           # load_state_dict + setup_ranker + eval
    model = model.to(device)          # moves params AND ranker.data buffer

    rows, raw = [], {'test': {}, 'benchmark': {}}

    # (1) simulated test scores, fixed all-7 population -----------------------
    if parts in ('both', 'test'):
        cache = load_all7_population(args, fp_loader)
        print(f"[{run_name}] all-7 test population: n={len(cache)}")
        for combo, keys in test_combos.items():
            rs, rt, cos = score_test_combo(cache, keys, model, device, batch_size)
            m = _agg(rs, rt, cos)
            raw['test'][combo] = {'rank_strict': rs, 'rank_tie': rt, 'cos': cos}
            rows.append({'run': run_name, 'ckpt': ckpt_path, 'part': 'test',
                         'split': 'test_all7pop', 'combo': combo, **m})
            _print_metric_line(f"test/{combo}", m)

    # (2) benchmark (journal) derep, NMR+MW and +formula ---------------------
    if parts in ('both', 'benchmark'):
        journal_path = os.path.join(BENCHMARK_ROOT, 'benchmark-journal.pkl')
        journal = pickle.load(open(journal_path, 'rb'))
        for split in bench_splits:
            split_data = {k: v for k, v in journal.items() if v.get('split') == split}
            for combo, keys in bench_combos.items():
                recs = run_benchmark_loop(split_data, data_module, model, fp_loader, keys,
                                          device, desc=f"{run_name} bench/{split}/{combo}")
                m = _agg_recs(recs)
                raw['benchmark'].setdefault(split, {})[combo] = recs
                rows.append({'run': run_name, 'ckpt': ckpt_path, 'part': 'benchmark',
                             'split': split, 'combo': combo, **m})
                _print_metric_line(f"bench/{split}/{combo}", m)

    del model, data_module, fp_loader
    if device == 'cuda':
        torch.cuda.empty_cache()
    return rows, raw


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_metric_line(label, m):
    print(f"  {label:<28} n={m['n']:<5} cos={m['mean_cos']:.4f} | "
          f"strict @1/5/10 {m['strict_top1']:.2f}/{m['strict_top5']:.2f}/{m['strict_top10']:.2f} | "
          f"tie @1/5/10 {m['tie_top1']:.2f}/{m['tie_top5']:.2f}/{m['tie_top10']:.2f}")


def write_csv(rows, path):
    cols = ['run', 'ckpt', 'part', 'split', 'combo'] + METRIC_FIELDS
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs_dir', required=True, help='parent dir; each subdir is a run')
    ap.add_argument('--out_dir', default='eval_grapefruit_out')
    ap.add_argument('--ckpt_name', default=None,
                    help="exact checkpoint filename to prefer in each run dir "
                         "(default priority: 'best*' > last.ckpt > newest *.ckpt)")
    ap.add_argument('--batch_size', type=int, default=32,
                    help='batch size for the test-split ranking (bounds ranker memory)')
    ap.add_argument('--bench_splits', default='val,test',
                    help='comma-separated journal splits for the benchmark part')
    ap.add_argument('--parts', choices=['both', 'test', 'benchmark'], default='both',
                    help='which parts to run: test-split scores, journal benchmark, or both')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dry_run', action='store_true', help='list discovered runs and exit')
    a = ap.parse_args()

    runs = discover_runs(a.runs_dir, a.ckpt_name)
    if not runs:
        print(f"No runs found under {a.runs_dir}")
        return
    print(f"Discovered {len(runs)} run(s):")
    for name, params_path, ckpt in runs:
        print(f"  {name}\n    params: {params_path}\n    ckpt:   {ckpt}")
    if a.dry_run:
        return

    os.makedirs(a.out_dir, exist_ok=True)
    bench_splits = [s for s in a.bench_splits.split(',') if s]
    all_rows, all_raw = [], {}
    for name, params_path, ckpt in runs:
        print(f"\n=== {name} ===")
        try:
            rows, raw = eval_run(name, params_path, ckpt, a.device, a.batch_size, bench_splits, a.parts)
        except Exception as e:
            print(f"[{name}] ERROR: {e!r}")
            all_rows.append({'run': name, 'ckpt': ckpt, 'part': 'error',
                             'split': repr(e), 'combo': '',
                             **{f: '' for f in METRIC_FIELDS}})
            continue
        all_rows.extend(rows)
        all_raw[name] = raw
        write_csv(all_rows, os.path.join(a.out_dir, 'eval_grapefruit.csv'))   # checkpoint each run

    write_csv(all_rows, os.path.join(a.out_dir, 'eval_grapefruit.csv'))
    with open(os.path.join(a.out_dir, 'eval_grapefruit_raw.pkl'), 'wb') as f:
        pickle.dump(all_raw, f)
    print(f"\nWrote {os.path.join(a.out_dir, 'eval_grapefruit.csv')} "
          f"and eval_grapefruit_raw.pkl ({len(all_rows)} rows)")


if __name__ == '__main__':
    main()
