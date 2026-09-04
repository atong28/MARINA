#!/usr/bin/env python3
"""
Produce one `metrics.json` per checkpoint for the MARINA website's dynamic model
selector (schema v1.0). For each checkpoint it evaluates EVERY non-empty subset of
the model's input_types and records, per combo, how well the checkpoint ranks the
true structure (strict, 0-based, against the checkpoint's own rankingset.pt).

Two sources per the contract, each combo resolves to exactly one:
  * benchmark (real-world journal, benchmark-journal.pkl) for combos the journal can
    supply -- i.e. subsets of the data-derived coverable set (NMR + mw, + formula
    derived from SMILES; NO MS channels). Three rows each: val, test, both (pooled).
    Combos are scored require-all-per-entry: only journal entries that actually carry
    every modality in the combo are used, so `modalities` is the EXACT set fed.
  * test (simulated) for the remaining combos (those containing mass_spec /
    mass_spec_neg), one row each (test_all7pop): a fixed population of test-split
    molecules that have all of the model's modalities present, masked down per combo.

Metrics: mean_cos in [0,1]; strict_top{1,5,10} as percents in [0,100]; null (never
NaN) when n==0. Ranking + tie convention reuse src/modules/core/ranker.py
(dot_prod_rank strict) and src/modules/benchmark.py::_rank_conventions.

Run from the MARINA repo root:
    python scripts/benchmark/eval_metrics_json.py --runs_dir /path/to/models
    python scripts/benchmark/eval_metrics_json.py --runs_dir /path/to/models --dry_run

Writes checkpoints/<model_dir>/metrics.json next to each params.json.
"""
import argparse
import datetime as dt
import glob
import json
import os
import pickle
from dataclasses import fields as dc_fields
from itertools import combinations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.modules import MARINA, MARINAArgs, MARINADataModule
from src.modules.marina.dataset import MARINADataset, collate
from src.modules.data.fp_loader import make_fp_loader
from src.modules.benchmark import filter_data, formula_vec_from_smiles, _rank_conventions, cos_sim
from src.modules.core.const import (
    BENCHMARK_ROOT, DATASET_ROOT, INPUTS_CANONICAL_ORDER,
)

SCHEMA_VERSION = "1.0"
EVAL_SCRIPT = "scripts/benchmark/eval_metrics_json.py"


def canon(mods):
    """Sort a modality iterable into canonical order."""
    s = set(mods)
    return [m for m in INPUTS_CANONICAL_ORDER if m in s]


def nonempty_subsets(input_types):
    """All non-empty subsets of input_types, each in canonical order."""
    it = canon(input_types)
    for r in range(1, len(it) + 1):
        for c in combinations(it, r):
            yield list(c)


def journal_coverable_set(journal):
    """Modalities the journal can supply: union of per-entry input keys, plus 'formula'
    (always derivable on the fly from SMILES). MS channels are absent from the journal,
    so combos containing them fall out as not-coverable automatically."""
    cov = set()
    for e in journal.values():
        cov |= set(e.get('input', {}).keys())
    cov.add('formula')
    return cov


def _to_device(batch_inputs, device):
    return {k: v.to(device) for k, v in batch_inputs.items()}


def _agg(ranks, cosines):
    """ranks: list/1D of 0-based strict ranks; cosines: matching cosines. -> schema dict."""
    n = len(ranks)
    if n == 0:
        return {'n': 0, 'metrics': {'mean_cos': None, 'strict_top1': None,
                                    'strict_top5': None, 'strict_top10': None}}
    ranks = [int(r) for r in ranks]
    mean_cos = sum(cosines) / n
    return {
        'n': n,
        'metrics': {
            'mean_cos': round(mean_cos, 4),
            'strict_top1': round(100.0 * sum(r < 1 for r in ranks) / n, 2),
            'strict_top5': round(100.0 * sum(r < 5 for r in ranks) / n, 2),
            'strict_top10': round(100.0 * sum(r < 10 for r in ranks) / n, 2),
        },
    }


# --------------------------------------------------------------------------- #
# Simulated (test split): fixed all-modalities-present population, batched.
# --------------------------------------------------------------------------- #
def load_full_population(args, fp_loader):
    """Test-split molecules with ALL of the model's modalities present; each item carries
    every modality (masked down per combo later). Returns list of (input_dict, fp)."""
    pop_types = canon(args.input_types)
    ds = MARINADataset(args, fp_loader, split='test', override_input_types=pop_types)
    return [ds[i] for i in range(len(ds))]


@torch.no_grad()
def score_sim(cache, keys, model, device, batch_size):
    keyset = set(keys)
    ranks, cosines = [], []
    for start in range(0, len(cache), batch_size):
        chunk = cache[start:start + batch_size]
        batch = [(filter_data(d, keyset), fp) for d, fp in chunk]
        batch_inputs, batch_fps = collate(batch)
        batch_inputs = _to_device(batch_inputs, device)
        batch_fps = batch_fps.to(device)
        queries = torch.sigmoid(model(batch=batch_inputs))
        rs = model.ranker.batched_rank(queries, batch_fps, tie_aware=False)  # strict
        qn = F.normalize(queries, dim=1, p=2.0)
        tn = F.normalize(batch_fps.float(), dim=1, p=2.0)
        cos = (qn * tn).sum(dim=1)
        ranks.extend(rs.cpu().tolist())
        cosines.extend(cos.cpu().tolist())
    return ranks, cosines


# --------------------------------------------------------------------------- #
# Benchmark (journal): require-all-per-entry, per-entry forward (small n).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_bench(entries, keys, model, data_module, fp_loader, device):
    """Score `keys` on journal `entries`, using only entries that carry EVERY modality in
    `keys` (formula is always derivable from SMILES, so it never gates). Feeds exactly
    `keys`. Returns (ranks, cosines) lists (strict)."""
    keyset = set(keys)
    ranks, cosines = [], []
    for e in entries:
        if not all(m == 'formula' or m in e['input'] for m in keys):
            continue
        raw = dict(e['input'])
        if 'formula' in keyset:
            raw['formula'] = formula_vec_from_smiles(e['smiles'])
        clean = {m: raw[m] for m in keys}
        inputs = data_module.format_inference_data(clean)
        inputs = {'batch': _to_device(inputs['batch'], device)}
        pred = torch.sigmoid(model(**inputs)[0])
        sfp = fp_loader.build_mfp_for_smiles(e['smiles'])
        sfp = (sfp / torch.norm(sfp)).to(pred.device)
        rs, _ = _rank_conventions(pred, sfp, model.ranker)  # strict rank (0-based)
        ranks.append(rs)
        cosines.append(cos_sim(pred, sfp).item())
    return ranks, cosines


# --------------------------------------------------------------------------- #
# Discovery + per-model driver
# --------------------------------------------------------------------------- #
def _pick_ckpt(candidates, ckpt_name):
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
        same_dir = [c for c in ckpts if os.path.dirname(c) == os.path.dirname(params_path)]
        runs.append((os.path.basename(sub), params_path, _pick_ckpt(same_dir or ckpts, ckpt_name)))
    return runs


def build_args(params, ckpt):
    valid = {f.name for f in dc_fields(MARINAArgs)}
    kw = {k: v for k, v in params.items() if k in valid}
    kw.update(train=False, test=False, benchmark=True, load_from_checkpoint=ckpt)
    return MARINAArgs(**kw)


def eval_model(run_name, params_path, ckpt_path, device, batch_size, model_id):
    params = json.load(open(params_path))
    if params.get('project_name', 'MARINA') != 'MARINA':
        raise ValueError(f"{run_name}: only MARINA is supported (got {params.get('project_name')})")
    args = build_args(params, ckpt_path)
    input_types = canon(args.input_types)

    fp_loader = make_fp_loader(
        args.fp_type, entropy_out_dim=args.out_dim,
        retrieval_path=os.path.join(DATASET_ROOT, 'retrieval.pkl'),
    )
    model = MARINA(args, fp_loader)
    data_module = MARINADataModule(args, fp_loader)

    ckpt = torch.load(ckpt_path, map_location='cpu')          # single load -> weights + epoch
    ckpt_epoch = ckpt.get('epoch')
    model.load_state_dict(ckpt['state_dict'])
    del ckpt
    model.setup_ranker()
    model.eval()
    model = model.to(device)

    # Classify every non-empty subset: benchmark-coverable (subset of the data-derived
    # coverable set) vs simulated (everything else, i.e. anything with an MS channel).
    journal = pickle.load(open(os.path.join(BENCHMARK_ROOT, 'benchmark-journal.pkl'), 'rb'))
    coverable = journal_coverable_set(journal)
    val_entries = [v for v in journal.values() if v.get('split') == 'val']
    test_entries = [v for v in journal.values() if v.get('split') == 'test']

    subsets = list(nonempty_subsets(input_types))
    bench_combos = [s for s in subsets if set(s) <= coverable]
    sim_combos = [s for s in subsets if not set(s) <= coverable]
    print(f"[{run_name}] {len(subsets)} combos: {len(bench_combos)} benchmark, {len(sim_combos)} simulated")

    measurements = []

    # benchmark: val + test + both, per coverable combo
    for keys in tqdm(bench_combos, desc=f"{run_name} benchmark"):
        vr, vc = score_bench(val_entries, keys, model, data_module, fp_loader, device)
        tr, tc = score_bench(test_entries, keys, model, data_module, fp_loader, device)
        for split, (rk, cs) in (('val', (vr, vc)), ('test', (tr, tc)), ('both', (vr + tr, vc + tc))):
            agg = _agg(rk, cs)
            measurements.append({'modalities': keys, 'eval_set': 'benchmark', 'split': split,
                                 'n': agg['n'], 'metrics': agg['metrics']})

    # simulated: one fixed all-modalities population, mask down per MS-containing combo
    if sim_combos:
        cache = load_full_population(args, fp_loader)
        print(f"[{run_name}] simulated population n={len(cache)}")
        for keys in tqdm(sim_combos, desc=f"{run_name} simulated"):
            rk, cs = score_sim(cache, keys, model, device, batch_size)
            agg = _agg(rk, cs)
            measurements.append({'modalities': keys, 'eval_set': 'test', 'split': 'test_all7pop',
                                 'n': agg['n'], 'metrics': agg['metrics']})
        del cache

    doc = {
        'schema_version': SCHEMA_VERSION,
        'model_id': model_id or run_name,
        'input_types': input_types,
        'generated_at': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'eval_provenance': {
            'ckpt': os.path.basename(ckpt_path),
            'ckpt_epoch': ckpt_epoch,
            'dataset': os.path.basename(DATASET_ROOT.rstrip('/')),
            'rankingset': f"{args.fp_type}/rankingset.pt",
            'test_population': "test split, molecules with all of the model's modalities present",
            'benchmark_source': 'benchmark-journal.pkl',
            'eval_script': EVAL_SCRIPT,
            'notes': "benchmark combos scored require-all-per-entry (exact input set fed); "
                     "simulated combos scored on the fixed all-modalities test population.",
        },
        'measurements': measurements,
    }
    out_path = os.path.join(os.path.dirname(params_path), 'metrics.json')
    with open(out_path, 'w') as f:
        json.dump(doc, f, indent=2)
    print(f"[{run_name}] wrote {out_path} ({len(measurements)} measurements)")

    del model, data_module, fp_loader
    if device == 'cuda':
        torch.cuda.empty_cache()
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs_dir', required=True, help='parent dir; each subdir is a model')
    ap.add_argument('--ckpt_name', default=None,
                    help="preferred checkpoint filename (default: 'best*' > last.ckpt > newest *.ckpt)")
    ap.add_argument('--model_id', default=None,
                    help='override model_id (only meaningful with a single run; else the dir name is used)')
    ap.add_argument('--batch_size', type=int, default=32, help='batch size for simulated ranking')
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

    model_id = a.model_id if len(runs) == 1 else None
    for name, params_path, ckpt in runs:
        print(f"\n=== {name} ===")
        try:
            eval_model(name, params_path, ckpt, a.device, a.batch_size, model_id)
        except Exception as e:
            print(f"[{name}] ERROR: {e!r}")


if __name__ == '__main__':
    main()
