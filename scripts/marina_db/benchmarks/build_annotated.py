#!/usr/bin/env python3
"""
Assemble the expanded Annotated benchmark from the CSVs in `filtered/`.

Refactor of the old Benchmark/build_benchmark.py: paths from config, CSV parsing from
lib.csv_peaks, one canonicalize_smiles. The curated seed (byte-for-byte carried over,
splits included) is read from config.BENCH_ANNOTATED; new complete-CSV compounds are
folded in and the merged set is written back (or to --out).

Entry structure matches the seed exactly:
  {int: {'npid': str, 'smiles': str, '3d_smiles': str, 'split': str,
         'input': {'h_nmr': Tensor[N,1], 'c_nmr': Tensor[M,1],
                   'hsqc': Tensor[K,3], 'mw': float}}}

Conventions (verified against the original):
  - 'smiles' is canonicalize_smiles('3d_smiles') -- stereo-stripped fixed point.
  - 'mw' is CalcExactMolWt (monoisotopic).
  - Shifts sort descending; HSQC sorts by 1H shift descending.
  - Splits are per-compound random (seed 0), NOT grouped by source paper.

Note: 1H peaks now come from lib.csv_peaks.read_1h_csv, which drops exchangeable
(OH/NH/COOH) protons -- the h_nmr = "protons with a 13C partner" convention already
baked into the seed entries. This is the one behavioural change from the old builder's
read_1d_csv, which kept them for the new compounds.
"""
import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[3]))

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from config import BENCH_ROOT, BENCH_FILTERED, BENCH_ANNOTATED, RETRIEVAL_PKL, SEED
from lib.csv_peaks import read_1h_csv, read_13c_csv, read_hsqc_csv
from src.modules.data.smiles import canonicalize_smiles

RDLogger.DisableLog('rdApp.*')


# --- structure lookup ------------------------------------------------------------

def load_structures():
    """npid -> isomeric SMILES, from every local NP-MRD source."""
    out = {}
    dump = BENCH_ROOT / 'npmrd_json' / 'npmrd_natural_products_NP0300001_NP0350000.json'
    for r in json.load(open(dump))['np_mrd']['natural_product']:
        if r.get('accession') and r.get('smiles'):
            out[r['accession']] = r['smiles']
    for npid, r in json.load(open(BENCH_ROOT / 'npmrd_index.json')).items():
        if r.get('smiles'):
            out.setdefault(npid, r['smiles'])
    missing = BENCH_ROOT / 'missing_structures.json'
    if missing.exists():
        for npid, r in json.load(open(missing)).items():
            if r.get('smiles'):
                out.setdefault(npid, r['smiles'])
    return out


def complete_npids():
    out = []
    for npid in sorted(os.listdir(BENCH_FILTERED)):
        d = os.path.join(BENCH_FILTERED, npid)
        if not npid.startswith('NP') or not os.path.isdir(d):
            continue
        if all(os.path.exists(os.path.join(d, f))
               for f in ('13C.csv', '1H.csv', 'HSQC.csv')):
            out.append(npid)
    return out


# --- build -----------------------------------------------------------------------

def build_entry(npid, iso_smiles):
    """Return an entry dict, or (None, reason) if this compound cannot be assembled."""
    d = os.path.join(BENCH_FILTERED, npid)
    h = sorted(read_1h_csv(os.path.join(d, '1H.csv')), reverse=True)
    c = sorted(read_13c_csv(os.path.join(d, '13C.csv')), reverse=True)
    hsqc = sorted(read_hsqc_csv(os.path.join(d, 'HSQC.csv')), key=lambda x: -x[1])

    if not h and not c:
        return None, 'both 1H and 13C empty after parsing'
    if not hsqc:
        return None, 'HSQC empty after parsing'

    mol = Chem.MolFromSmiles(iso_smiles)
    if mol is None:
        return None, 'RDKit cannot parse the NP-MRD SMILES'
    flat = canonicalize_smiles(iso_smiles)
    if flat is None:
        return None, 'SMILES has no canonical fixed point'
    mw = float(rdMolDescriptors.CalcExactMolWt(mol))

    return {
        'npid': npid,
        'smiles': flat,
        '3d_smiles': iso_smiles,
        'split': None,                       # assigned below
        'input': {
            'h_nmr': torch.tensor([[v] for v in h], dtype=torch.float32),
            'c_nmr': torch.tensor([[v] for v in c], dtype=torch.float32),
            'hsqc': torch.tensor(hsqc, dtype=torch.float32),
            'mw': mw,
        },
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=BENCH_ANNOTATED)
    a = ap.parse_args()
    out_pkl = a.out
    out_report = out_pkl.with_name(out_pkl.stem + '-report.json')

    existing = pickle.load(open(BENCH_ANNOTATED, 'rb'))   # curated seed
    have = {e['npid'] for e in existing.values()}
    structures = load_structures()

    report = {'existing_entries': len(existing)}
    new_npids = [n for n in complete_npids() if n not in have]
    report['new_candidates'] = len(new_npids)

    built, rejected = {}, {}
    for npid in new_npids:
        iso = structures.get(npid)
        if not iso:
            rejected[npid] = 'no structure in any NP-MRD source'
            continue
        entry, reason = build_entry(npid, iso)
        if entry is None:
            rejected[npid] = reason
            continue
        built[npid] = entry

    report['new_built'] = len(built)
    report['new_rejected'] = rejected

    # Splits: preserve every existing assignment, then balance the new ones toward 50/50.
    n_val = sum(1 for e in existing.values() if e['split'] == 'val')
    order = sorted(built)
    random.Random(SEED).shuffle(order)
    target_val = (len(existing) + len(built)) // 2 - n_val
    for i, npid in enumerate(order):
        built[npid]['split'] = 'val' if i < target_val else 'test'

    merged = {}
    for i, e in enumerate(list(existing.values()) + [built[n] for n in sorted(built)]):
        merged[i] = e

    report['total_entries'] = len(merged)
    report['split_counts'] = {
        s: sum(1 for e in merged.values() if e['split'] == s) for s in ('val', 'test')
    }

    # --- checks ---------------------------------------------------------------
    # Duplicate canonical SMILES (the seed already carries a few): reported, not enforced.
    by_smiles = {}
    for e in merged.values():
        by_smiles.setdefault(e['smiles'], []).append(e['npid'])
    dupes = {s: v for s, v in by_smiles.items() if len(v) > 1}
    report['unique_canonical_smiles'] = len(by_smiles)
    report['duplicate_groups'] = dupes

    # Retrieval-set presence. A benchmark molecule absent from retrieval silently caps
    # every top-k.
    if os.path.exists(RETRIEVAL_PKL):
        retrieval = {v['smiles']
                     for v in pickle.load(open(RETRIEVAL_PKL, 'rb')).values()}
        absent = sorted({e['npid'] for e in merged.values()
                         if e['smiles'] not in retrieval})
        report['absent_from_retrieval'] = absent

    # Shape sanity.
    bad = []
    for e in merged.values():
        inp = e['input']
        if inp['hsqc'].ndim != 2 or inp['hsqc'].shape[1] != 3:
            bad.append((e['npid'], 'hsqc shape'))
        for k in ('h_nmr', 'c_nmr'):
            if inp[k].ndim != 2 or inp[k].shape[1] != 1:
                bad.append((e['npid'], f'{k} shape'))
        if any(torch.isnan(inp[k]).any() for k in ('h_nmr', 'c_nmr', 'hsqc')):
            bad.append((e['npid'], 'NaN'))
    report['shape_failures'] = bad

    with open(out_pkl, 'wb') as f:
        pickle.dump(merged, f)
    with open(out_report, 'w') as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f'\n{len(merged)} entries -> {out_pkl}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
