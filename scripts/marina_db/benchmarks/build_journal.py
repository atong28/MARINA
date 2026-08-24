#!/usr/bin/env python3
"""
Build the Journal benchmark from the CSVs in `filtered/`, canonicalize inline, and
attach the leakage / retrieval metadata in one pass.

This subsumes both the old Benchmark/build_journal.py and scripts/benchmark/journal_prep.py:
building the set AND canonicalizing SMILES at build time means journal_prep's
canonicalize-retrofit is unnecessary, and the leakage flags it computed are added here.

Writes:
  config.BENCH_JOURNAL           plain set (npid, smiles, canonical_2d_smiles, mw, input, split)
  config.BENCH_JOURNAL_PREPARED  plain set + per-family leakage flags + retrieval_idx

Prepared-entry fields reproduced from journal_prep:
  canonical_2d_smiles, marina1_split, spectre_split,
  marina_clean, spectre_clean, both_clean, retrieval_idx.

Conventions:
  - 'smiles' = canonicalize_smiles(source, keep_stereo=True)  (source kept stereo)
  - 'canonical_2d_smiles' = canonicalize_smiles(smiles, keep_stereo=False)  (2D key)
  - 1H excludes exchangeable protons; shifts sort descending; HSQC by 1H desc.
  - Split: preserve the annotated-benchmark val/test label per NPID; new NPIDs get a
    deterministic 50/50 split (seed 0) so recompiles are stable.
"""
import collections
import glob
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

from config import (BENCH_ROOT, BENCH_FILTERED, BENCH_ANNOTATED, BENCH_JOURNAL,
                    BENCH_JOURNAL_PREPARED, INDEX_PKL, METADATA_JSON,
                    SPECTRE_SPLITS_PKL, SEED)
from lib.csv_peaks import read_1h_csv, read_13c_csv, read_hsqc_csv
from src.modules.data.smiles import canonicalize_smiles

RDLogger.DisableLog('rdApp.*')


def load_npid_smiles():
    """npid -> SMILES: prefer the curated annotated benchmark, fall back to the NP-MRD
    dump and the live-fetched structures for newer NPIDs."""
    bench = pickle.load(open(BENCH_ANNOTATED, 'rb'))
    npid_smiles = {e['npid']: e['smiles'] for e in bench.values() if e.get('smiles')}
    for jf in glob.glob(os.path.join(BENCH_ROOT, 'npmrd_json', '*.json')):
        for r in json.load(open(jf))['np_mrd']['natural_product']:
            acc, smi = r.get('accession'), r.get('smiles')
            if acc and smi and acc not in npid_smiles:
                npid_smiles[acc] = smi
    fetched = BENCH_ROOT / 'npmrd_fetched_smiles.json'
    if fetched.exists():
        for acc, rec in json.load(open(fetched)).items():
            if rec.get('smiles') and acc not in npid_smiles:
                npid_smiles[acc] = rec['smiles']
    return bench, npid_smiles


def meta_for(npid, npid_smiles):
    """Canonical stereo SMILES + monoisotopic exact mass. ('', 0.0) if unavailable."""
    smi = npid_smiles.get(npid, '')
    if smi:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            return {'smiles': canonicalize_smiles(smi, keep_stereo=True) or '',
                    'mw': float(rdMolDescriptors.CalcExactMolWt(m))}
    return {'smiles': '', 'mw': 0.0}


def split_smiles(index_path):
    """index.pkl -> {split: set(smiles)}."""
    by_split = collections.defaultdict(set)
    for entry in pickle.load(open(index_path, 'rb')).values():
        by_split[entry['split']].add(entry['smiles'])
    return by_split


def main():
    bench, npid_smiles = load_npid_smiles()

    journal, skipped, issues = {}, [], []
    for npid in sorted(os.listdir(BENCH_FILTERED)):
        d = os.path.join(BENCH_FILTERED, npid)
        if not os.path.isdir(d) or not npid.startswith('NP'):
            continue

        h_path = os.path.join(d, '1H.csv')
        c_path = os.path.join(d, '13C.csv')
        s_path = os.path.join(d, 'HSQC.csv')
        if not any(os.path.exists(p) for p in (h_path, c_path, s_path)):
            skipped.append(npid)            # no CSVs = no paper
            continue

        h_shifts = read_1h_csv(h_path) if os.path.exists(h_path) else []
        c_shifts = read_13c_csv(c_path) if os.path.exists(c_path) else []
        hsqc_data = read_hsqc_csv(s_path) if os.path.exists(s_path) else []
        if not h_shifts and not c_shifts:
            issues.append(f'{npid}: both 1H and 13C empty after parsing')
            skipped.append(npid)
            continue

        h_shifts = sorted(h_shifts, reverse=True)
        c_shifts = sorted(c_shifts, reverse=True)
        hsqc_data = sorted(hsqc_data, key=lambda x: -x[1])

        meta = meta_for(npid, npid_smiles)
        smiles = meta['smiles']
        mw = float(meta['mw'])
        canon2d = canonicalize_smiles(smiles, keep_stereo=False) if smiles else None

        journal[npid] = {
            'npid': npid,
            'smiles': smiles,
            'canonical_2d_smiles': canon2d,
            'mw': mw,
            'input': {
                'h_nmr': torch.tensor([[v] for v in h_shifts], dtype=torch.float32),
                'c_nmr': torch.tensor([[v] for v in c_shifts], dtype=torch.float32),
                'hsqc': torch.tensor(hsqc_data, dtype=torch.float32) if hsqc_data
                        else torch.zeros((0, 3), dtype=torch.float32),
                'mw': mw,
            },
        }

    # --- split assignment: preserve annotated split by NPID, new NPIDs 50/50 (seed 0) ---
    orig_split = {e['npid']: e['split'] for e in bench.values()
                  if e.get('npid') and e.get('split')}
    new_npids = sorted(npid for npid in journal if npid not in orig_split)
    random.Random(SEED).shuffle(new_npids)
    half = len(new_npids) // 2
    new_split = {npid: ('val' if i < half else 'test')
                 for i, npid in enumerate(new_npids)}
    for npid, entry in journal.items():
        entry['split'] = orig_split.get(npid) or new_split[npid]

    print(f'Built {len(journal)} entries, skipped {len(skipped)}')
    print(f'  preserved from annotated benchmark: '
          f'{sum(1 for n in journal if n in orig_split)}; new 50/50: {len(new_npids)}')
    if issues:
        print(f'Issues ({len(issues)}):')
        for i in issues:
            print(' ', i)

    with open(BENCH_JOURNAL, 'wb') as f:
        pickle.dump(journal, f)
    print(f'Saved plain set -> {BENCH_JOURNAL}')

    # --- prepare: leakage flags + retrieval index (folds in journal_prep) ---
    marina = split_smiles(INDEX_PKL)
    spectre = pickle.load(open(SPECTRE_SPLITS_PKL, 'rb'))
    spectre = {s: spectre.get(s, set()) for s in ('train', 'val', 'test')}
    meta = json.load(open(METADATA_JSON))
    retrieval = {e['canonical_2d_smiles']: int(i) for i, e in meta.items()}

    unparseable = []
    for npid, entry in journal.items():
        smi = entry['canonical_2d_smiles']
        if smi is None:
            unparseable.append(npid)
        entry['marina1_split'] = next(
            (s for s in ('train', 'val', 'test') if smi in marina[s]), 'absent')
        entry['spectre_split'] = next(
            (s for s in ('train', 'val', 'test') if smi in spectre[s]), 'absent')
        entry['marina_clean'] = entry['marina1_split'] != 'train'
        entry['spectre_clean'] = entry['spectre_split'] != 'train'
        entry['both_clean'] = entry['marina_clean'] and entry['spectre_clean']
        entry['retrieval_idx'] = retrieval.get(smi)   # None => needs appending

    marina_clean = [k for k, v in journal.items() if v['marina_clean']]
    both_clean = [k for k, v in journal.items() if v['both_clean']]
    novel = sorted(k for k, v in journal.items() if v['retrieval_idx'] is None)

    report = {
        'n_total': len(journal),
        'n_skipped': len(skipped),
        'unparseable': unparseable,
        'canonical_smiles_in_retrieval': sum(
            1 for v in journal.values() if v['canonical_2d_smiles'] in retrieval),
        'marina1_split_counts': dict(collections.Counter(
            v['marina1_split'] for v in journal.values())),
        'spectre_split_counts': dict(collections.Counter(
            v['spectre_split'] for v in journal.values())),
        'usable': {
            'marina_only': len(marina_clean),
            'both_families': len(both_clean),
            'both_by_journal_split': dict(collections.Counter(
                journal[k]['split'] for k in both_clean)),
            'marina_by_journal_split': dict(collections.Counter(
                journal[k]['split'] for k in marina_clean)),
        },
        'novel_needing_retrieval_rows': {'count': len(novel), 'npids': novel},
    }

    with open(BENCH_JOURNAL_PREPARED, 'wb') as f:
        pickle.dump(journal, f)
    with open(BENCH_JOURNAL_PREPARED.with_name('journal_prep_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f'Saved prepared set -> {BENCH_JOURNAL_PREPARED}')
    print(f"  canonical in retrieval: {report['canonical_smiles_in_retrieval']}"
          f"/{len(journal)}")
    print(f"  MARINA splits: {report['marina1_split_counts']}")
    print(f"  SPECTRE splits: {report['spectre_split_counts']}")
    print(f'  usable: MARINA {len(marina_clean)}, both families {len(both_clean)}')
    print(f'  novel (need retrieval rows): {len(novel)}')


if __name__ == '__main__':
    main()
