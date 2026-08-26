#!/usr/bin/env python3
"""10_build_journal.py -- prepare the Journal benchmark against the current index/splits.

The Journal (config.BENCH_JOURNAL) is a FROZEN curated input: hand-extracted shifts +
canonical SMILES + val/test label per NPID, produced once by curation. This stage does
NOT rebuild it. It loads the frozen set and attaches the leakage / retrieval metadata
that depends on the build being scored against, writing only:

  config.BENCH_JOURNAL_PREPARED  frozen set + canonical_2d_smiles + per-family leakage
                                 flags (marina1_split, spectre_split, marina_clean,
                                 spectre_clean, both_clean) + retrieval_idx

Inputs consumed: BENCH_JOURNAL (frozen), INDEX_PKL (split=... after 7_splits.py),
SPECTRE_SPLITS_PKL (5_spectre_splits.py), METADATA_JSON (8_assemble_arrow.py).

retrieval_idx is now always defined: 3_build_retrieval.py folds the journal SMILES into the
retrieval bank from the start, so no compound is 'novel' (novel count should be 0).

Conventions:
  - 'smiles' = frozen stereo-preserving canonical SMILES (from curation)
  - 'canonical_2d_smiles' = canonicalize_smiles(smiles, keep_stereo=False)  (2D key)
"""
import collections
import json
import pickle
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[3]))

from rdkit import RDLogger

from config import (BENCH_JOURNAL, BENCH_JOURNAL_PREPARED, INDEX_PKL, METADATA_JSON,
                    SPECTRE_SPLITS_PKL)
from src.modules.data.smiles import canonicalize_smiles

RDLogger.DisableLog('rdApp.*')


def split_smiles(index_path):
    """index.pkl -> {split: set(smiles)}."""
    by_split = collections.defaultdict(set)
    for entry in pickle.load(open(index_path, 'rb')).values():
        by_split[entry['split']].add(entry['smiles'])
    return by_split


def main():
    journal = pickle.load(open(BENCH_JOURNAL, 'rb'))
    print(f'Loaded {len(journal)} frozen Journal entries from {BENCH_JOURNAL}')

    # 2D canonical key for every entry (the frozen set stores only stereo SMILES).
    for entry in journal.values():
        smi = entry.get('smiles')
        entry['canonical_2d_smiles'] = (
            canonicalize_smiles(smi, keep_stereo=False) if smi else None)

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
        entry['retrieval_idx'] = retrieval.get(smi)   # None => absent from retrieval bank

    marina_clean = [k for k, v in journal.items() if v['marina_clean']]
    both_clean = [k for k, v in journal.items() if v['both_clean']]
    novel = sorted(k for k, v in journal.items() if v['retrieval_idx'] is None)

    report = {
        'n_total': len(journal),
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
        'novel_absent_from_retrieval': {'count': len(novel), 'npids': novel},
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
    print(f'  novel (absent from retrieval bank): {len(novel)}')
    if novel:
        print(f'  WARNING: expected 0 novel now that 3_build_retrieval folds journal in: {novel[:10]}')


if __name__ == '__main__':
    main()
