#!/usr/bin/env python3
"""
Build the Simulated benchmark from a shift-prediction model's JSONL output.

Ported from scripts/benchmark/port_benchmark.ipynb: each JSONL line carries a molecule's
predicted 1H/13C shifts; peaks are re-assembled into h_nmr / c_nmr / hsqc and paired with
the 2D-canonical SMILES. Only HSQC-status==SUCCESS records with non-empty H and C survive.

Entry structure:
  {int: {'input': {'h_nmr': Tensor[N,1], 'c_nmr': Tensor[M,1], 'hsqc': Tensor[K,3]},
         'smiles': str}}
"""
import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[3]))

import torch
from tqdm import tqdm

from config import BENCH_ROOT, BENCH_SIMULATED
from src.modules.data.smiles import canonicalize_smiles


def process_file(path):
    """JSONL -> {smiles: {'h_nmr', 'c_nmr', 'atoms'}} for successful HSQC predictions."""
    nmrs = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            if data['predictions']['hsqc']['status'] != 'SUCCESS':
                continue
            h_nmr = data['predictions']['hsqc']['H']
            c_nmr = data['predictions']['hsqc']['C']
            if h_nmr is None or c_nmr is None or len(h_nmr) == 0 or len(c_nmr) == 0:
                continue
            nmrs[data['smiles']] = {'h_nmr': h_nmr, 'c_nmr': c_nmr, 'atoms': data['atoms']}
    return nmrs


def _atom_sign_from_name(atom_name: str) -> int:
    return -1 if 'CH2' in atom_name else +1


def assemble_nmr_data(preds: Dict[str, Any]) -> Dict[str, List]:
    data = {'h_nmr': [], 'c_nmr': [], 'hsqc': [],
            'h_nmr_error': [], 'c_nmr_error': [], 'hsqc_error': []}

    atom_name_by_idx: Dict[int, str] = {int(a['number']): a['name'] for a in preds['atoms']}

    c_by_atom: Dict[int, Tuple[float, float]] = {}
    for c in preds['c_nmr']:
        for atom in c['atom']:
            atom_idx = int(atom['index'])
            c_shift = float(c['shift']['value'])
            c_err = float(c['shift']['error'])
            c_by_atom[atom_idx] = (c_shift, c_err)
            data['c_nmr'].append(c_shift)
            data['c_nmr_error'].append(c_err)

    for h in preds['h_nmr']:
        for atom in h['atom']:
            atom_idx = atom['index']
            h_shift = float(h['shift']['value'])
            h_err = float(h['shift']['error'])
            data['h_nmr'].append(h_shift)
            data['h_nmr_error'].append(h_err)

            if atom_idx not in c_by_atom:
                if atom_name_by_idx[atom_idx] in ('CH', 'CH2', 'CH3'):
                    raise ValueError()
                continue
            c_shift, c_err = c_by_atom[atom_idx]
            sign = _atom_sign_from_name(atom_name_by_idx[atom_idx])
            data['hsqc'].append([c_shift, h_shift, sign])
            data['hsqc_error'].append([c_err, h_err, 0.0])

    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', type=Path, default=BENCH_ROOT / 'benchmark_sim_3d.jsonl')
    ap.add_argument('--out', type=Path, default=BENCH_SIMULATED)
    a = ap.parse_args()

    all_nmrs = process_file(a.jsonl)
    nmr_data = {smiles: assemble_nmr_data(nmr) for smiles, nmr in tqdm(all_nmrs.items())}

    benchmark_data = {
        idx: {
            'input': {
                'h_nmr': torch.tensor(nmr_data[smiles]['h_nmr']).reshape(-1, 1),
                'c_nmr': torch.tensor(nmr_data[smiles]['c_nmr']).reshape(-1, 1),
                'hsqc': torch.tensor(nmr_data[smiles]['hsqc']),
            },
            'smiles': canonicalize_smiles(smiles),
        }
        for idx, smiles in enumerate(nmr_data.keys())
    }

    with open(a.out, 'wb') as f:
        pickle.dump(benchmark_data, f)
    print(f'{len(benchmark_data)} entries -> {a.out}')


if __name__ == '__main__':
    main()
