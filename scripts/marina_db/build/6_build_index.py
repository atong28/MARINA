"""Assemble spectral data and build the index (has_* flags + MW filter).

This is generate_dataset.py minus the train/val/test decision: every index entry is
written with split=None. Split assignment (and the benchmark->test forcing that used
to live here) is deferred to 7_splits.py, which also writes the per-split jsonl.

Retrieval is NOT built here -- it is the structure track (3_build_retrieval.py) so the
rankingsets are not blocked on spectral data. MS/MS is OPTIONAL: if data/raw/ms_predictions
is empty (positive re-prediction in flight), the build proceeds with has_mass_spec=False
and the spectral MS source simply absent.
"""
import json
from typing import Any, Dict, List, Tuple
import pickle
import os
from rdkit import Chem
import glob
from tqdm import tqdm
from rdkit.Chem import rdMolDescriptors

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # scripts/marina_db -> config
sys.path.insert(0, str(_HERE.parents[3]))   # repo root -> src
from config import DATA_RAW, DATA_CLEANED, INDEX_PKL, MW_MAX_EXACT, MIN_HEAVY_ATOMS, SOURCE_PRIORITY
from src.modules.data.smiles import canonicalize_smiles

def initialize_mapping(data, smiles: str) -> Dict[str, Any]:
    if smiles not in data:
        data[smiles] = {}
    if 'hsqc' not in data[smiles]:
        data[smiles]['hsqc'] = {}
    if 'c_nmr' not in data[smiles]:
        data[smiles]['c_nmr'] = {}
    if 'h_nmr' not in data[smiles]:
        data[smiles]['h_nmr'] = {}
    if 'mass_spec' not in data[smiles]:
        data[smiles]['mass_spec'] = {}
    if 'idx' not in data[smiles]:
        data[smiles]['idx'] = None
    data[smiles]['smiles'] = smiles
    return data[smiles]

def priority_access(data, key_order: List[str]) -> Any:
    for key in key_order:
        if key in data and data[key] is not None:
            return data[key]
    return []

def _atom_sign_from_name(atom_name: str) -> int:
    return -1 if "CH2" in atom_name else +1

def assemble_nmr_data(preds: Dict[str, Any]) -> Dict[str, List]:
    data = {
        "h_nmr": [],
        "c_nmr": [],
        "hsqc": [],
        "h_nmr_error": [],
        "c_nmr_error": [],
        "hsqc_error": [],
    }

    atom_name_by_idx: Dict[int, str] = {}
    for a in preds['atoms']:
        idx = int(a["number"])
        atom_name_by_idx[idx] = a['name']

    c_by_atom: Dict[int, Tuple[float, float]] = {}
    for c in preds['c_nmr']:
        for atom in c['atom']:
            atom_idx = atom['index']
            c_shift = float(c['shift']['value'])
            c_err = float(c['shift']['error'])
            c_by_atom[int(atom_idx)] = (c_shift, c_err)
            data["c_nmr"].append(c_shift)
            data["c_nmr_error"].append(c_err)

    for h in preds['h_nmr']:
        for atom in h['atom']:
            atom_idx = atom['index']
            h_shift = float(h['shift']['value'])
            h_err = float(h['shift']['error'])
            data["h_nmr"].append(h_shift)
            data["h_nmr_error"].append(h_err)

            if atom_idx not in c_by_atom:
                if atom_name_by_idx[atom_idx] in ('CH', 'CH2', 'CH3'):
                    raise ValueError()
                continue
            c_shift, c_err = c_by_atom[atom_idx]
            sign = _atom_sign_from_name(atom_name_by_idx[atom_idx])

            data["hsqc"].append([c_shift, h_shift, sign])
            data["hsqc_error"].append([c_err, h_err, 0.0])

    return data
def process_file(file):
    nmrs = {}
    with open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['predictions']['hsqc']['status'] != 'SUCCESS':
                continue
            h_nmr = data['predictions']['hsqc']['H']
            c_nmr = data['predictions']['hsqc']['C']
            if h_nmr is None or c_nmr is None or len(h_nmr) == 0 or len(c_nmr) == 0:
                continue
            nmrs[canonicalize_smiles(data['smiles'])] = {
                'h_nmr': h_nmr,
                'c_nmr': c_nmr,
                'atoms': data['atoms']
            }
    return nmrs

def process_mnova_predictions():
    files = sorted(glob.glob(str(DATA_RAW / 'mnova_predictions' / '*.jsonl')))
    nmr_data = {}
    for file in tqdm(files, desc='Processing Mnova predictions'):
        nmrs = process_file(file)
        for smiles, nmr in nmrs.items():
            nmr_data[smiles] = assemble_nmr_data(nmr)
    return nmr_data

def process_ms_predictions():
    files = sorted(glob.glob(str(DATA_RAW / 'ms_predictions' / '*.json')))
    if not files:
        print('WARNING: no MS/MS predictions in data/raw/ms_predictions -- building with '
              'has_mass_spec=False (positive re-prediction in flight; see 1_download.sh).')
        return {}
    ms_data = {}
    for file in tqdm(files, desc='Processing MS predictions'):
        ms = json.load(open(file, 'r'))
        for data in ms:
            ms_data[canonicalize_smiles(data['SMILES'])] = data['peaks']
    return ms_data

def build_spectral_data() -> Dict[str, Dict]:
    print('Building spectral data...')
    os.makedirs(DATA_CLEANED, exist_ok=True)
    # load old moonshotdatasetv3, map to key spectre
    index = pickle.load(open(DATA_RAW / 'index.pkl', 'rb'))
    mapping = {}
    for idx, data in tqdm(index.items(), desc='Building mapping'):
        smiles = canonicalize_smiles(data['smiles'])
        mapping[smiles] = initialize_mapping(mapping, smiles)
        mapping[smiles]['idx'] = idx
    spectral_data = {}
    for split in ['train', 'val', 'test']:
        spectral_data[split] = {}
        with open(DATA_RAW / f'{split}.jsonl', 'r') as f:
            for line in tqdm(f, desc=f'Loading {split} data'):
                data = json.loads(line)
                spectral_data[split][data['idx']] = data
    # transfer experimental spectral data for every legacy molecule unconditionally
    for idx, data in tqdm(index.items(), desc='Transferring SPECTRE spectral data'):
        smiles = canonicalize_smiles(data['smiles'])
        split_data = spectral_data[index[idx]['split']][idx]
        hsqc = split_data.get('hsqc', None)
        if hsqc is not None and len(hsqc) > 0:
            mapping[smiles]['hsqc']['spectre'] = hsqc
        c_nmr = split_data.get('c_nmr', None)
        if c_nmr is not None and len(c_nmr) > 0:
            mapping[smiles]['c_nmr']['spectre'] = c_nmr
        h_nmr = split_data.get('h_nmr', None)
        if h_nmr is not None and len(h_nmr) > 0:
            mapping[smiles]['h_nmr']['spectre'] = h_nmr
        mass_spec = split_data.get('mass_spec', None)
        if mass_spec is not None and len(mass_spec) > 0:
            mapping[smiles]['mass_spec']['spectre'] = mass_spec

    # load mnova simulation data
    nmr_data = process_mnova_predictions()
    for smiles, nmr in tqdm(nmr_data.items(), desc='Loading Mnova predictions'):
        smiles = canonicalize_smiles(smiles)
        initialize_mapping(mapping, smiles)
        mapping[smiles]['h_nmr']['mnova'] = nmr['h_nmr']
        mapping[smiles]['c_nmr']['mnova'] = nmr['c_nmr']
        mapping[smiles]['hsqc']['mnova'] = nmr['hsqc']

    # load ms simulation data
    ms_data = process_ms_predictions()
    for smiles, ms in tqdm(ms_data.items(), desc='Loading MS predictions'):
        smiles = canonicalize_smiles(smiles)
        initialize_mapping(mapping, smiles)
        mapping[smiles]['mass_spec']['ms'] = ms
    return mapping

def build_json(mapping: Dict[str, Dict]) -> Dict[str, Dict]:
    print('Building index...')
    os.makedirs(DATA_CLEANED, exist_ok=True)
    index = {}
    idx = 0

    for smiles, data in tqdm(mapping.items(), desc='Building index'):
        data_entry = {
            'hsqc': priority_access(data['hsqc'], SOURCE_PRIORITY['hsqc']),
            'c_nmr': priority_access(data['c_nmr'], SOURCE_PRIORITY['c_nmr']),
            'h_nmr': priority_access(data['h_nmr'], SOURCE_PRIORITY['h_nmr']),
            'mass_spec': priority_access(data['mass_spec'], SOURCE_PRIORITY['mass_spec'])
        }
        mol = Chem.MolFromSmiles(smiles)
        mw = rdMolDescriptors.CalcExactMolWt(mol)
        index_entry = {
            'idx': idx,
            'smiles': smiles,
            'split': None,
            'has_hsqc': data_entry['hsqc'] != [],
            'has_c_nmr': data_entry['c_nmr'] != [],
            'has_h_nmr': data_entry['h_nmr'] != [],
            'has_mass_spec': data_entry['mass_spec'] != [],
            'has_mw': True,
            'has_formula': True,
            'mw': mw,
            'formula': rdMolDescriptors.CalcMolFormula(mol),
        }
        if mw > MW_MAX_EXACT or mol.GetNumHeavyAtoms() < MIN_HEAVY_ATOMS:
            continue
        index[idx] = index_entry
        idx += 1
    pickle.dump(index, open(INDEX_PKL, 'wb'))
    print('Done!')

if __name__ == "__main__":
    if os.path.exists(DATA_CLEANED / 'mapping.json'):
        mapping = json.load(open(DATA_CLEANED / 'mapping.json', 'r'))
    else:
        mapping = build_spectral_data()
        with open(DATA_CLEANED / 'mapping.json', 'w') as f:
            json.dump(mapping, f)
    build_json(mapping)
