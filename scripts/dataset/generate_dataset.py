import json
from typing import Any, Dict, List, Tuple
from src.modules.data.fp_loader import EntropyFPLoader
import pickle
import os
from rdkit import Chem
import glob
from tqdm import tqdm
import random
from rdkit.Chem import rdMolDescriptors

random.seed(0)

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

def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)

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
    files = sorted(glob.glob('data/raw/mnova_predictions/*.jsonl'))
    nmr_data = {}
    for file in tqdm(files, desc='Processing Mnova predictions'):
        nmrs = process_file(file)
        for smiles, nmr in nmrs.items():
            nmr_data[smiles] = assemble_nmr_data(nmr)
    return nmr_data

def process_ms_predictions():
    files = sorted(glob.glob('data/raw/ms_predictions/*.json'))
    ms_data = {}
    for file in tqdm(files, desc='Processing MS predictions'):
        ms = json.load(open(file, 'r'))
        for data in ms:
            ms_data[canonicalize_smiles(data['SMILES'])] = data['peaks']
    return ms_data

def build_spectral_data() -> Dict[str, Dict]:
    print('Building spectral data...')
    os.makedirs('data/cleaned', exist_ok=True)
    # load old moonshotdatasetv3, map to key spectre
    index = pickle.load(open('data/raw/index.pkl', 'rb'))
    mapping = {}
    for idx, data in tqdm(index.items(), desc='Building mapping'):
        smiles = canonicalize_smiles(data['smiles'])
        mapping[smiles] = initialize_mapping(mapping, smiles)
        mapping[smiles]['idx'] = idx
    spectral_data = {}
    for split in ['train', 'val', 'test']:
        spectral_data[split] = {}
        with open(f'data/raw/{split}.jsonl', 'r') as f:
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

def build_retrieval_set(mapping: Dict[str, Dict]) -> Dict[str, Dict]:
    print('Building retrieval set...')
    os.makedirs('data/cleaned', exist_ok=True)
    with open('data/cleaned/smiles_dict.json', 'r') as f:
        smiles_dict = json.load(f)
    all_smiles = sorted(set(smiles_dict.keys()) | set(mapping.keys()))
    print(f'smiles_dict: {len(smiles_dict)}, mapping: {len(mapping)}, union: {len(all_smiles)}')
    retrieval = {idx: {'smiles': smiles} for idx, smiles in enumerate(all_smiles)}
    print(f'Built {len(retrieval)} retrieval set')
    with open('data/cleaned/retrieval.pkl', 'wb') as f:
        pickle.dump(retrieval, f)
    return retrieval

def build_json(mapping: Dict[str, Dict]) -> Dict[str, Dict]:
    print('Building index...')
    os.makedirs('data/cleaned', exist_ok=True)
    dataset = {
        'train': [],
        'val': [],
        'test': []
    }
    index = {}
    splits = ['train', 'val', 'test']
    allocations = [random.choices(splits, weights=[0.9, 0.05, 0.05])[0] for _ in range(len(mapping))]
    idx = 0
    
    # load benchmarking set
    benchmark_data = pickle.load(open('data/raw/benchmark.pkl', 'rb'))
    benchmark_smiles = {canonicalize_smiles(data['smiles']) for data in benchmark_data.values()}
    for smiles, data in tqdm(mapping.items(), desc='Building index'):
        data_entry = {
            'hsqc': priority_access(data['hsqc'], ['spectre', 'mnova']),
            'c_nmr': priority_access(data['c_nmr'], ['mnova', 'spectre']),
            'h_nmr': priority_access(data['h_nmr'], ['mnova', 'spectre']),
            'mass_spec': priority_access(data['mass_spec'], ['spectre', 'ms'])
        }
        mw = rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(smiles))
        split = allocations[idx] if smiles not in benchmark_smiles else 'test'
        index_entry = {
            'idx': idx,
            'smiles': smiles,
            'split': split,
            'has_hsqc': data_entry['hsqc'] != [],
            'has_c_nmr': data_entry['c_nmr'] != [],
            'has_h_nmr': data_entry['h_nmr'] != [],
            'has_mass_spec': data_entry['mass_spec'] != [],
            'has_mw': True,
            'has_formula': True,
            'mw': mw,
            'formula': rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles)),
        }
        if mw > 1000:
            continue
        data_entry.update(index_entry)
        index[idx] = index_entry
        dataset[split].append(data_entry)
        idx += 1
    for split in splits:
        with open(f'data/cleaned/{split}.jsonl', 'w') as f:
            f.write('\n'.join([json.dumps(entry) for entry in dataset[split]]))
    pickle.dump(index, open('data/cleaned/index.pkl', 'wb'))
    print('Done!')

if __name__ == "__main__":
    if os.path.exists('data/cleaned/mapping.json'):
        mapping = json.load(open('data/cleaned/mapping.json', 'r'))
    else:
        mapping = build_spectral_data()
        with open('data/cleaned/mapping.json', 'w') as f:
            json.dump(mapping, f)
    if not os.path.exists('data/cleaned/retrieval.pkl'):
        build_retrieval_set(mapping)
    build_json(mapping)