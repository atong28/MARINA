import csv
import os
from typing import Dict
from rdkit import Chem
from tqdm import tqdm
import json

csv.field_size_limit(1000000)

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/cleaned"

def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)

def canonical_3d_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

def process_npmrd(input_dir: str, smiles_dict: Dict[str, Dict]) -> Dict[str, Dict]:
    count = 0
    with open(os.path.join(input_dir, "npmrd.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            smiles = canonicalize_smiles(row["SMILES"])
            smiles_3d = canonical_3d_smiles(row["SMILES"])
            if smiles is not None:
                db_entry = {"npid": row["NP_MRD_ID"], "name": row["Natural_Products_Name"], "original_smiles": smiles_3d}
                if smiles not in smiles_dict:
                    smiles_dict[smiles] = {
                        "smiles": smiles,
                        "canonical_3d_smiles": smiles_3d,
                        "npmrd": db_entry,
                        "coconut": None,
                        "lotus": None,
                    }
                    count += 1
                else:
                    smiles_dict[smiles]["npmrd"] = db_entry
                    if smiles_dict[smiles].get("canonical_3d_smiles") is None:
                        smiles_dict[smiles]["canonical_3d_smiles"] = smiles_3d
    print('---------------------------------------------------------------')
    print(f'Added {count} new smiles to the dictionary from NP-MRD')
    print('---------------------------------------------------------------')
    return smiles_dict

def process_coconut(input_dir: str, smiles_dict: Dict[str, Dict]) -> Dict[str, Dict]:
    count = 0
    with open(os.path.join(input_dir, "coconut.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            smiles = canonicalize_smiles(row["canonical_smiles"])
            smiles_3d = canonical_3d_smiles(row["canonical_smiles"])
            if smiles is not None:
                db_entry = {"coconut_id": row["identifier"], "name": row["name"], "original_smiles": smiles_3d}
                if smiles not in smiles_dict:
                    smiles_dict[smiles] = {
                        "smiles": smiles,
                        "canonical_3d_smiles": smiles_3d,
                        "coconut": db_entry,
                        "npmrd": None,
                        "lotus": None,
                    }
                    count += 1
                else:
                    smiles_dict[smiles]["coconut"] = db_entry
                    if smiles_dict[smiles].get("canonical_3d_smiles") is None:
                        smiles_dict[smiles]["canonical_3d_smiles"] = smiles_3d
    print('---------------------------------------------------------------')
    print(f'Added {count} new smiles to the dictionary from COCONUT')
    print('---------------------------------------------------------------')
    return smiles_dict

def process_lotus(input_dir: str, smiles_dict: Dict[str, Dict]) -> Dict[str, Dict]:
    count = 0
    with open(os.path.join(input_dir, "lotus.txt"), "r") as f:
        for line in tqdm(f):
            raw_smiles, lotusid = line.split()
            smiles = canonicalize_smiles(raw_smiles)
            smiles_3d = canonical_3d_smiles(raw_smiles)
            if smiles is not None:
                db_entry = {"lotus_id": lotusid, "original_smiles": smiles_3d}
                if smiles not in smiles_dict:
                    smiles_dict[smiles] = {
                        "smiles": smiles,
                        "canonical_3d_smiles": smiles_3d,
                        "lotus": db_entry,
                        "npmrd": None,
                        "coconut": None,
                    }
                    count += 1
                else:
                    smiles_dict[smiles]["lotus"] = db_entry
                    if smiles_dict[smiles].get("canonical_3d_smiles") is None:
                        smiles_dict[smiles]["canonical_3d_smiles"] = smiles_3d
    print('---------------------------------------------------------------')
    print(f'Added {count} new smiles to the dictionary from LOTUS')
    print('---------------------------------------------------------------')
    return smiles_dict

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    smiles_dict = {}
    process_npmrd(INPUT_DIR, smiles_dict)
    process_coconut(INPUT_DIR, smiles_dict)
    process_lotus(INPUT_DIR, smiles_dict)
    
    with open(os.path.join(OUTPUT_DIR, "smiles_dict.json"), "w") as f:
        json.dump(smiles_dict, f)
