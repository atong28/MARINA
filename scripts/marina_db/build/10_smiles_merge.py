"""Merge npmrd.csv / coconut.csv / lotus.txt into smiles_dict.json.

Per-source provenance is kept alongside a canonical (stereo-preserving) 3D SMILES;
the dict is keyed by the 2D canonical SMILES.
"""
import csv
import os
from typing import Dict

from tqdm import tqdm
import json

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # scripts/marina_db -> config
sys.path.insert(0, str(_HERE.parents[3]))   # repo root -> src
from config import DATA_RAW, DATA_CLEANED, SMILES_DICT
from src.modules.data.smiles import canonicalize_smiles

csv.field_size_limit(1000000)

INPUT_DIR = DATA_RAW
OUTPUT_DIR = DATA_CLEANED

def process_npmrd(input_dir: Path, smiles_dict: Dict[str, Dict]) -> Dict[str, Dict]:
    count = 0
    with open(os.path.join(input_dir, "npmrd.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            smiles = canonicalize_smiles(row["SMILES"], keep_stereo=False)
            smiles_3d = canonicalize_smiles(row["SMILES"], keep_stereo=True)
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

def process_coconut(input_dir: Path, smiles_dict: Dict[str, Dict]) -> Dict[str, Dict]:
    count = 0
    with open(os.path.join(input_dir, "coconut.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            smiles = canonicalize_smiles(row["canonical_smiles"], keep_stereo=False)
            smiles_3d = canonicalize_smiles(row["canonical_smiles"], keep_stereo=True)
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

def process_lotus(input_dir: Path, smiles_dict: Dict[str, Dict]) -> Dict[str, Dict]:
    count = 0
    with open(os.path.join(input_dir, "lotus.txt"), "r") as f:
        for line in tqdm(f):
            raw_smiles, lotusid = line.split()
            smiles = canonicalize_smiles(raw_smiles, keep_stereo=False)
            smiles_3d = canonicalize_smiles(raw_smiles, keep_stereo=True)
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

    with open(SMILES_DICT, "w") as f:
        json.dump(smiles_dict, f)
