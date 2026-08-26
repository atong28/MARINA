"""Write metadata.json, then convert the cleaned jsonl dataset into an Arrow dir.

Combines generate_metadata.py (metadata) with convert.py's --to-arrow path, run in
sequence. Fingerprint / rankingset building is a separate stage (60); the regular-file
copy here only copies those artifacts if they already exist.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from tqdm import tqdm

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # scripts/marina_db -> config
sys.path.insert(0, str(_HERE.parents[3]))   # repo root -> src
from config import (
    DATA_CLEANED, DATA_DATASET, METADATA_JSON, SMILES_DICT, RETRIEVAL_PKL, INDEX_PKL,
    SOURCE_PRIORITY, FP_RADIUS, FP_TYPE,
)

SPLITS = ("train", "val", "test")

# Source-priority order shared with the index build (stage 6) via config.SOURCE_PRIORITY.
# FragIdx is written by stage 9 (fp_fragidx), not here — the materialized JSONL
# never carries a "fragidx" field, so this stage only writes the spectral modalities.
MOD_DIRS = {
    "hsqc": "HSQC_NMR",
    "h_nmr": "H_NMR",
    "c_nmr": "C_NMR",
    "mass_spec": "MassSpec",
    "mass_spec_neg": "MassSpecNeg",
}

REGULAR_FILES = [
    "index.pkl",
    "retrieval.pkl",
    "metadata.json",
    f"count_hashes_under_radius_{FP_RADIUS}.pkl",
]
# Stage 4 (fp_rankingset) writes the FP-family dirs straight into DATA_DATASET, so this
# copy is a no-op unless a live family exists beside the cleaned data (guarded below).
REGULAR_DIRS = [FP_TYPE]


# ---- metadata -------------------------------------------------------------

def _inchi_inchikey(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    inchi = MolToInchi(mol)
    if inchi is None:
        return None, None
    return inchi, InchiToInchiKey(inchi)


def build_metadata() -> None:
    with open(SMILES_DICT) as f:
        smiles_dict = json.load(f)

    with open(RETRIEVAL_PKL, "rb") as f:
        retrieval = pickle.load(f)

    metadata = {}
    for idx, entry in tqdm(retrieval.items(), desc="Building metadata"):
        smiles = entry["smiles"]
        sd = smiles_dict.get(smiles, {})

        canonical_3d = sd.get("canonical_3d_smiles") or smiles
        meta = {
            "smiles": smiles,
            "canonical_2d_smiles": smiles,
            "canonical_3d_smiles": canonical_3d,
        }

        for db in ("npmrd", "coconut", "lotus"):
            db_data = sd.get(db)
            if db_data is None:
                meta[db] = None
                continue
            db_smiles = db_data.get("original_smiles") or smiles
            inchi, inchikey = _inchi_inchikey(db_smiles)
            db_entry = {k: v for k, v in db_data.items() if k != "original_smiles"}
            db_entry["inchi"] = inchi
            db_entry["inchikey"] = inchikey
            meta[db] = db_entry

        metadata[str(idx)] = meta

    with open(METADATA_JSON, "w") as f:
        json.dump(metadata, f)
    print(f"Wrote {len(metadata)} entries to {METADATA_JSON}")


# ---- materialize per-split jsonl ------------------------------------------

def _priority_access(mod_dict: Dict[str, Any], order: List[str]) -> Any:
    for key in order:
        val = mod_dict.get(key)
        if val:
            return val
    return []


def materialize_split_jsonl() -> None:
    """Join index.pkl (idx, smiles, split — assigned by 7_splits.py) with the spectral
    data cached in mapping.json (keyed by smiles) into per-split {split}.jsonl that
    convert_to_arrow consumes. Splits deferred out of stage 6 land here."""
    with open(INDEX_PKL, "rb") as f:
        index = pickle.load(f)
    with open(DATA_CLEANED / "mapping.json") as f:
        mapping = json.load(f)

    handles = {s: open(DATA_CLEANED / f"{s}.jsonl", "w", encoding="utf-8") for s in SPLITS}
    n = {s: 0 for s in SPLITS}
    try:
        for entry in tqdm(index.values(), desc="Materializing jsonl"):
            split = entry.get("split")
            if split not in handles:
                raise ValueError(f"idx {entry.get('idx')} has split={split!r}; run 7_splits.py first")
            m = mapping.get(entry["smiles"], {})
            row = {"idx": entry["idx"], "smiles": entry["smiles"]}
            for mod, order in SOURCE_PRIORITY.items():
                row[mod] = _priority_access(m.get(mod, {}), order)
            handles[split].write(json.dumps(row) + "\n")
            n[split] += 1
    finally:
        for h in handles.values():
            h.close()
    print(f"Materialized jsonl: " + ", ".join(f"{s}={n[s]}" for s in SPLITS))


# ---- arrow conversion -----------------------------------------------------

def _copy_regular_data(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for filename in REGULAR_FILES:
        src = os.path.join(input_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, filename))
    for dirname in REGULAR_DIRS:
        src = os.path.join(input_dir, dirname)
        if os.path.isdir(src):
            dst = os.path.join(output_dir, dirname)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def _flatten_with_shape(value: Any) -> tuple[list[float], list[int]]:
    if isinstance(value, list) and value and isinstance(value[0], list):
        shape = [len(value), len(value[0])]
        flat = [float(x) for row in value for x in row]
        return flat, shape
    if isinstance(value, list):
        shape = [len(value)]
        flat = [float(x) for x in value]
        return flat, shape
    return [], []


def convert_to_arrow(input_dir: str, output_dir: str) -> None:
    print(f"Converting to Arrow format: {input_dir} -> {output_dir}")
    _copy_regular_data(input_dir, output_dir)

    arrow_base = os.path.join(output_dir, "arrow")
    os.makedirs(arrow_base, exist_ok=True)

    for split in SPLITS:
        jsonl_path = os.path.join(input_dir, f"{split}.jsonl")
        if not os.path.isfile(jsonl_path):
            continue

        split_rows: dict[str, list[tuple[int, list[float], list[int]]]] = {
            "hsqc": [],
            "h_nmr": [],
            "c_nmr": [],
            "mass_spec": [],
            "mass_spec_neg": [],
        }

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"{split}"):
                if not line.strip():
                    continue
                row = json.loads(line)
                idx = int(row["idx"])
                for mod_key in split_rows:
                    if mod_key not in row or row[mod_key] in (None, []):
                        continue
                    flat, shape = _flatten_with_shape(row[mod_key])
                    split_rows[mod_key].append((idx, flat, shape))

        split_dir = os.path.join(arrow_base, split)
        os.makedirs(split_dir, exist_ok=True)
        for mod_key, mod_dir in MOD_DIRS.items():
            rows = split_rows.get(mod_key, [])
            if not rows:
                continue
            table = {
                "idx": [r[0] for r in rows],
                "data": [r[1] for r in rows],
                "shape": [r[2] for r in rows],
            }
            pq.write_table(pa.table(table), os.path.join(split_dir, f"{mod_dir}.parquet"))


if __name__ == "__main__":
    build_metadata()
    materialize_split_jsonl()
    convert_to_arrow(str(DATA_CLEANED), str(DATA_DATASET))
