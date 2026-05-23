import json
import os
import pickle

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from tqdm import tqdm

INPUT_DIR = "data/cleaned"
OUTPUT_DIR = "data/cleaned"


def _inchi_inchikey(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    inchi = MolToInchi(mol)
    if inchi is None:
        return None, None
    return inchi, InchiToInchiKey(inchi)


if __name__ == "__main__":
    with open(os.path.join(INPUT_DIR, "smiles_dict.json")) as f:
        smiles_dict = json.load(f)

    with open(os.path.join(INPUT_DIR, "retrieval.pkl"), "rb") as f:
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

    out_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(metadata, f)
    print(f"Wrote {len(metadata)} entries to {out_path}")
