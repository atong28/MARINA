"""How much experimental MS/MS can replace MARINA1's ICEBERG-simulated MS/MS?

MARINA1's mass_spec is ICEBERG-simulated (positive, 20 eV) for essentially every
molecule. MassSpecGym is experimental (GNPS + MassBank + MSnLib), already local
from the domain-compare run. This measures the drop-in replaceable fraction, and
breaks it out by ionization mode / adduct / collision energy / MARINA1 split.

Writes results/msms_overlap.json and results/msms_matched_molecules.parquet.
"""
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
MSG_TSV = HERE.parent / "domain-compare/raw/massspecgym/MassSpecGym.tsv"
MARINA1_INDEX = os.path.join(DATA_ROOT, "Datasets/MARINA1/index.pkl")


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    s = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


msg = pd.read_csv(MSG_TSV, sep="\t")
print(f"MassSpecGym spectra: {len(msg):,}")

# canonicalize once per unique source SMILES
uniq = msg["smiles"].dropna().unique()
print(f"unique source SMILES: {len(uniq):,}")
cmap = {}
for s in tqdm(uniq, desc="canonicalize"):
    cmap[s] = canonicalize(s)
msg["canon"] = msg["smiles"].map(cmap)
msg = msg[msg["canon"].notna()].copy()
print(f"spectra with parseable SMILES: {len(msg):,}  "
      f"unique canonical: {msg['canon'].nunique():,}")

# ionization mode from adduct charge suffix
msg["mode"] = np.where(msg["adduct"].astype(str).str.rstrip().str.endswith("-"),
                       "negative", "positive")

m1 = pickle.load(open(MARINA1_INDEX, "rb"))
m1_by_smiles = {}
for idx, e in m1.items():
    m1_by_smiles.setdefault(e["smiles"], (idx, e["split"], e["has_mass_spec"]))
print(f"MARINA1 molecules: {len(m1):,}  unique SMILES: {len(m1_by_smiles):,}")

msg["in_marina1"] = msg["canon"].isin(m1_by_smiles.keys())
matched = msg[msg["in_marina1"]].copy()
matched["m1_split"] = matched["canon"].map(lambda s: m1_by_smiles[s][1])

print(f"\nspectra matching a MARINA1 molecule: {len(matched):,} "
      f"({100*len(matched)/len(msg):.1f}% of MassSpecGym)")
print(f"distinct MARINA1 molecules covered: {matched['canon'].nunique():,} "
      f"({100*matched['canon'].nunique()/len(m1_by_smiles):.2f}% of MARINA1)")


def breakdown(df, col, top=None):
    c = df[col].astype(str).value_counts()
    return {k: int(v) for k, v in (c.head(top) if top else c).items()}


# per-molecule spectrum counts -- how many experimental spectra each molecule gets
per_mol = matched.groupby("canon").size()
per_mol_pos = matched[matched["mode"] == "positive"].groupby("canon").size()

summary = {
    "massspecgym": {
        "spectra_total": int(len(msg)),
        "unique_canonical_smiles": int(msg["canon"].nunique()),
        "by_mode": breakdown(msg, "mode"),
        "by_adduct_top10": breakdown(msg, "adduct", 10),
        "by_instrument": breakdown(msg, "instrument_type"),
        "collision_energy": {
            "non_null": int(msg["collision_energy"].notna().sum()),
            "min": float(msg["collision_energy"].min()),
            "median": float(msg["collision_energy"].median()),
            "max": float(msg["collision_energy"].max()),
        },
    },
    "overlap_with_marina1": {
        "marina1_unique_smiles": int(len(m1_by_smiles)),
        "matched_spectra": int(len(matched)),
        "matched_molecules": int(matched["canon"].nunique()),
        "pct_of_marina1": round(
            100 * matched["canon"].nunique() / len(m1_by_smiles), 3),
        "pct_of_massspecgym_spectra": round(100 * len(matched) / len(msg), 2),
        "matched_by_mode": breakdown(matched, "mode"),
        "matched_molecules_by_mode": {
            m: int(g["canon"].nunique())
            for m, g in matched.groupby("mode")
        },
        "matched_molecules_by_marina1_split": breakdown(
            matched.drop_duplicates("canon"), "m1_split"),
        "matched_by_adduct_top10": breakdown(matched, "adduct", 10),
        "spectra_per_molecule": {
            "mean": round(float(per_mol.mean()), 2),
            "median": float(per_mol.median()),
            "p90": float(per_mol.quantile(0.9)),
            "max": int(per_mol.max()),
        },
        "spectra_per_molecule_positive_only": {
            "molecules": int(len(per_mol_pos)),
            "mean": round(float(per_mol_pos.mean()), 2),
            "median": float(per_mol_pos.median()),
            "max": int(per_mol_pos.max()),
        },
    },
}

with open(RESULTS / "msms_overlap.json", "w") as f:
    json.dump(summary, f, indent=2)

out = (matched.drop_duplicates("canon")[["canon", "m1_split"]]
       .rename(columns={"canon": "smiles"}))
out["n_spectra"] = out["smiles"].map(per_mol)
out["n_spectra_positive"] = out["smiles"].map(per_mol_pos).fillna(0).astype(int)
out.to_parquet(RESULTS / "msms_matched_molecules.parquet", index=False)

print("\n" + json.dumps(summary, indent=2))
