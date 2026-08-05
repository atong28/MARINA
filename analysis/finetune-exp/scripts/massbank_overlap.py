"""MassBank 2026.03 overlap with MARINA1 -- the negative-ionization source.

MassSpecGym is 100% positive mode, so it cannot answer the negative-mode
question at all. MassBank is CC-BY(-SA) licensed, ships both modes, and is small
enough to parse whole. Same join as msms_overlap.py so the numbers are
comparable.

Writes results/massbank_overlap.json + results/massbank_matched_molecules.parquet.
"""
import json
import os
import pickle
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
MSP = HERE / "raw/massbank/MassBank_NISTformat.msp"
MARINA1_INDEX = os.path.join(DATA_ROOT, "Datasets/MARINA1/index.pkl")

FIELDS = ("SMILES", "Precursor_type", "Ion_mode", "Collision_energy",
          "Instrument_type", "Spectrum_type", "DB#", "Num Peaks")


def parse_msp(path):
    """Yield one dict per record. Records are separated by blank lines."""
    rec = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if rec:
                    yield rec
                    rec = {}
                continue
            if ":" in line and not line[0].isdigit():
                key, _, val = line.partition(":")
                key = key.strip()
                if key in FIELDS:
                    rec[key] = val.strip()
    if rec:
        yield rec


records = []
for rec in tqdm(parse_msp(MSP), desc="parse msp"):
    if rec.get("Spectrum_type") not in ("MS2", "MS3", "MS4"):
        continue  # MS1 records carry no fragmentation
    records.append({
        "db": rec.get("DB#"),
        "smiles": rec.get("SMILES"),
        "adduct": rec.get("Precursor_type"),
        "mode": (rec.get("Ion_mode") or "").upper(),
        "ce": rec.get("Collision_energy"),
        "instrument": rec.get("Instrument_type"),
        "n_peaks": rec.get("Num Peaks"),
    })

mb = pd.DataFrame(records)
print(f"\nMS2+ records: {len(mb):,}")
mb = mb[mb["smiles"].notna() & (mb["smiles"].str.strip() != "")].copy()
mb["n_peaks"] = pd.to_numeric(mb["n_peaks"], errors="coerce")
print(f"with SMILES:  {len(mb):,}")

uniq = mb["smiles"].unique()
print(f"unique source SMILES: {len(uniq):,}")


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    s = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


cmap = {s: canonicalize(s) for s in tqdm(uniq, desc="canonicalize")}
mb["canon"] = mb["smiles"].map(cmap)
mb = mb[mb["canon"].notna()].copy()
print(f"parseable: {len(mb):,} spectra / {mb['canon'].nunique():,} molecules")

m1 = pickle.load(open(MARINA1_INDEX, "rb"))
m1_by_smiles = {e["smiles"]: e["split"] for e in m1.values()}

mb["in_marina1"] = mb["canon"].isin(m1_by_smiles.keys())
matched = mb[mb["in_marina1"]].copy()
matched["m1_split"] = matched["canon"].map(m1_by_smiles)


def bd(df, col, top=None):
    c = df[col].astype(str).value_counts()
    return {k: int(v) for k, v in (c.head(top) if top else c).items()}


def mode_split(df):
    return {m: {"spectra": int(len(g)), "molecules": int(g["canon"].nunique())}
            for m, g in df.groupby("mode")}


summary = {
    "massbank_release": "2026.03",
    "license": "CC-BY / CC-BY-SA 4.0 (per record)",
    "all_records": {
        "ms2plus_spectra": int(len(mb)),
        "unique_canonical_smiles": int(mb["canon"].nunique()),
        "by_mode": mode_split(mb),
        "by_adduct_top10": bd(mb, "adduct", 10),
        "by_instrument_top10": bd(mb, "instrument", 10),
        "median_peaks_per_spectrum": float(mb["n_peaks"].median()),
    },
    "overlap_with_marina1": {
        "matched_spectra": int(len(matched)),
        "matched_molecules": int(matched["canon"].nunique()),
        "pct_of_marina1": round(
            100 * matched["canon"].nunique() / len(m1_by_smiles), 3),
        "by_mode": mode_split(matched),
        "by_marina1_split": bd(matched.drop_duplicates("canon"), "m1_split"),
        "by_adduct_top10": bd(matched, "adduct", 10),
    },
}

# molecules reachable ONLY in negative mode -- the unique contribution
neg = set(matched[matched["mode"] == "NEGATIVE"]["canon"])
pos = set(matched[matched["mode"] == "POSITIVE"]["canon"])
summary["overlap_with_marina1"]["negative_only_molecules"] = int(len(neg - pos))
summary["overlap_with_marina1"]["positive_only_molecules"] = int(len(pos - neg))
summary["overlap_with_marina1"]["both_modes_molecules"] = int(len(pos & neg))

with open(RESULTS / "massbank_overlap.json", "w") as f:
    json.dump(summary, f, indent=2)

out = matched.drop_duplicates("canon")[["canon", "m1_split"]].rename(
    columns={"canon": "smiles"})
out["n_spectra"] = out["smiles"].map(matched.groupby("canon").size())
out["has_positive"] = out["smiles"].isin(pos)
out["has_negative"] = out["smiles"].isin(neg)
out.to_parquet(RESULTS / "massbank_matched_molecules.parquet", index=False)

print("\n" + json.dumps(summary, indent=2))
