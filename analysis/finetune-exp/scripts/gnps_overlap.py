"""GNPS full library overlap with MARINA1 -- the largest NP-focused MS/MS source.

MassSpecGym is a deduplicated benchmark subset (positive only) and MassBank is
small. GNPS is the primary natural-products MS/MS repository, so it should show
the highest overlap with MARINA1's NP chemical space. Streams the 4.5 GB MGF and
keeps only header fields.

LIBRARYQUALITY: 1 = gold, 2 = silver, 3 = bronze//challenge. Reported separately
because quality gating matters for a finetuning set.

Writes results/gnps_overlap.json + results/gnps_matched_molecules.parquet.
"""
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
MGF = HERE / "raw/gnps/ALL_GNPS.mgf"
MARINA1_INDEX = os.path.join(DATA_ROOT, "Datasets/MARINA1/index.pkl")
WANT = {"SMILES", "IONMODE", "MSLEVEL", "LIBRARYQUALITY", "SPECTRUMID",
        "SOURCE_INSTRUMENT", "CHARGE"}


def stream_headers(path):
    rec, npeaks = {}, 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("BEGIN IONS"):
                rec, npeaks = {}, 0
            elif line.startswith("END IONS"):
                rec["n_peaks"] = npeaks
                yield rec
                rec, npeaks = {}, 0
            elif "=" in line and line[0].isalpha():
                k, _, v = line.partition("=")
                if k in WANT:
                    rec[k] = v.strip()
            elif line[0].isdigit():
                npeaks += 1


records = []
for rec in tqdm(stream_headers(MGF), desc="parse mgf", unit=" spec"):
    records.append(rec)

g = pd.DataFrame(records)
print(f"\ntotal spectra: {len(g):,}")

g["MSLEVEL"] = pd.to_numeric(g.get("MSLEVEL"), errors="coerce")
g = g[g["MSLEVEL"] >= 2].copy()
print(f"MS2+: {len(g):,}")

g["SMILES"] = g["SMILES"].astype(str).str.strip()
g = g[~g["SMILES"].isin(["", "N/A", "n/a", "nan", "NA", "0"])].copy()
print(f"with a SMILES string: {len(g):,}")

uniq = g["SMILES"].unique()
print(f"unique raw SMILES: {len(uniq):,}")


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
g["canon"] = g["SMILES"].map(cmap)
g = g[g["canon"].notna()].copy()
print(f"parseable: {len(g):,} spectra / {g['canon'].nunique():,} molecules")

g["mode"] = (g["IONMODE"].astype(str).str.strip().str.lower()
             .map({"positive": "positive", "negative": "negative"})
             .fillna("unknown"))

m1 = pickle.load(open(MARINA1_INDEX, "rb"))
m1_by_smiles = {e["smiles"]: e["split"] for e in m1.values()}

g["in_marina1"] = g["canon"].isin(m1_by_smiles.keys())
matched = g[g["in_marina1"]].copy()
matched["m1_split"] = matched["canon"].map(m1_by_smiles)


def bd(df, col, top=None):
    c = df[col].astype(str).value_counts()
    return {k: int(v) for k, v in (c.head(top) if top else c).items()}


def mode_split(df):
    return {m: {"spectra": int(len(x)), "molecules": int(x["canon"].nunique())}
            for m, x in df.groupby("mode")}


pos = set(matched[matched["mode"] == "positive"]["canon"])
neg = set(matched[matched["mode"] == "negative"]["canon"])
gold = matched[matched["LIBRARYQUALITY"].astype(str) == "1"]

summary = {
    "gnps_library": {
        "ms2plus_spectra": int(len(g)),
        "unique_canonical_smiles": int(g["canon"].nunique()),
        "by_mode": mode_split(g),
        "by_library_quality": bd(g, "LIBRARYQUALITY"),
        "by_instrument_top10": bd(g, "SOURCE_INSTRUMENT", 10),
    },
    "overlap_with_marina1": {
        "marina1_unique_smiles": int(len(m1_by_smiles)),
        "matched_spectra": int(len(matched)),
        "matched_molecules": int(matched["canon"].nunique()),
        "pct_of_marina1": round(
            100 * matched["canon"].nunique() / len(m1_by_smiles), 3),
        "by_mode": mode_split(matched),
        "by_marina1_split": bd(matched.drop_duplicates("canon"), "m1_split"),
        "by_library_quality": bd(matched, "LIBRARYQUALITY"),
        "gold_only_molecules": int(gold["canon"].nunique()),
        "positive_only_molecules": int(len(pos - neg)),
        "negative_only_molecules": int(len(neg - pos)),
        "both_modes_molecules": int(len(pos & neg)),
        "spectra_per_molecule_median": float(
            matched.groupby("canon").size().median()),
    },
}

with open(RESULTS / "gnps_overlap.json", "w") as f:
    json.dump(summary, f, indent=2)

out = matched.drop_duplicates("canon")[["canon", "m1_split"]].rename(
    columns={"canon": "smiles"})
out["n_spectra"] = out["smiles"].map(matched.groupby("canon").size())
out["has_positive"] = out["smiles"].isin(pos)
out["has_negative"] = out["smiles"].isin(neg)
out["has_gold"] = out["smiles"].isin(set(gold["canon"]))
out.to_parquet(RESULTS / "gnps_matched_molecules.parquet", index=False)

print("\n" + json.dumps(summary, indent=2))
