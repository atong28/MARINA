"""What would a maximal experimental finetuning set for MARINA look like?

MARINA is trained with per-modality dropout, so a finetuning molecule does not need
every modality experimental -- it needs at least one. This builds the per-molecule
experimental-modality vector over all of MARINA1 and reports the distribution of
combinations, which is the thing that decides whether such a set is worth building.

Five slots, matching MARINA's spectral inputs plus the planned negative-mode channel:

    hsqc       JEOL-derived, already inside MARINA1 (provenance classification)
    c_nmr      nmrshiftdb2-2024 and/or NMRexp, restricted to MARINA1
    h_nmr      same two sources
    msms_pos   GNPS / MassBank / MassSpecGym, positive ionization
    msms_neg   GNPS / MassBank, negative ionization (new modality, not a backfill)

mw is excluded: it is computed from the formula for every molecule, so it is neither
simulated nor experimental and carries no domain gap.

Writes results/experimental_modality_matrix.parquet + .json.
"""
import json
import os
import pickle
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
DOMAIN_RAW = HERE.parent / "domain-compare/raw"
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
MARINA1 = Path(DATA_ROOT) / "Datasets/MARINA1"

SLOTS = ["hsqc", "c_nmr", "h_nmr", "msms_pos", "msms_neg"]


def canonicalize(smiles):
    """MARINA's rule: canonicalize twice, drop stereochemistry."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    once = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    mol = Chem.MolFromSmiles(once)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


def canon_map(raw_strings, label):
    """{raw -> canonical} over unique raw strings, in parallel."""
    uniq = sorted({s for s in raw_strings if isinstance(s, str) and s.strip()})
    print(f"[{label}] {len(raw_strings):,} rows -> {len(uniq):,} unique raw, canonicalizing...",
          flush=True)
    with Pool(8) as pool:
        canon = pool.map(canonicalize, uniq, chunksize=2000)
    return {r: c for r, c in zip(uniq, canon) if c}


# ---------------------------------------------------------------- MARINA1 base
index = pd.DataFrame.from_dict(pickle.load(open(MARINA1 / "index.pkl", "rb")), orient="index")
df = index[["smiles", "split"]].copy().reset_index(drop=True)
in_marina1 = set(df.smiles)
print(f"MARINA1: {len(df):,} molecules", flush=True)

flags = {slot: set() for slot in SLOTS}

# ------------------------------------------------------------- HSQC (JEOL)
hsqc_prov = pd.read_parquet(RESULTS / "marina1_hsqc_classified.parquet")
flags["hsqc"] = set(hsqc_prov[hsqc_prov.provenance == "experimental_jeol"].smiles)
print(f"  hsqc  (JEOL)          : {len(flags['hsqc']):,}", flush=True)

# ------------------------------------------------- 1D NMR: nmrshiftdb2-2024
nsdb = pickle.load(open(
    DOMAIN_RAW / "nmrshiftdb2/data/nmrshiftdb2_2024/mol_nmrshift_nmrshiftdb2_2024_all.pkl", "rb"))
nsdb_map = canon_map(nsdb["smiles"], "nmrshiftdb2")
for raw, mask in zip(nsdb["smiles"], nsdb["atom_mask"]):
    canon = nsdb_map.get(raw)
    if canon is None or canon not in in_marina1:
        continue
    nuclei = {int(x) for x in mask if int(x) != 0}
    if 6 in nuclei:
        flags["c_nmr"].add(canon)
    if 1 in nuclei:
        flags["h_nmr"].add(canon)
print(f"  after nmrshiftdb2 -> c_nmr {len(flags['c_nmr']):,}, h_nmr {len(flags['h_nmr']):,}",
      flush=True)

# -------------------------------------------------------- 1D NMR: NMRexp
nex = pq.read_table(DOMAIN_RAW / "nmrexp/NMRexp_10to24_1_1004.parquet",
                    columns=["SMILES", "NMR_type"]).to_pandas()
nex = nex[nex.NMR_type.isin(["13C NMR", "1H NMR"])]
nex_map = canon_map(nex.SMILES.tolist(), "nmrexp")
nex["canon"] = nex.SMILES.map(nex_map)
nex = nex[nex.canon.notna() & nex.canon.isin(in_marina1)]
flags["c_nmr"] |= set(nex[nex.NMR_type == "13C NMR"].canon)
flags["h_nmr"] |= set(nex[nex.NMR_type == "1H NMR"].canon)
print(f"  after NMRexp      -> c_nmr {len(flags['c_nmr']):,}, h_nmr {len(flags['h_nmr']):,}",
      flush=True)

# --------------------------------------------------------------- MS/MS modes
gnps = pd.read_parquet(RESULTS / "gnps_matched_molecules.parquet")
mbank = pd.read_parquet(RESULTS / "massbank_matched_molecules.parquet")
msgym = pd.read_parquet(RESULTS / "msms_matched_molecules.parquet")

flags["msms_pos"] = (set(gnps[gnps.has_positive].smiles)
                     | set(mbank[mbank.has_positive].smiles)
                     | set(msgym[msgym.n_spectra_positive > 0].smiles))
flags["msms_neg"] = set(gnps[gnps.has_negative].smiles) | set(mbank[mbank.has_negative].smiles)
print(f"  msms_pos              : {len(flags['msms_pos']):,}", flush=True)
print(f"  msms_neg              : {len(flags['msms_neg']):,}", flush=True)

# ------------------------------------------------------------------ assemble
for slot in SLOTS:
    df[slot] = df.smiles.isin(flags[slot])
df["n_exp"] = df[SLOTS].sum(axis=1)
df["combo"] = df[SLOTS].apply(lambda r: "+".join(s for s in SLOTS if r[s]) or "none", axis=1)

has_any = df[df.n_exp > 0]
print(f"\nmolecules with >=1 experimental modality: {len(has_any):,} "
      f"({len(has_any)/len(df):.2%} of MARINA1)", flush=True)

out = {
    "marina1_total": len(df),
    "per_slot": {s: int(df[s].sum()) for s in SLOTS},
    "any_experimental": int(len(has_any)),
    "by_n_modalities": {int(k): int(v) for k, v in df.n_exp.value_counts().sort_index().items()},
    "by_n_modalities_split": {
        str(k): v.to_dict() for k, v in has_any.groupby("n_exp").split.value_counts().unstack(fill_value=0).iterrows()
    },
    "combos": {k: int(v) for k, v in Counter(has_any.combo).most_common()},
    "split_totals": has_any.split.value_counts().to_dict(),
}
(RESULTS / "experimental_modality_matrix.json").write_text(json.dumps(out, indent=2))
df.to_parquet(RESULTS / "experimental_modality_matrix.parquet", index=False)

print("\n--- per slot ---")
for s in SLOTS:
    print(f"  {s:10s} {int(df[s].sum()):>7,}")
print("\n--- number of experimental modalities per molecule ---")
for k, v in df.n_exp.value_counts().sort_index().items():
    print(f"  {k}: {v:>7,}")
print("\n--- top combinations ---")
for k, v in Counter(has_any.combo).most_common(20):
    print(f"  {v:>6,}  {k}")
print(f"\nwrote {RESULTS/'experimental_modality_matrix.parquet'}")
