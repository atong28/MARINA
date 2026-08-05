"""Recover HSQC provenance for MARINA1.

generate_dataset.py resolves `priority_access(hsqc, ['spectre','mnova'])` at build
time and discards which source won, so MARINA1 itself cannot tell experimental
HSQC from simulated. This script recovers the label.

Signal: in the SPECTRE (MoonshotDatasetv3) HSQC store the third column is either
  * exactly +/-1        -> multiplicity known only as CH/CH3 vs CH2, or
  * an arbitrary float  -> a signed predicted cross-peak volume.

discriminate_hsqc.py settles which is which: the +/-1 population carries shifts
rounded to 1-2 dp (literature transcription) while the float population is 95-99%
full 7-dp precision with unphysical C shifts down to -22.8 ppm. So

  sign_only -> EXPERIMENTAL, derived from JEOL/CH-NMR-NP. That database stores
               13C/1H shifts plus attached-H count per carbon and no 2D spectra,
               so reconstructing HSQC from it yields exactly +/-1 multiplicity
               and rounded shifts -- precisely what is observed.
  intensity -> ACD/Labs Spectrus simulation (the paper's "simulated additional
               HSQC"), whose predicted volume lands in the third column.

Writes results/hsqc_provenance.json + results/marina1_hsqc_provenance.parquet
(one row per MARINA1 molecule with an HSQC).
"""
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
SPECTRE_INDEX = HERE / "raw/spectre/index.pkl"
SPECTRE_LMDB = str(HERE / "raw/spectre/_lmdb/{split}/HSQC_NMR.lmdb")
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
MARINA1 = os.path.join(DATA_ROOT, "Datasets/MARINA1")


def parse_tensor(buf):
    h1 = buf.index(b"|")
    h2 = buf.index(b"|", h1 + 1)
    h3 = buf.index(b"|", h2 + 1)
    dtype = np.dtype(buf[:h1].decode())
    shape = tuple(int(x) for x in buf[h2 + 1:h3].split(b","))
    return np.frombuffer(buf[h3 + 1:], dtype=dtype).reshape(shape)


def canonicalize(smiles):
    """MARINA's exact rule: double-pass, stereochemistry stripped."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    s = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


EXPERIMENTAL = "experimental_jeol"
ACD = "acd_simulated"
MNOVA = "mnova_simulated"


def classify(arr):
    """+/-1 third column -> JEOL-derived experimental; else ACD simulation."""
    if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
        return "degenerate"
    third = arr[:, 2]
    return EXPERIMENTAL if np.all(np.isin(third, (-1.0, 1.0))) else ACD


# ---- 1. classify every SPECTRE HSQC entry -------------------------------------
sp_index = pickle.load(open(SPECTRE_INDEX, "rb"))
sp_class, sp_npeaks = {}, {}
for split in ("train", "val", "test"):
    env = lmdb.open(SPECTRE_LMDB.format(split=split), readonly=True, lock=False)
    with env.begin() as txn:
        for k, v in tqdm(txn.cursor(), desc=f"spectre/{split}"):
            idx = int(k.decode())
            arr = parse_tensor(v)
            sp_class[idx] = classify(arr)
            sp_npeaks[idx] = int(arr.shape[0])
    env.close()

print("\nSPECTRE HSQC entries by third-column type:")
for kind, n in Counter(sp_class.values()).most_common():
    print(f"  {kind:12s} {n:>8,}")

# ---- 2. map those onto canonical SMILES --------------------------------------
sp_by_smiles = {}
for idx, kind in tqdm(sp_class.items(), desc="canonicalize spectre"):
    entry = sp_index.get(idx)
    if entry is None:
        continue
    smi = canonicalize(entry["smiles"])
    if smi is None:
        continue
    # a SMILES seen twice: experimental wins over simulation
    if sp_by_smiles.get(smi) != EXPERIMENTAL:
        sp_by_smiles[smi] = kind

print(f"\nunique canonical SMILES with SPECTRE HSQC: {len(sp_by_smiles):,}")
for kind, n in Counter(sp_by_smiles.values()).most_common():
    print(f"  {kind:12s} {n:>8,}")

# ---- 3. join onto MARINA1 -----------------------------------------------------
m1 = pickle.load(open(f"{MARINA1}/index.pkl", "rb"))
rows = []
for idx, e in tqdm(m1.items(), desc="marina1"):
    if not e.get("has_hsqc"):
        continue
    smi = e["smiles"]  # already canonical under the same rule
    rows.append({
        "idx": idx,
        "smiles": smi,
        "split": e["split"],
        "mw": e["mw"],
        "provenance": sp_by_smiles.get(smi, MNOVA),
    })
df = pd.DataFrame(rows)

# ---- 4. confirm the label survives into MARINA1's own stored HSQC -------------
# read MARINA1's arrow HSQC for a sample of each class and re-classify
sample_check = {}
for split in ("train", "val", "test"):
    path = f"{MARINA1}/arrow/{split}/HSQC_NMR.parquet"
    tbl = pq.read_table(path)
    idx_col = tbl.column("idx").to_pylist()
    data_col = tbl.column("data")
    shape_col = tbl.column("shape")
    pos = {v: i for i, v in enumerate(idx_col)}
    sub = df[df.split == split]
    for prov in (ACD, EXPERIMENTAL, MNOVA):
        cand = sub[sub.provenance == prov]["idx"].tolist()[:400]
        kinds = Counter()
        for i in cand:
            if i not in pos:
                continue
            p = pos[i]
            arr = np.asarray(data_col[p].as_py(), dtype=np.float64)
            shp = tuple(shape_col[p].as_py())
            kinds[classify(arr.reshape(shp))] += 1
        sample_check[f"{split}/{prov}"] = dict(kinds)

print("\nMARINA1 stored-HSQC re-classification (sample of 400 per cell):")
for k, v in sample_check.items():
    print(f"  {k:20s} {v}")

# ---- 5. report ----------------------------------------------------------------
summary = {
    "spectre_hsqc_entries": dict(Counter(sp_class.values())),
    "spectre_unique_smiles": dict(Counter(sp_by_smiles.values())),
    "marina1_hsqc_molecules": int(len(df)),
    "marina1_by_provenance": {k: int(v) for k, v in
                              df.provenance.value_counts().items()},
    "marina1_by_provenance_split": {
        prov: {s: int(n) for s, n in g.split.value_counts().items()}
        for prov, g in df.groupby("provenance")
    },
    "marina1_stored_hsqc_recheck": sample_check,
}
with open(RESULTS / "hsqc_provenance.json", "w") as f:
    json.dump(summary, f, indent=2)
df.to_parquet(RESULTS / "marina1_hsqc_provenance.parquet", index=False)

print("\n" + json.dumps(summary, indent=2))
