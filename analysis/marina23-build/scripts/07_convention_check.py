"""Is MARINA1's Mnova NMR per-atom or per-peak? And which source therefore aligns?

The substitution in MARINA2/MARINA3 only preserves the dataset's peak-count statistics
if the experimental source uses the same convention as the Mnova data it replaces.
nmrshiftdb2 is per-atom (a shift repeated once per symmetry-equivalent nucleus);
NMRexp is per-peak (one entry per resolved signal). This measures which one MARINA1
matches, by comparing stored list lengths against RDKit's atom counts and symmetry
classes on a sample.

Also measures, for the molecules actually substituted in MARINA2, how much the list
length changed -- the direct size of any convention mismatch introduced.
"""
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

OUT = Path(__file__).resolve().parent.parent / "results"
DATA_ROOT = Path(os.environ.get("MARINA_DATA_ROOT", "/home/user/atong"))
M1 = DATA_ROOT / "Datasets/MARINA1"
M2 = DATA_ROOT / "Datasets/MARINA2"
SAMPLE = 4000
random.seed(0)


def counts(smiles):
    """-> (n_carbons, n_unique_carbon_envs, n_hydrogens, n_unique_h_envs)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    molh = Chem.AddHs(mol)
    ranks = list(Chem.CanonicalRankAtoms(molh, breakTies=False))
    c_idx = [a.GetIdx() for a in molh.GetAtoms() if a.GetAtomicNum() == 6]
    h_idx = [a.GetIdx() for a in molh.GetAtoms() if a.GetAtomicNum() == 1]
    return (len(c_idx), len({ranks[i] for i in c_idx}),
            len(h_idx), len({ranks[i] for i in h_idx}))


def shard(ds, split, name):
    t = pq.read_table(ds / "arrow" / split / f"{name}.parquet").to_pydict()
    return {i: d for i, d in zip(t["idx"], t["data"])}


index = pickle.load(open(M1 / "index.pkl", "rb"))
val = [i for i, v in index.items() if v["split"] == "val"]
c1, h1 = shard(M1, "val", "C_NMR"), shard(M1, "val", "H_NMR")
c2, h2 = shard(M2, "val", "C_NMR"), shard(M2, "val", "H_NMR")

pool = random.sample([i for i in val if i in c1 and i in h1], min(SAMPLE, len(val)))
rows = []
for i in pool:
    cnt = counts(index[i]["smiles"])
    if cnt is None:
        continue
    n_c, n_cu, n_h, n_hu = cnt
    if n_c == 0 or n_h == 0:
        continue
    rows.append((len(c1[i]) / n_c, len(c1[i]) / max(n_cu, 1),
                 len(h1[i]) / n_h, len(h1[i]) / max(n_hu, 1),
                 len(c1[i]) == n_c, len(c1[i]) == n_cu,
                 len(h1[i]) == n_h, len(h1[i]) == n_hu))
a = np.array(rows, dtype=float)

report = {
    "sample": len(a),
    "marina1_c_nmr": {
        "len/total_carbons  median": round(float(np.median(a[:, 0])), 3),
        "len/unique_envs    median": round(float(np.median(a[:, 1])), 3),
        "exact match to total carbons": round(float(a[:, 4].mean()), 4),
        "exact match to unique envs": round(float(a[:, 5].mean()), 4),
    },
    "marina1_h_nmr": {
        "len/total_hydrogens median": round(float(np.median(a[:, 2])), 3),
        "len/unique_envs     median": round(float(np.median(a[:, 3])), 3),
        "exact match to total hydrogens": round(float(a[:, 6].mean()), 4),
        "exact match to unique envs": round(float(a[:, 7].mean()), 4),
    },
}

# How much did substitution change list length, for molecules actually substituted?
for mod, before, after in (("c_nmr", c1, c2), ("h_nmr", h1, h2)):
    changed = [i for i in before if i in after and list(before[i]) != list(after[i])]
    if not changed:
        continue
    ratio = np.array([len(after[i]) / max(len(before[i]), 1) for i in changed])
    report[f"substitution_length_change_{mod}"] = {
        "molecules_substituted_in_val": len(changed),
        "len_after/len_before median": round(float(np.median(ratio)), 3),
        "p10": round(float(np.percentile(ratio, 10)), 3),
        "p90": round(float(np.percentile(ratio, 90)), 3),
        "frac_shorter": round(float((ratio < 0.95).mean()), 4),
        "frac_longer": round(float((ratio > 1.05).mean()), 4),
    }

OUT.mkdir(exist_ok=True)
(OUT / "convention_check.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
