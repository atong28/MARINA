"""Is the intensity-valued HSQC population actually experimental?

The intensity heuristic in hsqc_provenance.py says 100,842 SPECTRE molecules are
experimental, but the public JEOL set (CH-NMR-NP) holds only ~35,500 compounds.
Either SPECTRE used a larger private JEOL set, or the heuristic is backwards and
the intensity-valued population is the ACD simulation.

Decisive test: a simulator emits exactly one cross-peak per protonated carbon
(it works from the structure). A real spectrum drops peaks -- overlap, weak
quaternary-adjacent signals, exchange broadening -- which is precisely the
"count penalty" already measured in benchmark-spectral-domain-gap.md.

So compare HSQC peak count against the protonated-carbon count implied by the
SMILES. Simulated => tight at parity. Experimental => systematic under-count.

Writes results/provenance_validation.json.
"""
import json
import pickle
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
SPECTRE_INDEX = HERE / "raw/spectre/index.pkl"
SPECTRE_LMDB = str(HERE / "raw/spectre/_lmdb/{split}/HSQC_NMR.lmdb")
SAMPLE_PER_CLASS = 6000
RNG = np.random.default_rng(0)


def parse_tensor(buf):
    h1 = buf.index(b"|")
    h2 = buf.index(b"|", h1 + 1)
    h3 = buf.index(b"|", h2 + 1)
    dtype = np.dtype(buf[:h1].decode())
    shape = tuple(int(x) for x in buf[h2 + 1:h3].split(b","))
    return np.frombuffer(buf[h3 + 1:], dtype=dtype).reshape(shape)


def classify(arr):
    third = arr[:, 2]
    return "sign_only" if np.all(np.isin(third, (-1.0, 1.0))) else "intensity"


def protonated_carbons(smiles):
    """(n_CH_CH3_carbons, n_CH2_carbons) -- the cross-peaks an HSQC should show.

    Counts carbons, not protons: HSQC gives one cross-peak per distinct C-H
    carbon environment (CH2 with inequivalent protons can give two, so this is
    a lower bound on a perfect spectrum).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    up = dn = 0
    for a in mol.GetAtoms():
        if a.GetSymbol() != "C":
            continue
        h = a.GetTotalNumHs()
        if h in (1, 3):
            up += 1
        elif h == 2:
            dn += 1
    return up, dn


sp_index = pickle.load(open(SPECTRE_INDEX, "rb"))

# collect (idx, class, npeaks, n_neg_peaks) for every HSQC entry
rows = []
for split in ("train", "val", "test"):
    env = lmdb.open(SPECTRE_LMDB.format(split=split), readonly=True, lock=False)
    with env.begin() as txn:
        for k, v in tqdm(txn.cursor(), desc=f"scan/{split}"):
            arr = parse_tensor(v)
            if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
                continue
            rows.append((int(k.decode()), classify(arr), int(arr.shape[0]),
                         int((arr[:, 2] < 0).sum())))
    env.close()

df = pd.DataFrame(rows, columns=["idx", "cls", "n_peaks", "n_neg"])
print(Counter(df.cls))

out = {}
for cls, grp in df.groupby("cls"):
    take = grp.sample(min(SAMPLE_PER_CLASS, len(grp)), random_state=0)
    recs = []
    for idx, n_peaks, n_neg in zip(take["idx"], take["n_peaks"], take["n_neg"]):
        entry = sp_index.get(idx)
        if entry is None:
            continue
        pc = protonated_carbons(entry["smiles"])
        if pc is None:
            continue
        up, dn = pc
        expected = up + dn
        if expected == 0:
            continue
        recs.append({
            "n_peaks": n_peaks,
            "expected": expected,
            "delta": n_peaks - expected,
            "ratio": n_peaks / expected,
            "n_neg": n_neg,
            "expected_ch2": dn,
            "neg_delta": n_neg - dn,
        })
    r = pd.DataFrame(recs)
    out[cls] = {
        "n_molecules_scored": int(len(r)),
        "peak_count_vs_expected": {
            "mean_delta": round(float(r["delta"].mean()), 3),
            "median_delta": float(r["delta"].median()),
            "mean_ratio": round(float(r["ratio"].mean()), 4),
            "pct_exact_match": round(float((r["delta"] == 0).mean() * 100), 2),
            "pct_under_count": round(float((r["delta"] < 0).mean() * 100), 2),
            "pct_over_count": round(float((r["delta"] > 0).mean() * 100), 2),
        },
        "ch2_sign_vs_expected": {
            "mean_neg_delta": round(float(r["neg_delta"].mean()), 3),
            "pct_exact_ch2_match": round(
                float((r["neg_delta"] == 0).mean() * 100), 2),
        },
    }
    print(f"\n--- {cls} ---")
    print(json.dumps(out[cls], indent=2))

with open(RESULTS / "provenance_validation.json", "w") as f:
    json.dump({"class_counts": {k: int(v) for k, v in Counter(df.cls).items()},
               "per_class": out}, f, indent=2)
