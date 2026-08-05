"""Sharper discriminators between the two SPECTRE HSQC populations.

validate_provenance.py's peak-count test came back identical for both classes
(and was mis-specified: it counted one cross-peak per CH2, but diastereotopic
CH2 protons give two, which is why both classes "over-count").

Two better tests here:

1. Decimal precision of the shifts. Peak lists transcribed from a publication
   are rounded (2 dp for 13C, 2-3 dp for 1H). A simulator emits full float
   precision. This separates transcribed-experimental from computed.

2. Shape of the third column. If it is a real detector intensity it should be
   heavy-tailed and roughly log-normal, with |CH2| and |CH| populations
   overlapping. A synthetic weight would look quantized or bimodal.

Writes results/hsqc_discrimination.json.
"""
import json
import pickle
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
SPECTRE_INDEX = HERE / "raw/spectre/index.pkl"
SPECTRE_LMDB = str(HERE / "raw/spectre/_lmdb/{split}/HSQC_NMR.lmdb")
SAMPLE = 4000


def parse_tensor(buf):
    h1 = buf.index(b"|")
    h2 = buf.index(b"|", h1 + 1)
    h3 = buf.index(b"|", h2 + 1)
    dtype = np.dtype(buf[:h1].decode())
    shape = tuple(int(x) for x in buf[h2 + 1:h3].split(b","))
    return np.frombuffer(buf[h3 + 1:], dtype=dtype).reshape(shape)


def classify(arr):
    return "sign_only" if np.all(np.isin(arr[:, 2], (-1.0, 1.0))) else "intensity"


def n_decimals(x, max_dp=6):
    """Smallest dp such that round(x, dp) == x, within float tolerance."""
    for dp in range(max_dp + 1):
        if abs(x - round(x, dp)) < 1e-9:
            return dp
    return max_dp + 1


buckets = {"sign_only": [], "intensity": []}
rng = np.random.default_rng(0)
for split in ("train", "val", "test"):
    env = lmdb.open(SPECTRE_LMDB.format(split=split), readonly=True, lock=False)
    with env.begin() as txn:
        for k, v in tqdm(txn.cursor(), desc=f"scan/{split}"):
            arr = parse_tensor(v)
            if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
                continue
            c = classify(arr)
            if len(buckets[c]) < SAMPLE:
                buckets[c].append(arr.copy())
    env.close()

out = {}
for cls, arrs in buckets.items():
    C = np.concatenate([a[:, 0] for a in arrs])
    H = np.concatenate([a[:, 1] for a in arrs])
    T = np.concatenate([a[:, 2] for a in arrs])

    c_dp = Counter(n_decimals(float(x)) for x in C[:60000])
    h_dp = Counter(n_decimals(float(x)) for x in H[:60000])
    tot_c = sum(c_dp.values())
    tot_h = sum(h_dp.values())

    entry = {
        "n_molecules": len(arrs),
        "n_peaks": int(len(C)),
        "c_shift": {"min": round(float(C.min()), 2),
                    "max": round(float(C.max()), 2),
                    "mean": round(float(C.mean()), 2)},
        "h_shift": {"min": round(float(H.min()), 2),
                    "max": round(float(H.max()), 2),
                    "mean": round(float(H.mean()), 2)},
        "c_decimals_pct": {str(k): round(100 * v / tot_c, 2)
                           for k, v in sorted(c_dp.items())},
        "h_decimals_pct": {str(k): round(100 * v / tot_h, 2)
                           for k, v in sorted(h_dp.items())},
    }
    if cls == "intensity":
        absT = np.abs(T)
        pos, neg = T[T > 0], T[T < 0]
        entry["third_col"] = {
            "abs_min": round(float(absT.min()), 3),
            "abs_median": round(float(np.median(absT)), 1),
            "abs_max": round(float(absT.max()), 1),
            "pct_negative": round(float(100 * (T < 0).mean()), 2),
            "pos_abs_median": round(float(np.median(pos)), 1),
            "neg_abs_median": round(float(np.median(np.abs(neg))), 1),
            "n_distinct_values_in_10k": int(len(np.unique(T[:10000]))),
            "log10_abs_std": round(float(np.std(np.log10(absT + 1e-9))), 3),
        }
    out[cls] = entry
    print(f"\n--- {cls} ---")
    print(json.dumps(entry, indent=2))

with open(RESULTS / "hsqc_discrimination.json", "w") as f:
    json.dump(out, f, indent=2)
