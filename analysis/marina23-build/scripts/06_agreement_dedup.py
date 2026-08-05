"""Re-run the NMRexp/nmrshiftdb2 agreement check, controlling for peak convention.

The first pass (05) compared nmrshiftdb2's PER-ATOM shift lists against NMRexp's
PER-PEAK lists with greedy nearest-neighbour matching. That is not apples-to-apples:
nmrshiftdb2 repeats a shift once per symmetry-equivalent carbon while NMRexp reports
one peak for the whole set, so the surplus atoms get force-matched to unrelated peaks
and inflate the deviation. NMRexp's own paper reports 98% skeleton accuracy, an order
of magnitude better than the 20.3% gross-disagreement that pass produced -- which is
itself evidence the method, not the data, was the problem.

Here both sides are collapsed to unique shift values (within a tolerance) before
matching, which is the comparison the two conventions actually support. Reports both
the naive and deduplicated numbers so the size of the artifact is visible.

Writes results/nmrexp_agreement_dedup.json
"""
import ast
import json
import os
import pickle
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("MARINA_DATA_ROOT", "/home/user/atong"))
OUT = ROOT / "results"
DOMAIN_RAW = ROOT.parent / "domain-compare" / "raw"
NSDB = DOMAIN_RAW / "nmrshiftdb2/data/nmrshiftdb2_2024/mol_nmrshift_nmrshiftdb2_2024_all.pkl"
NMREXP = DOMAIN_RAW / "nmrexp/NMRexp_10to24_1_1004.parquet"

INTEGRATION = re.compile(r"(\d+(?:\.\d+)?)\s*H", re.IGNORECASE)
C_RANGE, H_RANGE = (-30.0, 260.0), (-5.0, 20.0)
# Two shifts closer than this are the same signal for collapsing purposes.
DEDUP_TOL = {"c_nmr": 0.2, "h_nmr": 0.02}


def canon(s):
    if not isinstance(s, str) or not s.strip():
        return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    o = Chem.MolToSmiles(m, isomericSmiles=False, canonical=True)
    m = Chem.MolFromSmiles(o)
    return Chem.MolToSmiles(m, isomericSmiles=False, canonical=True) if m else None


def collapse(shifts, tol):
    """Merge shifts within tol into one representative value."""
    out = []
    for s in sorted(shifts):
        if not out or abs(s - out[-1]) > tol:
            out.append(s)
    return out


def matched_deviation(a, b):
    a, b = sorted(a), sorted(b)
    if not a or not b:
        return None
    pool, devs = list(b), []
    for x in a:
        if not pool:
            break
        j = min(range(len(pool)), key=lambda k: abs(pool[k] - x))
        devs.append(abs(pool.pop(j) - x))
    return float(np.median(devs))


targets = {v["smiles"] for v in
           pickle.load(open(DATA_ROOT / "Datasets/MARINA1/index.pkl", "rb")).values()}

nsdb = pickle.load(open(NSDB, "rb"))
with Pool(4) as p:
    u = sorted({s for s in nsdb["smiles"] if isinstance(s, str) and s.strip()})
    nmap = dict(zip(u, p.map(canon, u, chunksize=1000)))
nsdb_c, nsdb_h = {}, {}
for raw, tgt, mask in zip(nsdb["smiles"], nsdb["atom_target"], nsdb["atom_mask"]):
    c = nmap.get(raw)
    if c is None or c not in targets:
        continue
    cs = [float(t) for t, m in zip(tgt, mask) if int(m) == 6 and C_RANGE[0] <= float(t) <= C_RANGE[1]]
    hs = [float(t) for t, m in zip(tgt, mask) if int(m) == 1 and H_RANGE[0] <= float(t) <= H_RANGE[1]]
    if cs and len(cs) > len(nsdb_c.get(c, [])):
        nsdb_c[c] = cs
    if hs and len(hs) > len(nsdb_h.get(c, [])):
        nsdb_h[c] = hs

d = pq.read_table(NMREXP, columns=["SMILES", "NMR_type", "NMR_processed"]).to_pandas()
d = d[d.NMR_type.isin(["13C NMR", "1H NMR"])]
with Pool(4) as p:
    u = sorted(set(d.SMILES.dropna()))
    emap = dict(zip(u, p.map(canon, u, chunksize=2000)))
d["canon"] = d.SMILES.map(emap)
d = d[d.canon.notna() & d.canon.isin(targets)]
nex_c, nex_h = {}, {}
for c, typ, proc in zip(d.canon, d.NMR_type, d.NMR_processed):
    try:
        entries = ast.literal_eval(proc) if isinstance(proc, str) else proc
    except (ValueError, SyntaxError):
        continue
    if not entries:
        continue
    shifts = []
    if typ == "13C NMR":
        for e in entries:
            try:
                s = float(e[0])
            except (TypeError, ValueError, IndexError):
                continue
            if C_RANGE[0] <= s <= C_RANGE[1]:
                shifts.append(s)
        tgt = nex_c
    else:
        for e in entries:
            try:
                lo, hi = float(e[3]), float(e[4])
            except (TypeError, ValueError, IndexError):
                continue
            mid = (lo + hi) / 2.0
            if H_RANGE[0] <= mid <= H_RANGE[1]:
                shifts.append(mid)          # NOT integration-expanded: peak-level compare
        tgt = nex_h
    if shifts and len(shifts) > len(tgt.get(c, [])):
        tgt[c] = shifts
print(f"nmrshiftdb2 c {len(nsdb_c):,} h {len(nsdb_h):,} | NMRexp c {len(nex_c):,} h {len(nex_h):,}",
      flush=True)

report = {}
for label, A, B in (("c_nmr", nsdb_c, nex_c), ("h_nmr", nsdb_h, nex_h)):
    tol = DEDUP_TOL[label]
    gross_tol = 5.0 if label == "c_nmr" else 0.5
    both = sorted(set(A) & set(B))
    naive, dedup, ratio = [], [], []
    for s in both:
        n = matched_deviation(A[s], B[s])
        a2, b2 = collapse(A[s], tol), collapse(B[s], tol)
        dd = matched_deviation(a2, b2)
        if n is None or dd is None:
            continue
        naive.append(n)
        dedup.append(dd)
        ratio.append(len(b2) / len(a2))
    naive, dedup, ratio = np.array(naive), np.array(dedup), np.array(ratio)
    report[label] = {
        "molecules_compared": len(both),
        "naive_per_atom_vs_per_peak": {
            "median_deviation_ppm": round(float(np.median(naive)), 4),
            "gross_disagreement_frac": round(float((naive > gross_tol).mean()), 4),
        },
        "deduplicated_peak_vs_peak": {
            "median_deviation_ppm": round(float(np.median(dedup)), 4),
            "gross_disagreement_frac": round(float((dedup > gross_tol).mean()), 4),
            "unique_peak_count_ratio_median": round(float(np.median(ratio)), 3),
        },
        "gross_disagreement_tolerance_ppm": gross_tol,
    }

OUT.mkdir(exist_ok=True)
(OUT / "nmrexp_agreement_dedup.json").write_text(json.dumps(report, indent=2))
print("\n" + json.dumps(report, indent=2))
