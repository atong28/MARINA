"""Does NMRexp agree with nmrshiftdb2 where both cover the same MARINA1 molecule?

NMRexp supplies the majority of MARINA2/MARINA3's experimental 1D NMR, and its
structures are OCSR-read from drawn structures in SI PDFs -- a labelling process
with an unquantified error rate. nmrshiftdb2 is a curated database with per-atom
assignments. Where both cover a molecule we get a free validation set: agreement
is evidence the mining pipeline is sound, disagreement is evidence it is not.

Compares peak counts and, via greedy nearest-neighbour matching (assignment-free,
since NMRexp has no atom indices), the matched-peak absolute deviation.

Writes results/nmrexp_agreement.json
"""
import ast
import json
import os
import pickle
import re
from collections import defaultdict
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


def canon(s):
    if not isinstance(s, str) or not s.strip():
        return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    o = Chem.MolToSmiles(m, isomericSmiles=False, canonical=True)
    m = Chem.MolFromSmiles(o)
    return Chem.MolToSmiles(m, isomericSmiles=False, canonical=True) if m else None


def matched_deviation(a, b):
    """Greedy nearest-neighbour match between two shift lists -> (median|dev|, matched frac)."""
    a, b = sorted(a), sorted(b)
    if not a or not b:
        return None, 0.0
    pool = list(b)
    devs = []
    for x in a:
        if not pool:
            break
        j = min(range(len(pool)), key=lambda k: abs(pool[k] - x))
        devs.append(abs(pool.pop(j) - x))
    return float(np.median(devs)), len(devs) / max(len(a), len(b))


targets = {v["smiles"] for v in
           pickle.load(open(DATA_ROOT / "Datasets/MARINA1/index.pkl", "rb")).values()}

# ---- nmrshiftdb2 (per-atom, curated)
nsdb = pickle.load(open(NSDB, "rb"))
with Pool(4) as p:
    uniq = sorted({s for s in nsdb["smiles"] if isinstance(s, str) and s.strip()})
    nmap = dict(zip(uniq, p.map(canon, uniq, chunksize=1000)))
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
print(f"nmrshiftdb2 in MARINA1: c {len(nsdb_c):,} h {len(nsdb_h):,}", flush=True)

# ---- NMRexp (per-peak, OCSR-mined)
d = pq.read_table(NMREXP, columns=["SMILES", "NMR_type", "NMR_processed"]).to_pandas()
d = d[d.NMR_type.isin(["13C NMR", "1H NMR"])]
with Pool(4) as p:
    uniq = sorted(set(d.SMILES.dropna()))
    emap = dict(zip(uniq, p.map(canon, uniq, chunksize=2000)))
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
            if not (H_RANGE[0] <= mid <= H_RANGE[1]):
                continue
            m = INTEGRATION.search(str(e[2])) if len(e) > 2 and e[2] is not None else None
            shifts.extend([mid] * max(1, min(int(float(m.group(1))) if m else 1, 60)))
        tgt = nex_h
    if shifts and len(shifts) > len(tgt.get(c, [])):
        tgt[c] = shifts
print(f"NMRexp in MARINA1: c {len(nex_c):,} h {len(nex_h):,}", flush=True)

report = {}
for label, A, B in (("c_nmr", nsdb_c, nex_c), ("h_nmr", nsdb_h, nex_h)):
    both = sorted(set(A) & set(B))
    devs, fracs, cnt_ratio, exact_cnt = [], [], [], 0
    for s in both:
        med, frac = matched_deviation(A[s], B[s])
        if med is None:
            continue
        devs.append(med)
        fracs.append(frac)
        cnt_ratio.append(len(B[s]) / len(A[s]))
        exact_cnt += int(len(A[s]) == len(B[s]))
    devs, cnt_ratio = np.array(devs), np.array(cnt_ratio)
    tol = 1.0 if label == "c_nmr" else 0.1
    report[label] = {
        "molecules_compared": len(both),
        "median_matched_deviation_ppm": round(float(np.median(devs)), 4),
        "frac_molecules_median_dev_within_tol": round(float((devs <= tol).mean()), 4),
        "tolerance_ppm": tol,
        "frac_gross_disagreement_over_5x_tol": round(float((devs > 5 * tol).mean()), 4),
        "peak_count_exact_match_frac": round(exact_cnt / len(both), 4),
        "peak_count_ratio_nmrexp_over_nmrshiftdb2": {
            "median": round(float(np.median(cnt_ratio)), 3),
            "p10": round(float(np.percentile(cnt_ratio, 10)), 3),
            "p90": round(float(np.percentile(cnt_ratio, 90)), 3),
        },
    }

OUT.mkdir(exist_ok=True)
(OUT / "nmrexp_agreement.json").write_text(json.dumps(report, indent=2))
print("\n" + json.dumps(report, indent=2))
