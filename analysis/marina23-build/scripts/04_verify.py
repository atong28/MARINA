"""Verify MARINA2 and MARINA3 against MARINA1 and against the extracts.

Checks the properties the downstream experiment depends on: that MARINA2 differs
from MARINA1 only in substituted spectra, that MARINA3 is a strict experimental
subset, and that both share MARINA1's idx/split/retrieval universe so a model can
be pretrained on one and finetuned on another without leakage or index drift.
"""
import hashlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATA_ROOT = Path(os.environ.get("MARINA_DATA_ROOT", "/home/user/atong"))
DATASETS = DATA_ROOT / "Datasets"
M1, M2, M3 = DATASETS / "MARINA1", DATASETS / "MARINA2", DATASETS / "MARINA3"
RESULTS = Path(__file__).resolve().parent.parent / "results"
SPLITS = ("train", "val", "test")
MOD_FILE = {"c_nmr": "C_NMR", "h_nmr": "H_NMR", "hsqc": "HSQC_NMR", "mass_spec": "MassSpec"}

failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    t = pq.read_table(path).to_pydict()
    return {i: (d, s) for i, d, s in zip(t["idx"], t["data"], t["shape"])}


i1 = pickle.load(open(M1 / "index.pkl", "rb"))
i2 = pickle.load(open(M2 / "index.pkl", "rb"))
i3 = pickle.load(open(M3 / "index.pkl", "rb"))
exp_nmr = pickle.load(open(RESULTS / "nmr_experimental.pkl", "rb"))
exp_msms = pickle.load(open(RESULTS / "msms_experimental.pkl", "rb"))

print("\n--- molecule universe and splits ---")
check("MARINA2 has MARINA1's exact idx set", set(i1) == set(i2))
check("MARINA2 smiles unchanged at every idx",
      all(i1[i]["smiles"] == i2[i]["smiles"] for i in i1))
check("MARINA2 splits unchanged", all(i1[i]["split"] == i2[i]["split"] for i in i1))
check("MARINA3 idx is a subset of MARINA1", set(i3) <= set(i1))
check("MARINA3 smiles/splits inherited",
      all(i3[i]["smiles"] == i1[i]["smiles"] and i3[i]["split"] == i1[i]["split"] for i in i3))

print("\n--- shared retrieval universe (required for benchmark comparability) ---")
for f in ("retrieval.pkl", "count_hashes_under_radius_6.pkl"):
    h1 = sha(M1 / f)
    check(f"{f} identical across M1/M2/M3", h1 == sha(M2 / f) == sha(M3 / f))
h1 = sha(M1 / "RankingEntropy" / "rankingset.pt")
check("rankingset.pt identical across M1/M2/M3",
      h1 == sha(M2 / "RankingEntropy" / "rankingset.pt") == sha(M3 / "RankingEntropy" / "rankingset.pt"))

print("\n--- MARINA2: index flags agree with shard contents ---")
for mod, fname in MOD_FILE.items():
    n_flag = sum(v[f"has_{mod}"] for v in i2.values())
    n_rows = sum(pq.read_metadata(M2 / "arrow" / s / f"{fname}.parquet").num_rows for s in SPLITS)
    check(f"MARINA2 {mod}: flags == rows", n_flag == n_rows, f"{n_flag:,} vs {n_rows:,}")

print("\n--- MARINA3: index flags agree with shard contents ---")
for mod, fname in MOD_FILE.items():
    n_flag = sum(v[f"has_{mod}"] for v in i3.values())
    n_rows = sum(pq.read_metadata(M3 / "arrow" / s / f"{fname}.parquet").num_rows for s in SPLITS)
    check(f"MARINA3 {mod}: flags == rows", n_flag == n_rows, f"{n_flag:,} vs {n_rows:,}")
n_frag = sum(pq.read_metadata(M3 / "arrow" / s / "FragIdx.parquet").num_rows for s in SPLITS)
check("MARINA3 FragIdx covers every molecule", n_frag == len(i3), f"{n_frag:,} vs {len(i3):,}")

print("\n--- MARINA2 spectra actually substituted (value-level spot check) ---")
idx_of = {v["smiles"]: i for i, v in i1.items()}
substituted_idx = {idx_of[s] for s in exp_nmr if s in idx_of}
for split in SPLITS:
    r1c, r2c = rows(M1 / "arrow" / split / "C_NMR.parquet"), rows(M2 / "arrow" / split / "C_NMR.parquet")
    changed = same = 0
    for smiles, mods in exp_nmr.items():
        i = idx_of.get(smiles)
        if i is None or i1[i]["split"] != split or "c_nmr" not in mods:
            continue
        want = [round(float(x), 4) for x in sorted(mods["c_nmr"])]
        got = [round(float(x), 4) for x in sorted(r2c[i][0])]
        if want == got:
            changed += 1
        else:
            same += 1
    check(f"MARINA2/{split} c_nmr matches the extract", same == 0,
          f"{changed:,} correct, {same:,} mismatched")
    spot = [i for i in r1c if i not in substituted_idx][:2000]
    check(f"MARINA2/{split} c_nmr untouched rows byte-identical to MARINA1",
          all(list(r1c[i][0]) == list(r2c[i][0]) for i in spot), f"{len(spot):,} checked")

print("\n--- MARINA3 contains only experimental spectra ---")
exp_c = {idx_of[s] for s, m in exp_nmr.items() if "c_nmr" in m and s in idx_of}
exp_m = {idx_of[s] for s in exp_msms if s in idx_of}
for split in SPLITS:
    r3c = rows(M3 / "arrow" / split / "C_NMR.parquet")
    check(f"MARINA3/{split} C_NMR rows all have an experimental extract",
          set(r3c) <= exp_c, f"{len(r3c):,} rows")
    r3m = rows(M3 / "arrow" / split / "MassSpec.parquet")
    check(f"MARINA3/{split} MassSpec rows all have an experimental extract",
          set(r3m) <= exp_m, f"{len(r3m):,} rows")

print("\n--- MS/MS peak-value sanity (MARINA1 convention: base 100, floor 1, <=100 peaks) ---")
r3m = rows(M3 / "arrow" / "val" / "MassSpec.parquet")
bad_scale = bad_count = 0
for i, (data, shape) in r3m.items():
    a = np.asarray(data, dtype=np.float32).reshape(shape)
    if shape[0] > 100:
        bad_count += 1
    if a[:, 1].max() > 100.001 or a[:, 1].min() < 0.999:
        bad_scale += 1
check("MARINA3 MS/MS intensities within [1, 100]", bad_scale == 0, f"{bad_scale} violations")
check("MARINA3 MS/MS peak counts <= 100", bad_count == 0, f"{bad_count} violations")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURES: {failures}"))
sys.exit(1 if failures else 0)
