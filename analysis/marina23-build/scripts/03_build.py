"""Build MARINA2 and MARINA3 from MARINA1 plus the experimental extracts.

MARINA2 -- MARINA1 with experimental spectra substituted in wherever they exist.
           Same 487,028 molecules, same idx, same splits, same retrieval set and
           same Sherlock-FP vocabulary as MARINA1, so MARINA1-vs-MARINA2 isolates
           the spectra and nothing else. Experimental wins on conflict; where
           MARINA1 had no data for a modality and an experimental spectrum exists,
           coverage goes up.

MARINA3 -- the purely experimental slice: only molecules carrying at least one
           experimental modality, and for each, only its experimental modalities.
           Simulated spectra are dropped rather than kept. idx, splits, and the
           retrieval set are inherited unchanged from MARINA1 so a model pretrained
           on MARINA1 can be finetuned on MARINA3 with no leakage and benchmarked
           against the same candidate library.

HSQC is not substituted in either: MARINA1's experimental HSQC (JEOL) is already
the winning source for those molecules, and no new experimental HSQC exists.
Negative-mode MS/MS is out of scope -- it needs a modality slot the schema lacks.
FragIdx is the prediction target, derived from SMILES, so it is carried unchanged.
"""
import json
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("MARINA_DATA_ROOT", "/home/user/atong"))
RESULTS = ROOT / "results"
DATASETS = DATA_ROOT / "Datasets"
M1 = DATASETS / "MARINA1"
M2 = DATASETS / "MARINA2"
M3 = DATASETS / "MARINA3"
HSQC_PROV = ROOT.parent / "finetune-exp" / "results" / "marina1_hsqc_classified.parquet"

SPLITS = ("train", "val", "test")
MOD_FILE = {"c_nmr": "C_NMR", "h_nmr": "H_NMR", "hsqc": "HSQC_NMR", "mass_spec": "MassSpec"}
COPY_FILES = ["retrieval.pkl", "metadata.json", "count_hashes_under_radius_6.pkl"]
COPY_DIRS = ["RankingEntropy"]


def load_shard(path):
    """-> {idx: (data list, shape list)}"""
    t = pq.read_table(path).to_pydict()
    return {i: (d, s) for i, d, s in zip(t["idx"], t["data"], t["shape"])}


def write_shard(path, rows, schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    idxs = sorted(rows)
    table = pa.table(
        {
            "idx": pa.array(idxs, type=schema.field("idx").type),
            "data": pa.array([rows[i][0] for i in idxs], type=schema.field("data").type),
            "shape": pa.array([rows[i][1] for i in idxs], type=schema.field("shape").type),
        },
        schema=schema,
    )
    pq.write_table(table, path)


def filter_by_idx(src, dst, keep):
    """Copy a parquet keeping only rows whose idx is in `keep`, schema preserved.

    Used for FragIdx, whose schema is {idx, cols} rather than {idx, data, shape}.
    """
    table = pq.read_table(src)
    mask = pa.array([i in keep for i in table.column("idx").to_pylist()])
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.filter(mask), dst)


def flat(values):
    return [float(v) for v in values]


index = pickle.load(open(M1 / "index.pkl", "rb"))
split_of = {i: v["split"] for i, v in index.items()}
idx_of_smiles = {v["smiles"]: i for i, v in index.items()}
print(f"MARINA1 index: {len(index):,}", flush=True)

exp_nmr = pickle.load(open(RESULTS / "nmr_experimental.pkl", "rb"))
exp_msms = pickle.load(open(RESULTS / "msms_experimental.pkl", "rb"))
prov = pd.read_parquet(HSQC_PROV)
jeol_smiles = set(prov[prov.provenance == "experimental_jeol"].smiles)
print(f"experimental NMR molecules : {len(exp_nmr):,}", flush=True)
print(f"experimental MS/MS molecules: {len(exp_msms):,}", flush=True)
print(f"experimental HSQC (JEOL)    : {len(jeol_smiles):,}", flush=True)

# idx -> experimental payload, bucketed by split
exp_rows = {m: defaultdict(dict) for m in ("c_nmr", "h_nmr", "mass_spec")}
for smiles, mods in exp_nmr.items():
    i = idx_of_smiles.get(smiles)
    if i is None:
        continue
    sp = split_of[i]
    if "c_nmr" in mods:
        exp_rows["c_nmr"][sp][i] = (flat(mods["c_nmr"]), [len(mods["c_nmr"])])
    if "h_nmr" in mods:
        exp_rows["h_nmr"][sp][i] = (flat(mods["h_nmr"]), [len(mods["h_nmr"])])
for smiles, peaks in exp_msms.items():
    i = idx_of_smiles.get(smiles)
    if i is None:
        continue
    arr = np.asarray(peaks, dtype=np.float32)
    exp_rows["mass_spec"][split_of[i]][i] = (flat(arr.reshape(-1)), list(arr.shape))

jeol_idx = {idx_of_smiles[s] for s in jeol_smiles if s in idx_of_smiles}

# ------------------------------------------------------------------- MARINA2
print("\n=== building MARINA2 ===", flush=True)
if M2.exists():
    shutil.rmtree(M2)
M2.mkdir(parents=True)
stats2 = defaultdict(int)
m2_flags = {}

for split in SPLITS:
    for mod, fname in MOD_FILE.items():
        src = M1 / "arrow" / split / f"{fname}.parquet"
        dst = M2 / "arrow" / split / f"{fname}.parquet"
        schema = pq.read_schema(src)
        rows = load_shard(src)
        if mod in exp_rows:
            for i, payload in exp_rows[mod][split].items():
                stats2[f"{mod}_replaced" if i in rows else f"{mod}_added"] += 1
                rows[i] = payload
        write_shard(dst, rows, schema)
        for i in rows:
            m2_flags.setdefault(i, set()).add(mod)
    shutil.copy2(M1 / "arrow" / split / "FragIdx.parquet",
                 M2 / "arrow" / split / "FragIdx.parquet")
    print(f"  {split} done", flush=True)

m2_index = {}
for i, entry in index.items():
    e = dict(entry)
    have = m2_flags.get(i, set())
    e["has_c_nmr"] = "c_nmr" in have
    e["has_h_nmr"] = "h_nmr" in have
    e["has_hsqc"] = "hsqc" in have
    e["has_mass_spec"] = "mass_spec" in have
    m2_index[i] = e
pickle.dump(m2_index, open(M2 / "index.pkl", "wb"))
for f in COPY_FILES:
    shutil.copy2(M1 / f, M2 / f)
for d in COPY_DIRS:
    shutil.copytree(M1 / d, M2 / d)
print(f"  MARINA2 substitutions: {dict(stats2)}", flush=True)

# ------------------------------------------------------------------- MARINA3
print("\n=== building MARINA3 ===", flush=True)
if M3.exists():
    shutil.rmtree(M3)
M3.mkdir(parents=True)

keep = set(jeol_idx)
for mod in ("c_nmr", "h_nmr", "mass_spec"):
    for split in SPLITS:
        keep |= set(exp_rows[mod][split])
print(f"  molecules with >=1 experimental modality: {len(keep):,}", flush=True)

m3_flags = defaultdict(set)
for split in SPLITS:
    keep_split = {i for i in keep if split_of[i] == split}
    for mod, fname in MOD_FILE.items():
        src = M1 / "arrow" / split / f"{fname}.parquet"
        dst = M3 / "arrow" / split / f"{fname}.parquet"
        schema = pq.read_schema(src)
        if mod == "hsqc":
            # experimental HSQC already lives in MARINA1; keep only the JEOL rows
            m1rows = load_shard(src)
            rows = {i: v for i, v in m1rows.items() if i in jeol_idx}
        else:
            rows = dict(exp_rows[mod][split])
        write_shard(dst, rows, schema)
        for i in rows:
            m3_flags[i].add(mod)

    filter_by_idx(M1 / "arrow" / split / "FragIdx.parquet",
                  M3 / "arrow" / split / "FragIdx.parquet", keep_split)
    print(f"  {split} done ({len(keep_split):,} molecules)", flush=True)

m3_index = {}
for i in sorted(keep):
    e = dict(index[i])
    have = m3_flags.get(i, set())
    e["has_c_nmr"] = "c_nmr" in have
    e["has_h_nmr"] = "h_nmr" in have
    e["has_hsqc"] = "hsqc" in have
    e["has_mass_spec"] = "mass_spec" in have
    m3_index[i] = e
pickle.dump(m3_index, open(M3 / "index.pkl", "wb"))
for f in COPY_FILES:
    shutil.copy2(M1 / f, M3 / f)
for d in COPY_DIRS:
    shutil.copytree(M1 / d, M3 / d)

# --------------------------------------------------------------------- report
def coverage(idx_map):
    return {k: int(sum(v[k] for v in idx_map.values()))
            for k in ("has_hsqc", "has_c_nmr", "has_h_nmr", "has_mass_spec")}


report = {
    "marina1": {"molecules": len(index), **coverage(index)},
    "marina2": {"molecules": len(m2_index), **coverage(m2_index),
                "substitutions": dict(stats2)},
    "marina3": {"molecules": len(m3_index), **coverage(m3_index),
                "by_split": pd.Series([v["split"] for v in m3_index.values()])
                              .value_counts().to_dict()},
}
(RESULTS / "build_report.json").write_text(json.dumps(report, indent=2))
print("\n" + json.dumps(report, indent=2))
