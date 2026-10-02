#!/usr/bin/env python3
"""Export the NMR source (mnova / chnmr / absent) of every MARINA-DB test molecule, per modality, to
paper/results/marina_db_test_nmr_sources.json — the split used by the sim_exp_gap table (Mnova-simulated vs
CH-NMR-NP experimental test molecules). Source: Datasets/MARINA-DB-OPEN/{index.pkl,nmr_sources.parquet}.

  DATASETS_ROOT=~/Workspace/Datasets pixi run python paper/tools/export_test_nmr_sources.py   (run from ~/Workspace)
"""
import json
import os
import pickle

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.environ.get("DATASETS_ROOT", os.path.expanduser("~/Workspace/Datasets")), "MARINA-DB-OPEN")

idx = pickle.load(open(os.path.join(ROOT, "index.pkl"), "rb"))
test = {v["idx"] for v in idx.values() if v["split"] == "test"}
src = pd.read_parquet(os.path.join(ROOT, "nmr_sources.parquet"))
src = src[src.idx.isin(test)].fillna("absent")
out = {str(int(r.idx)): {"hsqc": r.hsqc, "c_nmr": r.c_nmr, "h_nmr": r.h_nmr} for r in src.itertuples()}
path = os.path.join(HERE, "..", "results", "marina_db_test_nmr_sources.json")
json.dump(out, open(path, "w"), separators=(",", ":"))
print(f"{len(out)} test molecules -> {path}")
for m in ("hsqc", "c_nmr", "h_nmr"):
    print(m, src[m].value_counts().to_dict())
