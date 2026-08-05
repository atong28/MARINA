"""Schema probes for the two upstream sources we join against MARINA1.

Prints structure only -- no outputs written. Run first to confirm assumptions
before hsqc_provenance.py / msms_overlap.py.
"""
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
SPECTRE_INDEX = HERE / "raw/spectre/index.pkl"
SPECTRE_LMDB = str(HERE / "raw/spectre/_lmdb/{split}/HSQC_NMR.lmdb")
MARINA1_INDEX = os.path.join(DATA_ROOT, "Datasets/MARINA1/index.pkl")
MSG_TSV = HERE.parent / "domain-compare/raw/massspecgym/MassSpecGym.tsv"


def show(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


show("SPECTRE (MoonshotDatasetv3) index.pkl")
sp = pickle.load(open(SPECTRE_INDEX, "rb"))
print(f"entries: {len(sp):,}  key type: {type(next(iter(sp)))}")
k0 = next(iter(sp))
print(f"sample key {k0!r} ->")
for kk, vv in sp[k0].items():
    print(f"   {kk:20s} {type(vv).__name__:10s} {str(vv)[:70]}")
splits = {}
for v in sp.values():
    splits[v.get("split")] = splits.get(v.get("split"), 0) + 1
print("splits:", splits)
has_keys = [k for k in sp[k0] if k.startswith("has_")]
print("has_* fields:", has_keys)
for hk in has_keys:
    print(f"   {hk}: {sum(1 for v in sp.values() if v.get(hk)):,}")

show("SPECTRE HSQC LMDB")
env = lmdb.open(SPECTRE_LMDB.format(split="train"), readonly=True, lock=False)
with env.begin() as txn:
    stat = txn.stat()
    print("entries:", f"{stat['entries']:,}")
    cur = txn.cursor()
    for i, (k, v) in enumerate(cur):
        try:
            obj = pickle.loads(v)
            desc = f"pickle {type(obj).__name__} shape={getattr(obj, 'shape', None)}"
            head = str(np.asarray(obj).reshape(-1)[:6])
        except Exception:
            obj = np.frombuffer(v, dtype=np.float32)
            desc = f"raw float32 n={obj.size}"
            head = str(obj[:6])
        print(f"  key={k!r} bytes={len(v)} {desc} head={head}")
        if i >= 2:
            break
env.close()

show("MARINA1 index.pkl")
m1 = pickle.load(open(MARINA1_INDEX, "rb"))
print(f"entries: {len(m1):,}")
k0 = next(iter(m1))
for kk, vv in m1[k0].items():
    print(f"   {kk:20s} {type(vv).__name__:10s} {str(vv)[:70]}")

show("MassSpecGym TSV")
msg = pd.read_csv(MSG_TSV, sep="\t", nrows=5)
print(msg.dtypes)
print(msg[["identifier", "smiles", "adduct", "instrument_type",
           "collision_energy", "fold"]].to_string())
