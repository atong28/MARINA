#!/usr/bin/env python3
"""check_open_identity.py -- MARINA-DB-OPEN leaves everything but NMR of MARINA-DB untouched.

build_open.py may only (a) swap the NMR of existing molecules (HSQC/C_NMR/H_NMR shards and the
has_hsqc/has_c_nmr/has_h_nmr flags), (b) APPEND index rows, retrieval rows, rankingset rows,
FragIdx rows and metadata rows, and (c) add a `ch_nmr_np` field to metadata rows. This checks,
against the source MARINA-DB, for every PRE-EXISTING idx / retrieval idx:

  1. every FP family's bitinfo_to_idx.pkl is byte-identical (vocab never recomputed)
  2. index.pkl fields other than the three NMR has_* flags are identical; new idx follow max+1
  3. MassSpec / MassSpecNeg / every FragIdx* shard rows are identical, per split
  4. retrieval.pkl rows identical; appended rows are contiguous
  5. rankingset.pt: the first N_src CSR rows are identical, column count unchanged
  6. metadata.json rows identical apart from the added `ch_nmr_np`; one row per retrieval row
  7. the fixed-vocab row construction used for the appended rows reproduces a sample of the
     source's own rankingset and FragIdx rows (guards radius / feature-kind drift)
  8. nmr_sources.parquet covers exactly the output index idx set, is_new marks exactly the
     appended rows, and each modality source is "absent" iff the index has_* flag is False
     (new rows never "mnova")

Exits nonzero on any failure.

    DATASET_ROOT=$PWD/data/dataset pixi run python scripts/marina_db/check_open_identity.py \
        --out ../Datasets/MARINA-DB-OPEN
"""
import argparse
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2]))
from config import DATA_DATASET, FP_RADIUS
from src.modules.data.fp_loader import FP_LOADERS
from src.modules.data.fp_utils import _init_csr, _worker_row_nonzeros

SPLITS = ("train", "val", "test")
NMR_FLAGS = {"has_hsqc", "has_c_nmr", "has_h_nmr"}
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def same_value(x, y):
    if isinstance(x, (list, tuple, np.ndarray)) or isinstance(y, (list, tuple, np.ndarray)):
        return type(x) is type(y) and np.array_equal(np.asarray(x), np.asarray(y))
    return type(x) is type(y) and x == y


def rows_for(table, idxs):
    t = table.filter(pc.is_in(table["idx"], value_set=idxs))
    return t.take(pc.sort_indices(t, [("idx", "ascending")]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DATA_DATASET), help="source MARINA-DB root")
    ap.add_argument("--out", required=True, help="built MARINA-DB-OPEN root")
    ap.add_argument("--sample", type=int, default=200, help="rows per family for check 7")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    fp_types = [t for t in FP_LOADERS if (src / t / "bitinfo_to_idx.pkl").exists()]

    # 1. vocab
    for t in fp_types:
        check(f"{t}: bitinfo_to_idx.pkl byte-identical",
              (out / t / "bitinfo_to_idx.pkl").exists() and
              sha(src / t / "bitinfo_to_idx.pkl") == sha(out / t / "bitinfo_to_idx.pkl"))

    # 2. index
    si = pickle.load(open(src / "index.pkl", "rb"))
    oi = pickle.load(open(out / "index.pkl", "rb"))
    check("every source idx present in output index", set(si) <= set(oi))
    bad = [i for i in si if i in oi and (set(si[i]) != set(oi[i]) or
           any(not same_value(si[i][k], oi[i][k]) for k in si[i] if k not in NMR_FLAGS))]
    check("pre-existing index rows identical except NMR has_* flags", not bad,
          f"{len(bad)} differ, first {bad[:5]}")
    new = sorted(set(oi) - set(si))
    check("new index idx are max+1.. contiguous", new == list(range(max(si) + 1, max(si) + 1 + len(new))),
          f"{len(new)} new rows")
    src_idx = pa.array(sorted(si), pa.int64())

    # 3. non-NMR shards
    for sp in SPLITS:
        names = sorted(p.name for p in (src / "arrow" / sp).glob("*.parquet")
                       if p.stem.startswith(("MassSpec", "FragIdx")))
        for name in names:
            s = rows_for(pq.read_table(src / "arrow" / sp / name), src_idx)
            o = rows_for(pq.read_table(out / "arrow" / sp / name), src_idx)
            check(f"{sp}/{name}: pre-existing rows identical", s.equals(o), f"{s.num_rows} vs {o.num_rows}")

    # 4. retrieval
    sr = pickle.load(open(src / "retrieval.pkl", "rb"))
    orr = pickle.load(open(out / "retrieval.pkl", "rb"))
    check("pre-existing retrieval rows identical", all(orr.get(k) == v for k, v in sr.items()))
    check("retrieval idx 0..N-1 (appended rows contiguous)", sorted(orr) == list(range(len(orr))),
          f"{len(sr)} -> {len(orr)}")

    # 5. rankingset prefix
    for t in fp_types:
        s = torch.load(src / t / "rankingset.pt", weights_only=True)
        o = torch.load(out / t / "rankingset.pt", weights_only=True)
        n, nnz = s.shape[0], int(s.crow_indices()[-1])
        ok = (o.shape == (len(orr), s.shape[1]) and torch.equal(o.crow_indices()[:n + 1], s.crow_indices())
              and torch.equal(o.col_indices()[:nnz], s.col_indices()) and torch.equal(o.values()[:nnz], s.values()))
        check(f"{t}: rankingset first {n} rows identical, shape {tuple(o.shape)}", ok)

    # 6. metadata
    sm = json.load(open(src / "metadata.json"))
    om = json.load(open(out / "metadata.json"))
    bad = [k for k, v in sm.items() if {kk: vv for kk, vv in om.get(k, {}).items() if kk != "ch_nmr_np"} != v]
    check("pre-existing metadata rows identical apart from ch_nmr_np", not bad, f"{len(bad)} differ, first {bad[:5]}")
    check("metadata has one row per retrieval row", set(om) == {str(k) for k in orr})
    del sm, om

    # 7. fixed-vocab reconstruction reproduces source rows
    rng = random.Random(0)
    ret_sample = rng.sample(sorted(sr), min(a.sample, len(sr)))
    idx_sample = rng.sample(sorted(si), min(a.sample, len(si)))
    for t in fp_types:
        loader = FP_LOADERS[t]
        _init_csr(FP_RADIUS, pickle.load(open(src / t / "bitinfo_to_idx.pkl", "rb")), loader.FEATURE_KIND)
        rs = torch.load(src / t / "rankingset.pt", weights_only=True)
        crow, col = rs.crow_indices(), rs.col_indices()
        bad = [k for k in ret_sample if _worker_row_nonzeros((k, sr[k]["smiles"]))[1] !=
               col[crow[k]:crow[k + 1]].tolist()]
        check(f"{t}: reconstruction reproduces {len(ret_sample)} source rankingset rows", not bad, f"{len(bad)} differ")
        frag = {}
        for sp in SPLITS:
            tab = pq.read_table(src / "arrow" / sp / loader.FRAGIDX_FILENAME)
            want = pa.array([i for i in idx_sample if si[i]["split"] == sp], pa.int64())
            tab = tab.filter(pc.is_in(tab["idx"], value_set=want)).to_pydict()
            frag.update(zip(tab["idx"], tab["cols"]))
        bad = [i for i in idx_sample if _worker_row_nonzeros((i, si[i]["smiles"]))[1] != frag.get(i)]
        check(f"{t}: reconstruction reproduces {len(idx_sample)} source FragIdx rows", not bad, f"{len(bad)} differ")

    # 8. per-idx NMR provenance table
    ns = pq.read_table(out / "nmr_sources.parquet").to_pydict()
    check("nmr_sources.parquet idx set == output index idx set (no duplicates)",
          sorted(ns["idx"]) == sorted(oi) and len(ns["idx"]) == len(oi), f"{len(ns['idx'])} rows")
    check("nmr_sources is_new marks exactly the appended index rows",
          all(n == (i not in si) for i, n in zip(ns["idx"], ns["is_new"])))
    bad = [i for k, i in enumerate(ns["idx"]) for mod in ("hsqc", "c_nmr", "h_nmr")
           if ns[mod][k] not in ("chnmr", "mnova", "absent")
           or (ns[mod][k] == "absent") == oi[i][f"has_{mod}"]
           or (ns["is_new"][k] and ns[mod][k] == "mnova")]
    check("nmr_sources consistent with index has_* flags", not bad, f"{len(bad)} bad, first {bad[:5]}")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILURES: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
