#!/usr/bin/env python3
"""check_marina_db.py -- verify the 2026-10-14 MARINA-DB build (build_open.py) against its reference.

The reference is the sentinel-free MARINA-DB-OPEN-OLD (2026-09-29 build + drop_mnova_sentinels). The rebuild
may only change MS/MS (current ICEBERG, NIST23 blacklist) and the fingerprints (vocab recomputed). Checks:

  1. index.pkl: same idx set; every field identical to the reference except has_mass_spec(_neg)
  2. NMR shards (C_NMR / H_NMR / HSQC_NMR): identical rows per split; no value <= -1000 (sentinel)
  3. MS/MS: has_* flag <=> row present; every row equals the current-ICEBERG prediction for the
     SMILES (re-prediction for reference rows, CH-NMR-NP run for appended rows); no NIST23 molecule
     (InChIKey first block) carries MS/MS; every non-NIST23 molecule with a prediction carries it
  4. retrieval.pkl and metadata.json identical to the reference; retrieval holds the whole journal
     and every confirmed CH-NMR-NP structure
  5. fingerprints: rankingset rows == retrieval rows; vocab size; sampled rankingset and FragIdx rows
     equal a fresh computation under the new vocab; every index idx has one FragIdx row in its split
  6. packed/ memmaps exist for every arrow shard of every split
  7. nmr_sources.parquet identical to the reference (NMR only ever from chnmr / mnova)

Exits nonzero on any failure.

    DATASET_ROOT=/tmp pixi run python scripts/marina_db/check_marina_db.py --out <new> --ref <ref> \
        --ms_all_pos <pos.jsonl.gz> --ms_all_neg <neg.jsonl.gz> --ms_pos <dir> --ms_neg <dir> \
        --ms_blacklist <nist23 dir> --structures <confirmed.tsv> --journal <benchmark-journal.pkl>
"""
import argparse
import json
import pickle
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2]))
from config import FP_RADIUS, FP_OUT_DIM
from build_open import load_iceberg, load_iceberg_jsonl, load_blacklist, _inchikey_block1, _canon_all
from src.modules.data.fp_loader import FP_LOADERS
from src.modules.data.fp_utils import _init_csr, _worker_row_nonzeros

SPLITS = ("train", "val", "test")
NMR_SHARDS = ("C_NMR", "H_NMR", "HSQC_NMR")
MS_SHARDS = {"mass_spec": "MassSpec", "mass_spec_neg": "MassSpecNeg"}
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def shard(root, split, name):
    t = pq.read_table(Path(root) / "arrow" / split / f"{name}.parquet").to_pydict()
    return {i: (d, s) for i, d, s in zip(t["idx"], t["data"], t["shape"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("out", "ref", "ms_all_pos", "ms_all_neg", "ms_pos", "ms_neg", "ms_blacklist", "structures", "journal"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out, ref = Path(a.out), Path(a.ref)
    rng = random.Random(0)

    # 1. index
    idx, ridx = pickle.load(open(out / "index.pkl", "rb")), pickle.load(open(ref / "index.pkl", "rb"))
    check("index idx set == reference", idx.keys() == ridx.keys(), f"{len(idx):,} vs {len(ridx):,}")
    ms_flags = {"has_mass_spec", "has_mass_spec_neg"}
    bad = [i for i in idx if {k: v for k, v in idx[i].items() if k not in ms_flags}
           != {k: v for k, v in ridx[i].items() if k not in ms_flags}]
    check("index fields other than has_mass_spec(_neg) identical", not bad, f"{len(bad)} differ")

    # 2. NMR
    for sp in SPLITS:
        for name in NMR_SHARDS:
            new, old = shard(out, sp, name), shard(ref, sp, name)
            check(f"{sp} {name} identical to reference", new == old, f"{len(new):,} rows")
            low = sum(any(v <= -1000 for v in d) for d, _ in new.values())
            check(f"{sp} {name} no sentinel", low == 0, f"{low} rows with a value <= -1000")

    # 3. MS/MS
    pred = {"mass_spec": load_iceberg_jsonl(a.ms_all_pos), "mass_spec_neg": load_iceberg_jsonl(a.ms_all_neg)}
    new_pred = {"mass_spec": load_iceberg(a.ms_pos, a.workers), "mass_spec_neg": load_iceberg(a.ms_neg, a.workers)}
    blacklist = load_blacklist(a.ms_blacklist)
    smiles = [e["smiles"] for e in idx.values()]
    with Pool(a.workers) as pool:
        block1 = dict(zip(smiles, pool.map(_inchikey_block1, smiles, chunksize=2000)))
    nist = {i for i, e in idx.items() if block1[e["smiles"]] in blacklist}
    src_tab = pq.read_table(out / "nmr_sources.parquet").to_pydict()
    is_new = {i for i, n in zip(src_tab["idx"], src_tab["is_new"]) if n}
    print(f"NIST23 molecules in index: {len(nist):,} "
          f"({ {sp: sum(idx[i]['split'] == sp for i in nist) for sp in SPLITS} })", flush=True)
    for mod, name in MS_SHARDS.items():
        rows = {}
        for sp in SPLITS:
            rows.update(shard(out, sp, name))
        flag = {i for i, e in idx.items() if e[f"has_{mod}"]}
        check(f"{name}: has_{mod} <=> row present", flag == set(rows), f"{len(flag):,} flagged, {len(rows):,} rows")
        check(f"{name}: no NIST23 molecule has MS/MS", not (nist & set(rows)), f"{len(nist & set(rows))} leak")
        source = {i: (new_pred if i in is_new else pred)[mod] for i in idx}
        expect = {i for i, e in idx.items() if i not in nist and e["smiles"] in source[i]}
        check(f"{name}: every non-NIST23 molecule with a current prediction has it", expect <= set(rows),
              f"{len(expect - set(rows))} missing")
        mism = 0
        for i, (d, s) in rows.items():
            p = source[i].get(idx[i]["smiles"])
            if p is None or not np.allclose(np.asarray(d).reshape(s), np.asarray(p, dtype=float), atol=1e-6):
                mism += 1
        check(f"{name}: every row == current ICEBERG prediction", mism == 0, f"{mism} mismatches")

    # 4. retrieval / metadata
    ret = pickle.load(open(out / "retrieval.pkl", "rb"))
    check("retrieval.pkl identical to reference", ret == pickle.load(open(ref / "retrieval.pkl", "rb")), f"{len(ret):,}")
    check("metadata.json identical to reference",
          (out / "metadata.json").read_bytes() == (ref / "metadata.json").read_bytes())
    in_ret = {e["smiles"] for e in ret.values()}
    journal = pickle.load(open(a.journal, "rb"))
    jc = _canon_all([e["smiles"] for e in journal.values()], a.workers)
    check("retrieval holds the whole journal", all(c in in_ret for c in jc), f"{sum(c not in in_ret for c in jc)} missing of {len(jc)}")
    tsv = [l.rstrip("\n").split("\t") for l in open(a.structures)][1:]
    cc = {c for c in _canon_all([r[1] for r in tsv], a.workers) if c}
    check("retrieval holds every confirmed CH-NMR-NP structure", cc <= in_ret, f"{len(cc - in_ret)} missing of {len(cc):,}")

    # 5. fingerprints
    for fp_type, loader in FP_LOADERS.items():
        vocab = pickle.load(open(out / fp_type / "bitinfo_to_idx.pkl", "rb"))
        rs = torch.load(out / fp_type / "rankingset.pt", weights_only=True)
        check(f"[{fp_type}] vocab size {FP_OUT_DIM}, rankingset rows == retrieval",
              len(vocab) == FP_OUT_DIM and rs.shape == (len(ret), FP_OUT_DIM), f"{len(vocab)} / {tuple(rs.shape)}")
        rows = rng.sample(sorted(ret), a.sample)
        with Pool(a.workers, initializer=_init_csr, initargs=(FP_RADIUS, vocab, loader.FEATURE_KIND)) as pool:
            got = dict(pool.map(_worker_row_nonzeros, [(k, ret[k]["smiles"]) for k in rows], chunksize=8))
        crow, col = rs.crow_indices(), rs.col_indices()
        bad = sum(sorted(col[crow[k]:crow[k + 1]].tolist()) != sorted(got[k]) for k in rows)
        check(f"[{fp_type}] sampled rankingset rows reproduce", bad == 0, f"{bad}/{len(rows)} differ")
        frag = {}
        for sp in SPLITS:
            t = pq.read_table(out / "arrow" / sp / loader.FRAGIDX_FILENAME).to_pydict()
            dup = len(t["idx"]) - len(set(t["idx"]))
            wrong = sum(idx[i]["split"] != sp for i in t["idx"])
            check(f"[{fp_type}] {sp} FragIdx: no duplicate idx, all in split", dup == 0 and wrong == 0, f"{dup} dup, {wrong} wrong split")
            frag.update(zip(t["idx"], t["cols"]))
        check(f"[{fp_type}] every index idx has a FragIdx row", frag.keys() == idx.keys(), f"{len(frag):,}")
        rows = rng.sample(sorted(idx), a.sample)
        with Pool(a.workers, initializer=_init_csr, initargs=(FP_RADIUS, vocab, loader.FEATURE_KIND)) as pool:
            got = dict(pool.map(_worker_row_nonzeros, [(i, idx[i]["smiles"]) for i in rows], chunksize=8))
        bad = sum(sorted(frag[i]) != sorted(got[i]) for i in rows)
        check(f"[{fp_type}] sampled FragIdx rows reproduce", bad == 0, f"{bad}/{len(rows)} differ")

    # 6. packed
    for sp in SPLITS:
        missing = [p.stem for p in (out / "arrow" / sp).glob("*.parquet") if not (out / "packed" / sp / p.stem).is_dir()]
        check(f"{sp} packed/ covers every arrow shard", not missing, ", ".join(missing))

    # 7. provenance
    a_src = pq.read_table(out / "nmr_sources.parquet").to_pandas()
    b_src = pq.read_table(ref / "nmr_sources.parquet").to_pandas()
    check("nmr_sources.parquet identical to reference", a_src.equals(b_src))
    vals = set(a_src[["hsqc", "c_nmr", "h_nmr"]].values.ravel())
    check("NMR sources only chnmr / mnova / absent", vals <= {"chnmr", "mnova", "absent"}, str(sorted(vals)))

    print(json.dumps({"failures": failures}), flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
