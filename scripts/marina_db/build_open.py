#!/usr/bin/env python3
"""build_open.py -- derive MARINA-DB-OPEN from a built MARINA-DB + CH-NMR-NP.

MARINA-DB-OPEN is MARINA-DB with NMR taken only from sources we may redistribute: CH-NMR-NP
(JEOL experimental natural-product NMR, NONCOMMERCIAL licence -- the release must be NC and
attribute CH-NMR-NP) first, the Mnova simulations second. The SPECTRE corpus (ACD/Labs HSQC,
CASPRE/PROSPERE 1D, and SPECTRE's own copy of CH-NMR-NP) is never consulted. MS/MS of the
MARINA-DB molecules is untouched (same SOURCE_PRIORITY as MARINA-DB). The earlier Mnova-only
MARINA-DB-OPEN (2026-09-16) is reproducible from MARINA 4913309.

  A. MARINA-DB molecules keep idx / split / FP / MS. Per modality (HSQC, 13C, 1H independently)
     the spectrum is CH-NMR-NP's when the molecule has a confirmed record carrying it, else
     Mnova (stage-6 cache mapping.json). Quality guards against the Mnova 13C prediction:
       coverage  CH-NMR-NP 13C covers < MIN_C_COVERAGE of the symmetry-distinct carbons
                 -> 13C and HSQC fall back to Mnova (1H kept)
       mae       optimal-assignment 13C MAE in (--mae_max, --mae_trust_above] on an UNCHARGED
                 molecule -> the record/structure pairing is not trusted: HSQC, 13C AND 1H all
                 fall back. Above --mae_trust_above, or on any molecule carrying a formal charge
                 (N-oxides, [n+] aromatics), Mnova is the broken side and CH-NMR-NP is kept.
     Per-idx provenance goes to nmr_sources.parquet at the dataset root (idx, hsqc, c_nmr,
     h_nmr in {"chnmr","mnova","absent"}, chnmr_id = the chosen confirmed record linked to the
     molecule whether or not any modality used it, null if none; chnmr_solvent = that record's
     free-text solvent, read by the solvent-jitter augmentation; is_new), so index.pkl rows keep
     MARINA-DB's schema.
  B. retrieval: confirmed structures not yet in retrieval.pkl are appended (idx max+1.., no
     filter); metadata.json gains `ch_nmr_np` (always a LIST of {id,no,chs,name,reference}, one
     per confirmed record sharing the structure) on every confirmed-structure row.
  C. index: confirmed structures not in the MARINA-DB index that pass stage 6's filters
     (exact MW <= MW_MAX_EXACT, >= MIN_HEAVY_ATOMS heavy atoms) and carry at least one spectrum
     become new rows (idx max+1.., sorted SMILES order). NMR from CH-NMR-NP only (coverage guard
     only; no Mnova), MS/MS from ICEBERG chunk dirs (--ms_pos -> mass_spec, --ms_neg ->
     mass_spec_neg; predict_parse.py schema, peaks stored verbatim exactly as stage 6 stores
     data/raw/ms_predictions*). Existing splits are frozen; new rows get stage 7's forced rules
     (journal -> test, SPECTRE partition) and the free pool fills the GLOBAL 90/5/5 targets.
  D. fingerprints: every FP family's vocab (bitinfo_to_idx.pkl) is copied byte-identical and
     NEVER recomputed (the vocab is entropy over the ORIGINAL retrieval set). New retrieval rows
     get rankingset rows and new index rows FragIdx rows under that fixed vocab
     (fp_utils._worker_row_nonzeros, as build_rankingset_csr / build_fragidx_parquets do).
     No count_*.pkl is written or read.
  E. stage 11 per-peak collapse, stage 13 pack.

    cd ~/Workspace/MARINA
    DATASET_ROOT=$PWD/data/dataset pixi run python scripts/marina_db/build_open.py \
        --structures ../CH-NMR-NP/<confirmed>.tsv --ms_pos <iceberg pos dir> --ms_neg <iceberg neg dir>
    DATASET_ROOT=$PWD/data/dataset pixi run python scripts/marina_db/build/12_verify.py \
        --dataset_root ../Datasets/MARINA-DB-OPEN
    DATASET_ROOT=$PWD/data/dataset pixi run python scripts/marina_db/check_open_identity.py \
        --out ../Datasets/MARINA-DB-OPEN
"""
import argparse
import glob
import importlib
import json
import math
import pickle
import random
import shutil
import sys
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))                        # config, lib
sys.path.insert(0, str(_HERE.parent / "build"))              # stage modules
sys.path.insert(0, str(_HERE.parents[1] / "dataset"))        # pack_arrow
sys.path.insert(0, str(_HERE.parents[2]))                    # repo root
from config import (DATA_DATASET, DATA_CLEANED, MARINA_DATA_ROOT, SOURCE_PRIORITY, BENCH_JOURNAL,
                    SPECTRE_SPLITS_PKL, MW_MAX_EXACT, MIN_HEAVY_ATOMS, FP_RADIUS, PEAK_COLLAPSE_TOL, SEED)
from lib import chnmr
from src.modules.data.smiles import canonicalize_smiles
from src.modules.data.formula import formula_to_vector
from src.modules.data.fp_loader import FP_LOADERS
from src.modules.data.fp_utils import _init_csr, _worker_row_nonzeros

assemble = importlib.import_module("8_assemble_arrow")
splits = importlib.import_module("7_splits")
collapse = importlib.import_module("11_collapse_peaks")
from pack_arrow import pack_root

RDLogger.DisableLog("rdApp.*")

SPLITS = ("train", "val", "test")
NMR = ("hsqc", "c_nmr", "h_nmr")
MODS = NMR + ("mass_spec", "mass_spec_neg")
OPEN_PRIORITY = {**SOURCE_PRIORITY, **{m: ["mnova"] for m in NMR}}
MIN_C_COVERAGE = 0.5
MAE_REPORT_THRESHOLDS = (2, 3, 4, 5, 6, 8, 10, 15, 20)


def _canon(smiles):
    try:
        return canonicalize_smiles(smiles)
    except ValueError:
        return None


def _canon_all(smiles, workers):
    with Pool(workers) as pool:
        return pool.map(_canon, smiles, chunksize=500)


def _collapse(values):
    return collapse.collapse(values, PEAK_COLLAPSE_TOL)


def load_iceberg(d, workers):
    """{canonical SMILES: peaks} from a dir of iceberg_*.json chunks (stage 6: last write wins)."""
    if not d:
        return {}
    recs = [r for f in sorted(glob.glob(str(Path(d) / "iceberg_*.json"))) for r in json.load(open(f))]
    return {c: r["peaks"] for c, r in zip(_canon_all([r["SMILES"] for r in recs], workers), recs) if c}


def chnmr_choice(info, mnova_c, mae_max, trust_above):
    """Per NMR modality: CH-NMR-NP spectrum or None, with the reason when None, plus the 13C
    MAE vs Mnova and its guard bucket. mnova_c=None skips the MAE guard (new molecules:
    coverage guard only)."""
    mae = chnmr.assignment_mae(info["c_nmr"], mnova_c, _collapse) if (info["c_nmr"] and mnova_c) else None
    bucket = None
    if mae is not None and mae_max is not None and mae > mae_max:
        bucket = ("kept_charged" if info["charged"] else
                  "kept_mae_gt_trust" if mae > trust_above else "fell_back")
    use, why = {}, {}
    for mod in NMR:
        if bucket == "fell_back":
            why[mod] = "mae"
        elif mod == "hsqc" and info["hsqc_status"] != "ok":
            why[mod] = f"hsqc_{info['hsqc_status']}"
        elif info[mod] is None:
            why[mod] = "absent_in_chnmr"
        elif mod != "h_nmr" and info["coverage"] < MIN_C_COVERAGE:
            why[mod] = "low_c_coverage"
        use[mod] = None if mod in why else info[mod]
    return use, why, mae, bucket


def pctl(v, ps=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100)):
    return {str(p): round(float(x), 3) for p, x in zip(ps, np.percentile(v, ps))} if len(v) else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DATA_DATASET), help="built MARINA-DB root")
    ap.add_argument("--out", default=str(MARINA_DATA_ROOT / "Datasets" / "MARINA-DB-OPEN"))
    ap.add_argument("--structures", required=True, help="TSV chnmr_id<TAB>smiles of CONFIRMED structures")
    ap.add_argument("--records", default=str(MARINA_DATA_ROOT / "CH-NMR-NP" / "records.jsonl"))
    ap.add_argument("--ms_pos", default=None, help="dir of ICEBERG [M+H]+ iceberg_*.json (new molecules)")
    ap.add_argument("--ms_neg", default=None, help="dir of ICEBERG [M-H]- iceberg_*.json (new molecules)")
    ap.add_argument("--mae_max", type=float, default=5.0,
                    help="HSQC/13C/1H fall back to Mnova above this CH-NMR-NP-vs-Mnova 13C MAE (ppm)")
    ap.add_argument("--mae_trust_above", type=float, default=15.0,
                    help="...unless the MAE exceeds this (Mnova is the broken side); charged molecules always kept")
    ap.add_argument("--workers", type=int, default=cpu_count())
    a = ap.parse_args()
    t0 = time.time()
    src, out = Path(a.src), Path(a.out)
    work = out.parent / (out.name + ".work")
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    summary = {"src": str(src), "out": str(out), "structures": a.structures, "records": a.records,
               "ms_pos": a.ms_pos, "ms_neg": a.ms_neg, "mae_max": a.mae_max,
               "mae_trust_above": a.mae_trust_above,
               "min_c_coverage": MIN_C_COVERAGE, "fp_radius": FP_RADIUS}

    index = pickle.load(open(src / "index.pkl", "rb"))
    retrieval = pickle.load(open(src / "retrieval.pkl", "rb"))
    assert sorted(retrieval) == list(range(len(retrieval))), "retrieval idx must be 0..N-1"
    idx_of = {e["smiles"]: i for i, e in index.items()}
    print(f"MARINA-DB index {len(index):,}, retrieval {len(retrieval):,}", flush=True)

    # ---- confirmed CH-NMR-NP structures -> one chosen record per canonical structure ----
    structures = chnmr.load_structures(a.structures)
    canon = _canon_all([s for _, s in structures], a.workers)
    by_canon = chnmr.group_confirmed(structures, canon)
    recs = chnmr.load_records(a.records, [i for ids in by_canon.values() for i in ids])
    unknown = sorted({i for ids in by_canon.values() for i in ids} - set(recs))
    by_canon = {c: [recs[i] for i in ids if i in recs] for c, ids in by_canon.items()}
    by_canon = {c: rs for c, rs in by_canon.items() if rs}
    tsv_smiles = dict(structures)
    info = {}
    for c, rs in by_canon.items():
        rec = chnmr.choose_record(rs)
        hsqc, status, cn, hn = chnmr.to_spectra(rec)
        mol = Chem.MolFromSmiles(c)
        info[c] = {"rec": rec, "hsqc": hsqc, "hsqc_status": status, "c_nmr": cn, "h_nmr": hn,
                   "coverage": chnmr.c_coverage(cn, mol),
                   "charged": any(at.GetFormalCharge() for at in mol.GetAtoms())}
    summary["chnmr"] = {
        "tsv_rows": len(structures), "tsv_canon_failed": sum(c is None for c in canon),
        "tsv_ids_not_in_records": len(unknown), "confirmed_records": sum(len(r) for r in by_canon.values()),
        "distinct_structures": len(by_canon), "structures_with_multiple_records":
            sum(len(r) > 1 for r in by_canon.values()),
        "structures_in_marinadb_index": sum(c in idx_of for c in by_canon),
        "hsqc_status": dict(Counter(v["hsqc_status"] for v in info.values())),
    }
    print(f"CH-NMR-NP: {json.dumps(summary['chnmr'])}  {time.time()-t0:.0f}s", flush=True)

    # ---- A. existing molecules: CH-NMR-NP first, Mnova fallback ----
    mapping = json.load(open(DATA_CLEANED / "mapping.json"))
    print(f"mapping: {len(mapping):,} SMILES  {time.time()-t0:.0f}s", flush=True)
    handles = {s: open(work / f"{s}.jsonl", "w", encoding="utf-8") for s in SPLITS}
    source = Counter()          # (mod, source, split)
    fallback = Counter()        # (mod, reason)
    maes, maes_cov_ok = [], {False: [], True: []}     # coverage-ok MAEs keyed by charged
    buckets = Counter()
    provenance = {}                                  # idx -> (hsqc, c_nmr, h_nmr sources, chnmr_id, chnmr_solvent)
    for e in tqdm(index.values(), desc="Materializing existing rows"):
        m = mapping.get(e["smiles"], {})
        row = {"idx": e["idx"], "smiles": e["smiles"]}
        for mod, order in OPEN_PRIORITY.items():
            row[mod] = assemble._priority_access(m.get(mod, {}), order)
        src_of = {mod: ("mnova" if row[mod] else "absent") for mod in NMR}
        chnmr_id = chnmr_solvent = None
        if e["smiles"] in info:
            v = info[e["smiles"]]
            chnmr_id, chnmr_solvent = v["rec"]["id"], v["rec"]["solvent"]
            use, why, mae, bucket = chnmr_choice(v, row["c_nmr"], a.mae_max, a.mae_trust_above)
            if bucket:
                buckets[bucket] += 1
            if mae is not None:
                maes.append(mae)
                if v["coverage"] >= MIN_C_COVERAGE:
                    maes_cov_ok[v["charged"]].append(mae)
            for mod in NMR:
                if use[mod] is not None:
                    row[mod], src_of[mod] = use[mod], "chnmr"
                else:
                    fallback[(mod, why[mod], src_of[mod])] += 1
        for mod in NMR:
            e[f"has_{mod}"] = row[mod] != []
            source[(mod, src_of[mod], e["split"])] += 1
        provenance[e["idx"]] = (src_of["hsqc"], src_of["c_nmr"], src_of["h_nmr"], chnmr_id, chnmr_solvent)
        handles[e["split"]].write(json.dumps(row) + "\n")
    del mapping

    summary["existing"] = {
        "modality_source_split": {mod: {s: {sp: source[(mod, s, sp)] for sp in SPLITS}
                                        for s in ("chnmr", "mnova", "absent")} for mod in NMR},
        "chnmr_fallback_by_reason": {mod: {f"{r} -> {s}": n for (m2, r, s), n in sorted(fallback.items())
                                           if m2 == mod} for mod in NMR},
        "c13_mae_vs_mnova": {
            "n": len(maes), "percentiles": pctl(maes),
            "coverage_ok_by_charge": {
                name: {"n": len(v), "percentiles": pctl(v),
                       "n_above": {str(t): int(np.sum(np.array(v) > t)) for t in MAE_REPORT_THRESHOLDS}}
                for name, v in (("uncharged", maes_cov_ok[False]), ("charged", maes_cov_ok[True]))},
            "guard_buckets_mae_gt_mae_max": {k: buckets[k] for k in
                                             ("kept_charged", "kept_mae_gt_trust", "fell_back")}},
        "c13_coverage_percentiles": pctl([info[c]["coverage"] for c in info if c in idx_of]),
    }

    # ---- C. new index rows ----
    ms = {"mass_spec": load_iceberg(a.ms_pos, a.workers), "mass_spec_neg": load_iceberg(a.ms_neg, a.workers)}
    next_idx = max(index) + 1
    new_index, new_rows, dropped, new_why = {}, {}, Counter(), Counter()
    for c in sorted(c for c in info if c not in idx_of):
        mol = Chem.MolFromSmiles(c)
        mw = rdMolDescriptors.CalcExactMolWt(mol)
        if mw > MW_MAX_EXACT:
            dropped["mw_gt_max"] += 1
            continue
        if mol.GetNumHeavyAtoms() < MIN_HEAVY_ATOMS:
            dropped["heavy_atoms_lt_min"] += 1
            continue
        use, why, _, _ = chnmr_choice(info[c], None, None, None)
        row = {**{mod: use[mod] or [] for mod in NMR}, **{mod: ms[mod].get(c, []) for mod in ms}}
        if not any(row[mod] for mod in MODS):
            dropped["no_spectrum"] += 1
            continue
        new_why.update((mod, r) for mod, r in why.items())
        formula = rdMolDescriptors.CalcMolFormula(mol)
        new_index[next_idx] = {
            "idx": next_idx, "smiles": c, "split": None,
            **{f"has_{mod}": row[mod] != [] for mod in MODS},
            "has_mw": True, "has_formula": True, "mw": mw, "formula": formula,
            "formula_vec": formula_to_vector(formula),
        }
        new_rows[next_idx] = row
        provenance[next_idx] = (*("chnmr" if row[m] else "absent" for m in NMR), info[c]["rec"]["id"],
                                info[c]["rec"]["solvent"])
        next_idx += 1

    # splits: stage 7's forced rules on new rows only; free pool fills the GLOBAL targets
    bench = splits.load_benchmark_smiles(BENCH_JOURNAL, a.workers)
    spec = splits.load_spectre_splits(SPECTRE_SPLITS_PKL, a.workers)
    assign, free, rule = {}, [], Counter()
    for i in sorted(new_index):
        c = new_index[i]["smiles"]
        if c in bench:
            assign[i] = "test"; rule["benchmark_test"] += 1
        elif c in spec["test"]:
            assign[i] = "test"; rule["spectre_test"] += 1
        elif c in spec["val"]:
            assign[i] = "val"; rule["spectre_val"] += 1
        elif c in spec["train"]:
            assign[i] = "train"; rule["spectre_train"] += 1
        else:
            free.append(i)
    existing_counts = Counter(e["split"] for e in index.values())
    forced = existing_counts + Counter(assign.values())
    targets, needs = splits.free_pool_allocation(len(index) + len(new_index), forced, len(free))
    random.Random(SEED).shuffle(free)
    pos = 0
    for s in SPLITS:
        for i in free[pos:pos + needs[s]]:
            assign[i] = s
        pos += needs[s]
    for i, e in new_index.items():
        e["split"] = assign[i]
        handles[e["split"]].write(json.dumps({"idx": i, "smiles": e["smiles"], **new_rows[i]}) + "\n")
    for h in handles.values():
        h.close()
    full_index = {**index, **new_index}
    pickle.dump(full_index, open(work / "index.pkl", "wb"))
    ids = sorted(provenance)
    pq.write_table(pa.table({
        "idx": pa.array(ids, pa.int64()),
        **{mod: pa.array([provenance[i][k] for i in ids], pa.string()) for k, mod in enumerate(NMR)},
        "chnmr_id": pa.array([provenance[i][3] for i in ids], pa.int64()),
        "chnmr_solvent": pa.array([provenance[i][4] for i in ids], pa.string()),
        "is_new": pa.array([i in new_index for i in ids], pa.bool_()),
    }), out / "nmr_sources.parquet")
    summary["nmr_sources"] = {
        "file": "nmr_sources.parquet", "rows": len(ids),
        "columns": "idx int64; hsqc/c_nmr/h_nmr string chnmr|mnova|absent; chnmr_id int64 = chosen "
                   "confirmed CH-NMR-NP record linked to the molecule (null if none; set even when "
                   "every modality fell back); chnmr_solvent string = that record's free-text solvent "
                   "(null if none); is_new bool (appended index row)",
        "with_chnmr_id": sum(p[3] is not None for p in provenance.values()),
        "any_modality_from_chnmr": sum("chnmr" in p[:3] for p in provenance.values())}

    new_split = Counter(e["split"] for e in new_index.values())
    summary["new_index_rows"] = {
        "candidates_not_in_index": sum(c not in idx_of for c in info),
        "filtered_out": dict(dropped), "added": len(new_index),
        "by_split": {s: new_split[s] for s in SPLITS},
        "forced_by_rule": dict(rule), "free_pool": len(free), "free_pool_allocation": needs,
        "global_targets": targets,
        "final_split_counts": dict(Counter(e["split"] for e in full_index.values())),
        "modality_present_by_split": {mod: {s: sum(e[f"has_{mod}"] for e in new_index.values() if e["split"] == s)
                                            for s in SPLITS} for mod in MODS},
        "chnmr_modality_missing_by_reason": {f"{m} {r}": n for (m, r), n in sorted(new_why.items())},
        "ms_coverage": {"iceberg_pos_records": len(ms["mass_spec"]), "iceberg_neg_records": len(ms["mass_spec_neg"]),
                        "pos_and_neg": sum(e["has_mass_spec"] and e["has_mass_spec_neg"] for e in new_index.values()),
                        "pos_only": sum(e["has_mass_spec"] and not e["has_mass_spec_neg"] for e in new_index.values()),
                        "neg_only": sum(e["has_mass_spec_neg"] and not e["has_mass_spec"] for e in new_index.values()),
                        "none": sum(not e["has_mass_spec"] and not e["has_mass_spec_neg"] for e in new_index.values())},
    }
    print(f"new index rows: {json.dumps(summary['new_index_rows'])}  {time.time()-t0:.0f}s", flush=True)

    # stage 8 arrow conversion (copies index.pkl from work; spectral parquets from the jsonl)
    assemble.convert_to_arrow(str(work), str(out))

    # ---- B. retrieval + metadata ----
    in_retrieval = {e["smiles"] for e in retrieval.values()}
    new_ret = sorted(c for c in info if c not in in_retrieval)
    full_retrieval = dict(retrieval)
    for k, c in enumerate(new_ret, start=len(retrieval)):
        full_retrieval[k] = {"smiles": c}
    pickle.dump(full_retrieval, open(out / "retrieval.pkl", "wb"))

    metadata = json.load(open(src / "metadata.json"))
    for k, e in full_retrieval.items():
        c = e["smiles"]
        if k >= len(retrieval):
            smi_3d = _canon_3d(tsv_smiles[info[c]["rec"]["id"]]) or c
            metadata[str(k)] = {"smiles": c, "canonical_2d_smiles": c, "canonical_3d_smiles": smi_3d,
                                "npmrd": None, "coconut": None, "lotus": None}
        if c in by_canon:
            metadata[str(k)]["ch_nmr_np"] = [chnmr.meta(r) for r in by_canon[c]]
    json.dump(metadata, open(out / "metadata.json", "w"))
    summary["retrieval"] = {"src_rows": len(retrieval), "added": len(new_ret), "rows": len(full_retrieval),
                            "metadata_rows_with_ch_nmr_np": sum(c in by_canon for c in
                                                                (e["smiles"] for e in full_retrieval.values()))}
    print(f"retrieval: {json.dumps(summary['retrieval'])}  {time.time()-t0:.0f}s", flush=True)
    del metadata

    # ---- D. fingerprints under the FIXED vocab ----
    todo = sorted(set(new_ret) | {e["smiles"] for e in new_index.values()})
    fp_types = [t for t in FP_LOADERS if (src / t / "bitinfo_to_idx.pkl").exists()]
    for fp_type in fp_types:
        loader = FP_LOADERS[fp_type]
        vocab = pickle.load(open(src / fp_type / "bitinfo_to_idx.pkl", "rb"))
        with Pool(a.workers, initializer=_init_csr, initargs=(FP_RADIUS, vocab, loader.FEATURE_KIND)) as pool:
            cols_of = {todo[k]: cols for k, cols in pool.map(_worker_row_nonzeros, list(enumerate(todo)), chunksize=8)}

        (out / fp_type).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / fp_type / "bitinfo_to_idx.pkl", out / fp_type / "bitinfo_to_idx.pkl")
        rs = torch.load(src / fp_type / "rankingset.pt", weights_only=True)
        assert rs.shape[0] == len(retrieval), f"{fp_type} rankingset rows {rs.shape[0]} != retrieval"
        crow, col, val = [rs.crow_indices()], [rs.col_indices()], [rs.values()]
        nnz = int(crow[0][-1])
        for c in new_ret:               # same row construction as fp_utils.build_rankingset_csr
            cols = cols_of[c]
            nnz += len(cols)
            crow.append(torch.tensor([nnz], dtype=torch.int64))
            col.append(torch.tensor(cols, dtype=torch.int64))
            val.append(torch.full((len(cols),), 1.0 / math.sqrt(len(cols)) if cols else 0.0, dtype=torch.float32))
        torch.save(torch.sparse_csr_tensor(torch.cat(crow), torch.cat(col), torch.cat(val),
                                           size=(len(full_retrieval), rs.shape[1])),
                   out / fp_type / "rankingset.pt")

        fname = loader.FRAGIDX_FILENAME
        for sp in SPLITS:
            tab = pq.read_table(src / "arrow" / sp / fname)
            rows = [(i, cols_of[e["smiles"]]) for i, e in sorted(new_index.items()) if e["split"] == sp]
            add = pa.table({"idx": pa.array([i for i, _ in rows], pa.int64()),
                            "cols": pa.array([c for _, c in rows], pa.list_(pa.int32()))}, schema=tab.schema)
            pq.write_table(pa.concat_tables([tab, add]), out / "arrow" / sp / fname)
        print(f"[{fp_type}] +{len(new_ret)} rankingset rows, +{len(new_index)} {fname} rows  "
              f"{time.time()-t0:.0f}s", flush=True)
    summary["fp_types"] = fp_types

    collapse.collapse_dataset(out)      # stage 11 (dataset only; journal untouched)
    pack_root(str(out), force=True)     # stage 13
    shutil.rmtree(work)
    summary["seconds"] = int(time.time() - t0)
    json.dump(summary, open(out / "open_build_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    print(f"MARINA-DB-OPEN built at {out}; now run 12_verify.py --dataset_root {out} and "
          f"check_open_identity.py --out {out}", flush=True)


def _canon_3d(smiles):
    try:
        return canonicalize_smiles(smiles, keep_stereo=True)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
