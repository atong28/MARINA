#!/usr/bin/env python3
"""Append journal-benchmark compounds missing from the retrieval bank.

Some journal compounds are in no retrieval set, so rank@k is undefined for them:
RankingSet.batched_rank counts how many stored rows beat the query's own gold
similarity and subtracts one for the gold row, which is only correct when the gold
row is actually stored. Appending the absent structures makes rank@k well-defined
over the whole journal benchmark without disturbing existing indices, since rows are
only added at the end.

Stored rows are L2-normalised binary fingerprints, so a new row is
build_mfp_for_smiles -> divide by its norm. Deduplicated by canonical SMILES: two
NPIDs can share a structure, and appending it twice would corrupt every rank that
retrieves it.

Source: scripts/benchmark/augment_rankingset.py. Differences: paths come from
config; novelty and dedup use canonicalize_smiles on the journal SMILES compared
against the retrieval bank (rather than the prepared pkl's precomputed fields); and
the index json maps ALL journal npids (present + appended), then asserts every one
resolves to a bank row so the "all journal present" guarantee is verifiable.

    DATASET_ROOT=/workspace pixi run python3 scripts/marina_db/build/70_retrieval_augment.py
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[3]))
from config import DATA_DATASET, RETRIEVAL_PKL, BENCH_JOURNAL_PREPARED, FP_TYPE, FP_OUT_DIM, FP_RADIUS
from src.modules.data.fp_utils import canonicalize_smiles
from src.modules.data.fp_loader import make_fp_loader


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp_type", default=FP_TYPE)
    ap.add_argument("--journal", default=str(BENCH_JOURNAL_PREPARED))
    ap.add_argument("--out_dim", type=int, default=FP_OUT_DIM)
    a = ap.parse_args()

    vocab_dir = Path(DATA_DATASET) / a.fp_type
    src = vocab_dir / "rankingset.pt"
    dst = vocab_dir / "rankingset_aug.pt"
    idx_out = vocab_dir / "journal_aug_index.json"

    journal = pickle.load(open(a.journal, "rb"))
    retrieval = pickle.load(open(RETRIEVAL_PKL, "rb"))
    # retrieval SMILES are already canonical (same fixed-point function); the rankingset
    # row index equals the retrieval key, so this doubles as canon -> bank-row.
    retrieval_idx = {entry["smiles"]: key for key, entry in retrieval.items()}

    fp_loader = make_fp_loader(a.fp_type, entropy_out_dim=a.out_dim,
                               max_radius=FP_RADIUS,
                               retrieval_path=str(RETRIEVAL_PKL))

    store = torch.load(src, map_location="cpu")
    n_before, dim = store.shape
    print(f"[{a.fp_type}] loaded {src}: {n_before} x {dim}")

    # Split journal into already-present (map to existing row) and novel-by-canon.
    assigned, novel = {}, {}
    for npid, entry in journal.items():
        # fp_utils.canonicalize_smiles raises on invalid/empty (its CSR workers can't take a
        # None), so guard by exception rather than a None check.
        try:
            canon = canonicalize_smiles(entry["smiles"])
        except ValueError:
            print(f"[{a.fp_type}] WARNING unparseable SMILES, skipping npid {npid}")
            continue
        if canon in retrieval_idx:
            assigned[npid] = retrieval_idx[canon]
        else:
            novel.setdefault(canon, []).append(npid)
    print(f"[{a.fp_type}] {len(assigned)} already in bank; "
          f"{len(novel)} distinct novel structures "
          f"({sum(len(v) for v in novel.values())} npids)")

    rows = []
    for offset, (smiles, npids) in enumerate(sorted(novel.items())):
        fp = fp_loader.build_mfp_for_smiles(smiles)
        norm = torch.norm(fp)
        if float(norm) == 0.0:
            # All-zero row gives cosine 0 to every query; skip rather than rank it last.
            print(f"[{a.fp_type}] WARNING zero fingerprint, skipping: {smiles[:60]}")
            continue
        rows.append(fp / norm)
        for npid in npids:
            assigned[npid] = n_before + offset

    if rows:
        added = torch.stack(rows).to_sparse_csr()
        store = torch.cat([store.to_sparse_coo(), added.to_sparse_coo()], dim=0).to_sparse_csr()

    torch.save(store, dst)
    json.dump({"fp_type": a.fp_type, "n_before": n_before, "n_after": store.shape[0],
               "npid_to_idx": assigned}, open(idx_out, "w"), indent=1)
    print(f"[{a.fp_type}] wrote {dst}: {store.shape[0]} x {store.shape[1]} "
          f"(+{store.shape[0] - n_before})")
    print(f"[{a.fp_type}] wrote {idx_out}")

    # Guarantee: every journal compound resolves to a bank row.
    missing = [npid for npid in journal if npid not in assigned]
    print(f"[{a.fp_type}] journal compounds present in bank: "
          f"{len(assigned)}/{len(journal)}")
    if missing:
        raise SystemExit(f"[{a.fp_type}] FAIL: {len(missing)} journal compounds "
                         f"not present in bank: {missing[:10]}")


if __name__ == "__main__":
    main()
