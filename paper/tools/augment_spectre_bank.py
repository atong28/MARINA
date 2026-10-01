#!/usr/bin/env python3
"""Build a SPECTRE retrieval dir for one SPECTRE-clean journal subset (spectre_comparison table).

The deployed SPECTRE bundle's retrieval set does not contain most clean-subset compounds (they are absent from
SPECTRE's train/val by construction), so dereplication needs them appended. Only structures NOT already in the
bundle's retrieval list are appended — appending a structure that is already there would add an FP-identical
twin that strict ranking counts against the gold. retrieval.pkl is extended in the same order so bank rows stay
aligned with SMILES.

  DATASET_ROOT=/tmp PYTHONPATH=. pixi run python paper/tools/augment_spectre_bank.py \
      --bundle ~/Deployments/SPECTRE-web/checkpoints/spectre_flexible \
      --journal <W>/bench/clean_test/benchmark-journal.pkl --out <W>/data/SPECTRE-clean-test
"""
import argparse
import json
import os
import pickle
import shutil

import torch

from src.modules.data.fp_loader import FP_LOADERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp_type", default="RankingEntropy")
    ap.add_argument("--out_dim", type=int, default=16384)
    a = ap.parse_args()

    retr_path = os.path.join(a.bundle, "retrieval.pkl")
    retrieval = pickle.load(open(retr_path, "rb"))
    present = {r["smiles"] for r in retrieval}
    journal = pickle.load(open(a.journal, "rb"))
    queries = list(dict.fromkeys(v["smiles"] for v in journal.values()))
    add = [s for s in queries if s not in present]
    print(f"bank rows {len(retrieval)}; subset structures {len(queries)}; already in bank {len(queries) - len(add)}; "
          f"appending {len(add)}")

    # the bundle's own feature map (bitinfo_to_idx.pkl) — never rebuild the vocabulary, columns must match the model
    mapping = os.path.join(a.bundle, a.fp_type, "bitinfo_to_idx.pkl")
    assert os.path.exists(mapping), f"missing {mapping}"
    fp = FP_LOADERS[a.fp_type](dataset_root=a.bundle, retrieval_path=retr_path)
    fp.setup(a.out_dim, 10, fp_type=a.fp_type, retrieval_path=retr_path)
    assert fp.out_dim == a.out_dim, fp.out_dim
    old = torch.load(os.path.join(a.bundle, a.fp_type, "rankingset.pt"), weights_only=False).cpu()
    crow, col, val = old.crow_indices().long(), old.col_indices().long(), old.values().float()
    # alignment check: structures already in the bank should rebuild to their stored row's on-bits. Known benign
    # difference: the bank was built with an older RDKit, which writes some fragment SMILES differently ('COc' vs
    # 'cOC'), moving one bit to a different column; anything beyond a few bits means the feature map is wrong.
    row_of = {r["smiles"]: i for i, r in enumerate(retrieval)}
    diffs = []
    for s in [q for q in queries if q in row_of]:
        i = row_of[s]
        stored = set(col[crow[i]:crow[i + 1]].tolist())
        built = set(fp.build_mfp_for_smiles(s).nonzero(as_tuple=False).flatten().tolist())
        diffs.append(len(stored ^ built))
    print(f"alignment check: {sum(d == 0 for d in diffs)}/{len(diffs)} in-bank structures rebuild exactly; "
          f"symmetric bit differences {diffs}")
    assert max(diffs, default=0) <= 4, "feature map does not reproduce the bank rows"
    running, extra_crow, new_cols, new_vals = int(crow[-1]), [], [], []
    for smi in add:
        mfp = fp.build_mfp_for_smiles(smi).float()
        n = torch.linalg.norm(mfp)
        mfp = mfp / n if n > 0 else mfp
        idx = mfp.nonzero(as_tuple=False).flatten().long()
        new_cols.append(idx)
        new_vals.append(mfp[idx])
        running += idx.numel()
        extra_crow.append(running)
    aug = torch.sparse_csr_tensor(torch.cat([crow, torch.tensor(extra_crow, dtype=torch.long)]),
                                  torch.cat([col] + new_cols), torch.cat([val] + new_vals),
                                  size=(old.shape[0] + len(add), old.shape[1]))
    os.makedirs(os.path.join(a.out, a.fp_type), exist_ok=True)
    torch.save(aug, os.path.join(a.out, a.fp_type, "rankingset.pt"))
    shutil.copy(os.path.join(a.bundle, a.fp_type, "bitinfo_to_idx.pkl"), os.path.join(a.out, a.fp_type))
    pickle.dump(retrieval + [{"smiles": s} for s in add], open(os.path.join(a.out, "retrieval.pkl"), "wb"))
    json.dump({"bundle": os.path.realpath(a.bundle), "journal": os.path.realpath(a.journal),
               "bank_rows_before": len(retrieval), "subset_structures": len(queries),
               "already_present": len(queries) - len(add), "appended": len(add), "bank_rows_after": aug.shape[0],
               "in_bank_rebuild_bit_differences": diffs},
              open(os.path.join(a.out, "augment_summary.json"), "w"), indent=2)
    print(f"wrote {a.out}: {aug.shape[0]} rows")


if __name__ == "__main__":
    main()
