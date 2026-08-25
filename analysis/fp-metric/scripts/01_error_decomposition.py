#!/usr/bin/env python3
"""Split MARINA's fingerprint error into "same substructure, wrong Morgan context" and
"wrong chemistry".

    pixi run python3 scripts/01_error_decomposition.py

Three views, weakest to strongest:

  (1) Exact L1 decomposition, threshold-free. For a group G of columns describing the
      same substructure, with residual r_i = p_i - x_i,

          total_G   = sum_i |r_i|
          between_G = |sum_i r_i|          <- wrong amount of this chemistry
          within_G  = total_G - between_G  <- right amount, wrong slots   (>= 0)

      Non-negative by the triangle inequality; zero iff every residual in the group
      shares a sign, i.e. no slot swapping. `within` is exactly the error annihilated
      by the group-merge projection G^T, so within/total upper-bounds what any
      same-substructure kernel can recover.

  (2) Hard decisions at top-k (k = the molecule's true on-bit count -- calibration-free,
      no threshold to choose). Every false positive is one of:
        A1  in-vocab slot error   -- fragment present, a sibling bit carries it in the target
        A2  target-coding artifact -- fragment present, but no sibling is on: the entropy
                                      selection dropped the context this molecule uses, so
                                      the *label* is wrong, not the model
        B   chemistry error        -- fragment genuinely absent
      A2 matters because G^T cannot fix it (the merged target still reads 0) but the
      retrained substructure fingerprint can -- it keys on the fragment alone. So A2 is
      the margin by which a metric-only experiment understates the substructure FP.
      False negatives get the mirror split.

  (3) Retrieval-weighted: cos(p, x) vs cos(G^T p, G^T x), the same quantity in the units
      the ranker actually uses.
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REDUNDANCY = os.path.join(os.path.dirname(HERE), "fp-redundancy")

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default=os.path.join(REDUNDANCY, "results", "preds.npz"))
ap.add_argument("--groups", default=os.path.join(HERE, "results", "groups.npz"))
ap.add_argument("--out", default=os.path.join(HERE, "results", "error_decomposition.json"))
cli = ap.parse_args()

VARIANTS = ("strict", "atom")
t0 = time.time()

z = np.load(cli.preds, allow_pickle=True)
P = z["probs"].astype(np.float32)
M, B = P.shape
fp_idx, fp_ptr = z["fp_idx"], z["fp_ptr"]
X = sp.csr_matrix((np.ones(len(fp_idx), dtype=np.float32), fp_idx, fp_ptr), shape=(M, B))
g = np.load(cli.groups, allow_pickle=True)
assert int(g["n_bits"]) == B and np.array_equal(g["mol_idx"], z["mol_idx"]), \
    "groups.npz was built against different predictions"
radii = g["radii"]
print(f"{M} molecules x {B} bits, {X.nnz} true on-bits "
      f"({X.nnz/M:.1f}/molecule), ckpt={z['ckpt']}")

R = P - X.toarray()                                  # residual, (M, B)
abs_R = np.abs(R)
total = abs_R.sum(axis=1)
r0_share = abs_R[:, radii == 0].sum() / abs_R.sum()
k_true = np.diff(fp_ptr).astype(np.int64)

report = {"n_molecules": int(M), "n_bits": int(B), "ckpt": str(z["ckpt"]),
          "mean_true_on_bits": float(k_true.mean()),
          "radius0_share_of_abs_error": float(r0_share), "variants": {}}
print(f"radius-0 bits carry {r0_share:.4%} of total |error| "
      f"(they are the only bits the two grouping variants disagree on)\n")

for variant in VARIANTS:
    gid = g[f"gid_{variant}"]
    Gn = int(g[f"n_groups_{variant}"])
    Gmat = sp.csr_matrix((np.ones(B, dtype=np.float32), (np.arange(B), gid)), shape=(B, Gn))

    # ---- (1) exact L1 decomposition -------------------------------------------------
    between = np.abs(R @ Gmat).sum(axis=1)
    within = total - between
    assert within.min() > -1e-2, f"within went negative ({within.min():.3e})"
    within = np.maximum(within, 0.0)

    # ---- (3) retrieval-weighted -----------------------------------------------------
    def _cos(A, Bm):
        num = (A * Bm).sum(axis=1)
        return num / np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(Bm, axis=1), 1e-12)

    Xd = X.toarray()
    cos_bit = _cos(P, Xd)
    cos_grp = _cos(P @ Gmat, Xd @ Gmat)

    # ---- (2) hard decisions at top-k ------------------------------------------------
    pres_idx, pres_ptr = g[f"present_{variant}_idx"], g[f"present_{variant}_ptr"]
    cnt = dict(A1=0, A2=0, B=0, fn_slot=0, fn_missed=0, fp=0, fn=0, sib_not_present=0)
    grp_true = np.zeros(Gn, dtype=bool)
    grp_pred = np.zeros(Gn, dtype=bool)
    grp_pres = np.zeros(Gn, dtype=bool)
    for m in range(M):
        k = int(k_true[m])
        if k == 0:
            continue
        truth = fp_idx[fp_ptr[m]:fp_ptr[m + 1]].astype(np.int64)
        pred = np.argpartition(-P[m], k - 1)[:k]
        tset = np.zeros(B, dtype=bool); tset[truth] = True
        pset = np.zeros(B, dtype=bool); pset[pred] = True
        fp_bits = pred[~tset[pred]]
        fn_bits = truth[~pset[truth]]

        gt, gp = gid[truth], gid[pred]
        gpres = pres_idx[pres_ptr[m]:pres_ptr[m + 1]]
        grp_true[gt] = True; grp_pred[gp] = True; grp_pres[gpres] = True

        gfp = gid[fp_bits]
        sib = grp_true[gfp]                 # a sibling bit carries this fragment in the target
        chem = grp_pres[gfp]                # the fragment occurs somewhere in the molecule
        cnt["A1"] += int(sib.sum())
        cnt["A2"] += int((chem & ~sib).sum())
        cnt["B"] += int((~chem & ~sib).sum())
        cnt["sib_not_present"] += int((sib & ~chem).sum())   # should be 0 by construction
        cnt["fp"] += len(fp_bits)

        gfn = gid[fn_bits]
        slot = grp_pred[gfn]                # model put this fragment in a different slot
        cnt["fn_slot"] += int(slot.sum())
        cnt["fn_missed"] += int((~slot).sum())
        cnt["fn"] += len(fn_bits)

        grp_true[gt] = False; grp_pred[gp] = False; grp_pres[gpres] = False

    frac = within.sum() / total.sum()
    report["variants"][variant] = {
        "n_groups": Gn,
        "within_frac_of_L1": float(frac),
        "within_frac_per_molecule_mean": float((within / np.maximum(total, 1e-12)).mean()),
        "within_frac_per_molecule_p10_p50_p90":
            [float(v) for v in np.percentile(within / np.maximum(total, 1e-12), [10, 50, 90])],
        "mean_total_L1": float(total.mean()),
        "mean_between_L1": float(between.mean()),
        "mean_within_L1": float(within.mean()),
        "cos_bit_level": float(cos_bit.mean()),
        "cos_group_level": float(cos_grp.mean()),
        "counts": cnt,
    }

    print(f"===== variant: {variant}  ({Gn} groups) =====")
    print(f"(1) L1 decomposition, threshold-free")
    print(f"      total   {total.mean():8.3f} /molecule")
    print(f"      between {between.mean():8.3f}   ({1-frac:6.2%})  wrong chemistry")
    print(f"      within  {within.mean():8.3f}   ({frac:6.2%})  right chemistry, wrong slot")
    p10, p50, p90 = np.percentile(within / np.maximum(total, 1e-12), [10, 50, 90])
    print(f"      per-molecule within-fraction  p10 {p10:.3f}  p50 {p50:.3f}  p90 {p90:.3f}")
    print(f"(2) hard decisions at top-k (k = true on-bit count)")
    fpn = max(cnt["fp"], 1); fnn = max(cnt["fn"], 1)
    print(f"      false positives  {cnt['fp']:6d}  ({cnt['fp']/M:.1f}/molecule)")
    print(f"        A1 in-vocab slot error      {cnt['A1']:6d}  {cnt['A1']/fpn:6.2%}")
    print(f"        A2 target-coding artifact   {cnt['A2']:6d}  {cnt['A2']/fpn:6.2%}")
    print(f"        B  chemistry error          {cnt['B']:6d}  {cnt['B']/fpn:6.2%}")
    print(f"      false negatives  {cnt['fn']:6d}")
    print(f"        slot-swapped (sibling predicted) {cnt['fn_slot']:6d}  {cnt['fn_slot']/fnn:6.2%}")
    print(f"        missed chemistry                 {cnt['fn_missed']:6d}  {cnt['fn_missed']/fnn:6.2%}")
    if cnt["sib_not_present"]:
        print(f"      !! {cnt['sib_not_present']} FP bits had a sibling on in the target but the "
              f"fragment absent from the RDKit enumeration -- SMILES/target mismatch")
    print(f"(3) retrieval-weighted")
    print(f"      cos(p, x)         {cos_bit.mean():.4f}")
    print(f"      cos(G'p, G'x)     {cos_grp.mean():.4f}   (+{cos_grp.mean()-cos_bit.mean():.4f})\n")

with open(cli.out, "w") as f:
    json.dump(report, f, indent=2)
print(f"wrote {cli.out}  ({time.time()-t0:.1f}s)")
