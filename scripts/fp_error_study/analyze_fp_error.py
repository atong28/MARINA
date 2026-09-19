"""
Analyze MARINA's FP error distribution in FRAGMENT space, to calibrate the Moonshot
corruption augmenter.

Consumes the dump from src/fp_error_dump.py and the FP vocab (bitinfo_to_idx.pkl).
Maps every column -> (fragment, multiplicity bucket), then for each molecule compares
MARINA's thresholded prediction to the true fingerprint, per fragment.

Key outputs (JSON + npz + printed summary):
  - threshold (tau) sweep: bit-level recall/precision/Tanimoto, count-L1
  - at the chosen tau (max mean-Tanimoto), the fragment-level error model:
      * P(recall), mean Delta-k, P(exact count) conditioned on true count k_true
      * false-positive fragment rate per molecule and their count distribution
      * off-prefix (non-thermometer) rate among predicted-present fragments
      * Delta-k histogram
"""
import os
import json
import pickle
import argparse
from collections import defaultdict

import numpy as np


def load(dump_dir, vocab_path):
    pred = np.load(os.path.join(dump_dir, "pred_probs.npy"))  # (N, D) fp16
    with open(os.path.join(dump_dir, "true_cols.pkl"), "rb") as f:
        true_cols = pickle.load(f)
    with open(vocab_path, "rb") as f:
        b2i = pickle.load(f)  # {(frag,bucket): col}
    D = pred.shape[1]
    col2key = [None] * D
    for k, c in b2i.items():
        col2key[c] = k
    return pred, true_cols, col2key


def build_maps(col2key):
    frags = sorted({k[0] for k in col2key})
    frag2id = {f: i for i, f in enumerate(frags)}
    D = len(col2key)
    col_fragid = np.empty(D, dtype=np.int32)
    col_bucket = np.empty(D, dtype=np.int32)
    frag_buckets = defaultdict(list)  # fid -> [(bucket, col)]
    for c, (frag, bucket) in enumerate(col2key):
        fid = frag2id[frag]
        col_fragid[c] = fid
        col_bucket[c] = bucket
        frag_buckets[fid].append((bucket, c))
    frag_surv = {fid: np.array(sorted(b for b, _ in v), dtype=np.int32)
                 for fid, v in frag_buckets.items()}
    return frags, col_fragid, col_bucket, frag_surv


def kmax_per_frag(cols, col_fragid, col_bucket, nfrags):
    """Max lit bucket per fragment given active column indices (0 => absent)."""
    k = np.zeros(nfrags, dtype=np.int32)
    if len(cols):
        np.maximum.at(k, col_fragid[cols], col_bucket[cols])
    return k


def tau_sweep(pred, true_cols, col_fragid, col_bucket, nfrags, taus):
    rows = []
    for tau in taus:
        tp = fp = fn = 0
        tani = []
        cl1 = []
        for i, tc in enumerate(true_cols):
            pc = np.nonzero(pred[i] >= tau)[0]
            tset = set(tc.tolist())
            pset = set(pc.tolist())
            inter = len(tset & pset)
            tp += inter
            fp += len(pset) - inter
            fn += len(tset) - inter
            union = len(tset | pset)
            tani.append(inter / union if union else 1.0)
            kt = kmax_per_frag(tc, col_fragid, col_bucket, nfrags)
            kp = kmax_per_frag(pc, col_fragid, col_bucket, nfrags)
            cl1.append(int(np.abs(kp - kt).sum()))
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rows.append(dict(tau=float(tau), recall=rec, precision=prec,
                         mean_tanimoto=float(np.mean(tani)),
                         mean_count_l1=float(np.mean(cl1))))
    return rows


def detailed(pred, true_cols, col_fragid, col_bucket, frag_surv, nfrags, tau):
    kbins = [1, 2, 3, 4, 5]  # 6 => "6+"
    def kbin(k): return min(k, 6)
    per_bin = defaultdict(lambda: dict(n=0, recalled=0, exact=0, dk_sum=0,
                                       dk_hist=defaultdict(int)))
    dk_all = []
    fp_counts_per_mol = []
    fp_kpred = []
    offprefix_num = 0
    offprefix_den = 0

    for i, tc in enumerate(true_cols):
        pc = np.nonzero(pred[i] >= tau)[0]
        kt = kmax_per_frag(tc, col_fragid, col_bucket, nfrags)
        kp = kmax_per_frag(pc, col_fragid, col_bucket, nfrags)
        true_present = np.nonzero(kt > 0)[0]
        pred_present = np.nonzero(kp > 0)[0]

        for fid in true_present:
            b = kbin(int(kt[fid]))
            d = per_bin[b]
            d["n"] += 1
            dk = int(kp[fid]) - int(kt[fid])
            if kp[fid] > 0:
                d["recalled"] += 1
            if dk == 0:
                d["exact"] += 1
            d["dk_sum"] += dk
            d["dk_hist"][int(np.clip(dk, -8, 8))] += 1
            dk_all.append(dk)

        # false-positive fragments (predicted present, truly absent)
        fp_frags = pred_present[kt[pred_present] == 0]
        fp_counts_per_mol.append(int(len(fp_frags)))
        for fid in fp_frags:
            fp_kpred.append(int(kp[fid]))

        # off-prefix among predicted-present frags: are active pred buckets a
        # contiguous prefix of the fragment's surviving buckets?
        pred_buckets_by_frag = defaultdict(set)
        for c in pc:
            pred_buckets_by_frag[int(col_fragid[c])].add(int(col_bucket[c]))
        for fid, abuckets in pred_buckets_by_frag.items():
            surv = frag_surv[fid]
            j = len(abuckets)
            prefix = set(surv[:j].tolist())
            offprefix_den += 1
            if abuckets != prefix:
                offprefix_num += 1

    bin_summary = {}
    for b in kbins + [6]:
        d = per_bin.get(b)
        label = f"{b}" if b < 6 else "6+"
        if not d or d["n"] == 0:
            continue
        bin_summary[label] = dict(
            n=d["n"],
            recall=d["recalled"] / d["n"],
            p_exact=d["exact"] / d["n"],
            mean_dk=d["dk_sum"] / d["n"],
            dk_hist={int(k): int(v) for k, v in sorted(d["dk_hist"].items())},
        )
    dk_all = np.array(dk_all)
    dk_hist = {int(v): int(c) for v, c in
               zip(*np.unique(np.clip(dk_all, -6, 6), return_counts=True))}
    return dict(
        tau=float(tau),
        n_molecules=len(true_cols),
        by_true_count=bin_summary,
        delta_k_hist_clipped=dk_hist,
        frac_dk_zero=float((dk_all == 0).mean()),
        frac_dk_neg=float((dk_all < 0).mean()),
        frac_dk_pos=float((dk_all > 0).mean()),
        fp_frags_per_mol_mean=float(np.mean(fp_counts_per_mol)),
        fp_frags_per_mol_median=float(np.median(fp_counts_per_mol)),
        fp_kpred_mean=float(np.mean(fp_kpred)) if fp_kpred else 0.0,
        offprefix_rate=offprefix_num / offprefix_den if offprefix_den else 0.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--vocab", default="data/dataset/RankingEntropyUniqueMultiplicity/bitinfo_to_idx.pkl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pred, true_cols, col2key = load(args.dump, args.vocab)
    frags, col_fragid, col_bucket, frag_surv = build_maps(col2key)
    nfrags = len(frags)
    print(f"[analyze] N={len(true_cols)} D={pred.shape[1]} nfrags={nfrags}")

    taus = [0.3, 0.4, 0.5, 0.6, 0.7]
    sweep = tau_sweep(pred, true_cols, col_fragid, col_bucket, nfrags, taus)
    best = max(sweep, key=lambda r: r["mean_tanimoto"])
    print("\n[tau sweep]  tau   recall  prec   Tanimoto  countL1")
    for r in sweep:
        star = " *" if r is best else ""
        print(f"  {r['tau']:.1f}   {r['recall']:.3f}  {r['precision']:.3f}  "
              f"{r['mean_tanimoto']:.3f}    {r['mean_count_l1']:.2f}{star}")

    det = detailed(pred, true_cols, col_fragid, col_bucket, frag_surv, nfrags, best["tau"])
    print(f"\n[detailed @ tau={det['tau']}]")
    print(f"  Delta-k: exact={det['frac_dk_zero']:.3f}  "
          f"undercount={det['frac_dk_neg']:.3f}  overcount={det['frac_dk_pos']:.3f}")
    print(f"  FP fragments/mol: mean={det['fp_frags_per_mol_mean']:.2f} "
          f"median={det['fp_frags_per_mol_median']:.0f}")
    print(f"  off-prefix rate (non-thermometer preds): {det['offprefix_rate']:.4f}")
    print("  by true count k_true:  k   n     recall  P(exact)  mean_dk")
    for label, d in det["by_true_count"].items():
        print(f"     {label:>3}  {d['n']:>6}  {d['recall']:.3f}   "
              f"{d['p_exact']:.3f}     {d['mean_dk']:+.3f}")

    out = args.out or os.path.join(args.dump, "error_model.json")
    with open(out, "w") as f:
        json.dump(dict(tau_sweep=sweep, best_tau=best["tau"], detailed=det), f, indent=2)
    print(f"\n[analyze] wrote {out}")


if __name__ == "__main__":
    main()
