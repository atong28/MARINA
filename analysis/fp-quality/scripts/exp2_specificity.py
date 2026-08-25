"""Exp 2 — mass-stratified fingerprint specificity (collision analysis).

Precedent: Huber & Pollmann, "Count your bits" (J. Cheminform. 2026), Fig 2 — a
duplicate spanning a large mass difference is a worse collision than one within 1 Da.
Extends analysis/fp-collision-recheck (column-set hashing) with mass stratification.

Runs in MARINA's own pixi env (needs torch to read the CSR rankingset + rdkit for mass).
Read-only on Datasets/ and the PVC.

For each fingerprint (CSR rankingset.pt, binary thermometer/selected bits) it groups
molecules by identical nonzero column-set, then for each tie group computes the maximum
pairwise monoisotopic mass difference and stratifies duplicate molecules into mass bins.

Usage:
    python exp2_specificity.py --retrieval /path/retrieval.pkl \
        --fp NAME=/path/RankingEntropy/rankingset.pt [--fp NAME2=...] \
        --out results/exp2_specificity.json
"""
import argparse, json, os, pickle, sys, time
from collections import Counter, defaultdict

import numpy as np

MASS_BINS = [0, 1, 10, 50, 100, 200, 300, 400, float("inf")]
MASS_LABELS = ["0-1", "1-10", "10-50", "50-100", "100-200", "200-300", "300-400", ">400"]


def load_csr_colsets(path):
    """Return (N, D, list-of-colindex-arrays) from a torch CSR rankingset.pt."""
    import torch
    csr = torch.load(path, weights_only=True)
    crow = csr.crow_indices().numpy()
    col = csr.col_indices().numpy()
    N, D = int(csr.shape[0]), int(csr.shape[1])
    return N, D, crow, col


def mass_vector(retrieval_pkl, n_rows):
    """Monoisotopic mass per row, aligned to retrieval.pkl integer keys 0..N-1."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    with open(retrieval_pkl, "rb") as f:
        R = pickle.load(f)
    # dict{int: {'smiles':..}} or list
    getter = (lambda i: R[i]) if isinstance(R, dict) else (lambda i: R[i])
    mass = np.full(n_rows, np.nan, dtype=np.float64)
    t0 = time.time()
    for i in range(n_rows):
        e = getter(i)
        smi = e["smiles"] if isinstance(e, dict) else e
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is not None:
            mass[i] = Descriptors.ExactMolWt(m)
        if i % 50000 == 0:
            print(f"  mass {i}/{n_rows}  {time.time()-t0:.0f}s", flush=True)
    return mass


def specificity(crow, col, mass, N):
    groups = defaultdict(list)
    for i in range(N):
        groups[col[crow[i]:crow[i + 1]].tobytes()].append(i)
    n_zero = len(groups.get(b"", []))
    ties = {k: v for k, v in groups.items() if len(v) > 1}
    n_in_tie = sum(len(v) for v in ties.values())
    largest = max((len(v) for v in ties.values()), default=0)

    # mass span per tie group; bin the molecules by their group's max pairwise mass diff
    bin_counts = np.zeros(len(MASS_LABELS), dtype=np.int64)
    n_bad = 0  # molecules in a group spanning > 200 Da
    spans = []
    for v in ties.values():
        mv = mass[v]
        mv = mv[~np.isnan(mv)]
        span = float(mv.max() - mv.min()) if mv.size >= 2 else 0.0
        spans.append(span)
        b = int(np.searchsorted(MASS_BINS, span, side="right") - 1)
        b = min(max(b, 0), len(MASS_LABELS) - 1)
        bin_counts[b] += len(v)
        if span > 200:
            n_bad += len(v)
    return {
        "N": N,
        "all_zero": n_zero,
        "distinct": len(groups),
        "tie_groups": len(ties),
        "n_in_tie": n_in_tie,
        "collision_pct": 100 * n_in_tie / N,
        "ceiling_pct": 100 * (N - n_in_tie) / N,
        "largest_group": largest,
        "bad_collision_pct": 100 * n_bad / N,  # groups spanning >200 Da
        "median_group_span_Da": float(np.median(spans)) if spans else 0.0,
        "mass_bin_labels": MASS_LABELS,
        "mass_bin_molecule_counts": bin_counts.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--fp", action="append", default=[], help="NAME=/path/rankingset.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fps = {}
    for spec in args.fp:
        name, path = spec.split("=", 1)
        if os.path.exists(path):
            fps[name] = path
        else:
            print(f"MISSING fingerprint {name}: {path}", flush=True)

    if not fps:
        print("No fingerprint matrices found — nothing to do.", flush=True)
        json.dump({"error": "no fingerprints found", "requested": args.fp},
                  open(args.out, "w"), indent=2)
        return

    # one mass vector, sized to the first FP (all share the retrieval row order)
    name0 = next(iter(fps))
    N0, D0, crow0, col0 = load_csr_colsets(fps[name0])
    print(f"reference FP {name0}: N={N0} D={D0}", flush=True)
    mass = mass_vector(args.retrieval, N0)
    print(f"masses parsed: {np.isfinite(mass).sum()}/{N0}", flush=True)

    out = {"fingerprints": {}}
    for name, path in fps.items():
        N, D, crow, col = load_csr_colsets(path)
        if N != N0:
            print(f"WARN {name}: N={N} != {N0}; skipping mass stratification alignment", flush=True)
        res = specificity(crow, col, mass, N)
        res["D"] = D
        out["fingerprints"][name] = res
        print(f"\n=== {name} (D={D}) ===")
        print(f"  collision {res['collision_pct']:.2f}%  ceiling {res['ceiling_pct']:.2f}%  "
              f"largest {res['largest_group']}  bad(>200Da) {res['bad_collision_pct']:.2f}%")
        print(f"  mass-bin molecule counts: "
              + ", ".join(f"{l}:{c}" for l, c in zip(MASS_LABELS, res['mass_bin_molecule_counts'])))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
