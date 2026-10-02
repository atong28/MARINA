"""Exp 2 over the new retrieval set, plus a prefix reproduction check.

Uses exp2_specificity.py's own load_csr_colsets / mass_vector / specificity (identical metric
definitions). The new MARINA-DB retrieval.pkl is the old 531,087 rows in the same order with 840
CH-NMR-NP rows appended, so restricting every FP to rows [0, --prefix) must reproduce the published
(old-set) numbers — a check that the rebuilt reference FPs match the published definitions.
Masses are cached (.npy) so per-FP invocations don't redo the single-threaded mass pass.
Adds onbits_mean (np.diff(crow).mean(), the table's On-bits/mol) to each entry.

Usage: python exp2_with_prefix.py --retrieval <pkl> --mass-cache masses.npy --prefix 531087 \
           --fp NAME=/path/rankingset.pt [...] --out exp2_X.json
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from exp2_specificity import load_csr_colsets, mass_vector, specificity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--mass-cache", required=True)
    ap.add_argument("--prefix", type=int, default=531087)
    ap.add_argument("--fp", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = {"fingerprints": {}, "prefix_check": {"prefix_rows": a.prefix, "fingerprints": {}}}
    mass = None
    for spec in a.fp:
        name, path = spec.split("=", 1)
        N, D, crow, col = load_csr_colsets(path)
        if mass is None:
            if os.path.exists(a.mass_cache):
                mass = np.load(a.mass_cache)
            else:
                mass = mass_vector(a.retrieval, N)
                np.save(a.mass_cache, mass)
            assert len(mass) == N, (len(mass), N)
        nnz = np.diff(crow)
        full = specificity(crow, col, mass, N)
        full.update(D=D, onbits_mean=float(nnz.mean()))
        pre = specificity(crow, col, mass[:a.prefix], a.prefix)
        pre.update(D=D, onbits_mean=float(nnz[:a.prefix].mean()))
        out["fingerprints"][name] = full
        out["prefix_check"]["fingerprints"][name] = pre
        print(f"{name:20s} D={D:5d} N={N}  on-bits {full['onbits_mean']:.1f}  coll {full['collision_pct']:.2f}%  "
              f"bad {full['bad_collision_pct']:.2f}%  largest {full['largest_group']}   | prefix: on-bits "
              f"{pre['onbits_mean']:.1f} coll {pre['collision_pct']:.2f}% bad {pre['bad_collision_pct']:.2f}% "
              f"largest {pre['largest_group']}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
