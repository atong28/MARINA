"""Histogram the entropy-selected sherlock (RankingEntropy) bits by Morgan radius.

The feature keys in bitinfo_to_idx.pkl are 4-tuples (bit_id, atom_symbol, frag_smiles, radius).
Answers directly: of the 16384 entropy-selected bits, what fraction come from radius > 6? If
~0%, entropy selection did not want the larger environments a higher radius cap exposes.

Usage: python radius_hist.py --fp NAME=/path/bitinfo_to_idx.pkl [--fp NAME2=...]

Deps: stdlib only. Runs in any env.
"""
import argparse, pickle
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", action="append", default=[], help="NAME=/path/bitinfo_to_idx.pkl")
    args = ap.parse_args()
    for spec in args.fp:
        name, path = spec.split("=", 1)
        try:
            with open(path, "rb") as f:
                m = pickle.load(f)
        except FileNotFoundError:
            print(f"MISSING {name}: {path}", flush=True)
            continue
        keys = list(m.keys())
        # sherlock keys are (bit_id, atom, frag, radius); guard for other shapes
        radii = [k[3] for k in keys if isinstance(k, tuple) and len(k) == 4 and isinstance(k[3], int)]
        n = len(keys)
        print(f"\n=== {name} ({n} selected bits, {len(radii)} with a radius field) ===")
        if not radii:
            print("  (no 4-tuple radius keys — not a sherlock-style map)")
            continue
        c = Counter(radii)
        n_gt6 = sum(v for r, v in c.items() if r > 6)
        for r in sorted(c):
            print(f"  r={r:2d}: {c[r]:6d}  ({100*c[r]/len(radii):5.2f}%)")
        print(f"  radius>6: {n_gt6}/{len(radii)}  ({100*n_gt6/len(radii):.2f}%)")


if __name__ == "__main__":
    main()
