#!/usr/bin/env python3
"""Average single_atom_bit_eval.py per-seed JSONs across seeds.

All seeds are scored on the same benchmark molecules, so per-element sample counts are
identical across seeds and only the accuracy/error values are averaged (unweighted mean
over seeds). Deltas are recomputed from the averaged combos.

Usage:
    python scripts/benchmark/average_single_atom_bits.py \
        results/single_atom_bits_uniqmult_s1.json \
        results/single_atom_bits_uniqmult_s2.json \
        --out results/single_atom_bits_uniqmult_mean.json
"""
import argparse
import json
from statistics import mean


def _avg(dicts, path):
    """Mean of the value at a dotted path across a list of dicts."""
    vals = []
    for d in dicts:
        cur = d
        for p in path.split("."):
            cur = cur[p]
        vals.append(cur)
    return mean(vals)


def average(seed_results):
    """seed_results: list of the per-seed `results` dicts. Returns averaged structure."""
    splits = seed_results[0].keys()
    out = {}
    for split in splits:
        out[split] = {}
        combos = seed_results[0][split].keys()
        for combo in combos:
            cs = [sr[split][combo] for sr in seed_results]
            if any(c.get("n", 0) == 0 for c in cs):
                out[split][combo] = {"n": 0}
                continue
            per_keys = cs[0]["element_total"]["per_key"].keys()
            out[split][combo] = {
                "n": cs[0]["n"],
                "bit": {
                    "mean_wrong_per_mol": _avg(cs, "bit.mean_wrong_per_mol"),
                    "accuracy_pct": _avg(cs, "bit.accuracy_pct"),
                    "n_single_cols": cs[0]["bit"]["n_single_cols"],
                },
                "element_token": {
                    "overall_acc_pct": _avg(cs, "element_token.overall_acc_pct"),
                    "mol_all_correct_pct": _avg(cs, "element_token.mol_all_correct_pct"),
                },
                "element_total": {
                    "overall_acc_pct": _avg(cs, "element_total.overall_acc_pct"),
                    "mol_all_correct_pct": _avg(cs, "element_total.mol_all_correct_pct"),
                    "per_key": {
                        k: {"acc_pct": mean(c["element_total"]["per_key"][k]["acc_pct"] for c in cs),
                            "mean_abserr": mean(c["element_total"]["per_key"][k]["mean_abserr"] for c in cs),
                            "n": cs[0]["element_total"]["per_key"][k]["n"]}
                        for k in per_keys
                    },
                },
            }
    return out


def print_report(results, n_seeds):
    for split, combos in results.items():
        print(f"\n{'='*72}\nSPLIT: {split}   (mean of {n_seeds} seeds)\n{'='*72}")
        for combo, m in combos.items():
            if m.get("n", 0) == 0:
                print(f"[{combo}] no entries"); continue
            b = m["bit"]
            print(f"\n[{combo}]  n={m['n']}")
            print(f"  bit          : {b['mean_wrong_per_mol']:.3f} wrong/mol   acc={b['accuracy_pct']:.2f}%")
            for name in ("element_token", "element_total"):
                r = m[name]
                print(f"  {name:13}: overall_acc={r['overall_acc_pct']:.2f}%  "
                      f"mol_all_correct={r['mol_all_correct_pct']:.2f}%")
            r = m["element_total"]["per_key"]
            print("  per-element  : " + "  ".join(
                f"{k}={v['acc_pct']:.0f}%(n{v['n']})" for k, v in r.items()))
        if "nmr" in combos and "nmr_formula" in combos and combos["nmr"].get("n"):
            a, c = combos["nmr"], combos["nmr_formula"]
            print(f"\n  DELTA (nmr_formula - nmr):")
            print(f"    bit wrong/mol : {c['bit']['mean_wrong_per_mol'] - a['bit']['mean_wrong_per_mol']:+.3f}")
            print(f"    bit acc pp    : {c['bit']['accuracy_pct'] - a['bit']['accuracy_pct']:+.2f}")
            print(f"    el-total acc  : {c['element_total']['overall_acc_pct'] - a['element_total']['overall_acc_pct']:+.2f} pp")
            print(f"    el-total full : {c['element_total']['mol_all_correct_pct'] - a['element_total']['mol_all_correct_pct']:+.2f} pp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_files", nargs="+", help="per-seed single_atom_bits_*.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    loaded = [json.load(open(f)) for f in args.json_files]
    n_cols = {d["n_single_cols"] for d in loaded}
    fp_types = {d["fp_type"] for d in loaded}
    if len(n_cols) != 1 or len(fp_types) != 1:
        raise SystemExit(f"seeds disagree on vocab: fp_type={fp_types} n_single_cols={n_cols}")

    avg = average([d["results"] for d in loaded])
    print_report(avg, len(loaded))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"seeds": args.json_files, "fp_type": fp_types.pop(),
                       "n_single_cols": n_cols.pop(), "n_seeds": len(loaded),
                       "results": avg}, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
