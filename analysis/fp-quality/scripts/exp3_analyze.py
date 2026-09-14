"""Exp 3 (analyze) — clean far-collision tail from the within-bucket MCS parquet.

The raw exp3 tail is contaminated two ways that are NOT real scaffold hops:
  - exotic/inorganic dataset junk (Sr/Hg/Te/As/Pb/Si/Se, radicals) and <6-heavy-atom
    fragments, where MCS-over-edges is degenerate;
  - `partial` pairs (rdFMCS timed out) whose sim is only a LOWER bound.
This filters to NP/drug-like organic (C,H,N,O,S,P,F,Cl,Br,I), >=6 heavy atoms, complete
MCS, and reports P(MCS sim | same FP) + the per-tie-group far-collision rate. A "far
collision" == same fingerprint, low MCS == a same-formula isomer with different connectivity
(the case where an exact-FP match does NOT pin structure for Moonshot).

    python exp3_analyze.py --in results/exp3_collision_mces_uniqmult.parquet
"""
import argparse, json, os
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
NPSET = set("C H N O S P F Cl Br I".split())


def clean_flag(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return (False, 0)
    org = (all(a.GetSymbol() in NPSET for a in m.GetAtoms())
           and all(a.GetNumRadicalElectrons() == 0 for a in m.GetAtoms()))
    return (org, m.GetNumHeavyAtoms())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--min-heavy", type=int, default=6)
    ap.add_argument("--out", default=None, help="optional summary JSON")
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    for c in ("i", "j"):
        r = df["smiles_" + c].map(clean_flag)
        df["org_" + c] = r.map(lambda x: x[0]); df["hv_" + c] = r.map(lambda x: x[1])
    df["clean"] = df.org_i & df.org_j & (df[["hv_i", "hv_j"]].min(axis=1) >= args.min_heavy)
    df["complete"] = np.isfinite(df.mcs_sim) & (~df.partial)

    sub = df[df.clean & df.complete]
    v = sub.mcs_sim.to_numpy()
    g = sub.groupby("gid").mcs_sim.min()
    thrs = (0.3, 0.5, 0.7, 0.9)
    summ = {
        "pairs_total": int(len(df)),
        "pairs_junk": int((~df.clean).sum()),
        "pairs_partial": int(df.partial.sum()),
        "pairs_clean_complete": int(len(sub)),
        "mcs_median": float(np.median(v)), "mcs_p25": float(np.quantile(v, .25)),
        "mcs_p10": float(np.quantile(v, .10)), "mcs_p5": float(np.quantile(v, .05)),
        "far_pair_frac": {f"<{t}": float((v < t).mean()) for t in thrs},
        "tie_groups_clean": int(len(g)),
        "far_group_frac": {f"<{t}": float((g < t).mean()) for t in (0.5, 0.7)},
    }
    print(f"total {summ['pairs_total']} | junk {summ['pairs_junk']} "
          f"({summ['pairs_junk']/summ['pairs_total']:.1%}) | partial {summ['pairs_partial']}")
    print(f"\nCLEAN organic >= {args.min_heavy} heavy + complete MCS: n={summ['pairs_clean_complete']}")
    print(f"  median={summ['mcs_median']:.3f} p25={summ['mcs_p25']:.3f} "
          f"p10={summ['mcs_p10']:.3f} p5={summ['mcs_p5']:.3f}")
    for t in thrs:
        print(f"  far collisions (pairs) MCS<{t}: {summ['far_pair_frac'][f'<{t}']:.2%}")
    for t in (0.5, 0.7):
        print(f"  tie-groups with a far member (min-pair MCS<{t}): {summ['far_group_frac'][f'<{t}']:.2%}")

    if args.out:
        json.dump(summ, open(args.out, "w"), indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
