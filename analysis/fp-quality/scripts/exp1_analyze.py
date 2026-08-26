"""Exp 1 (analyze) — Spearman(FP-sim, MCES) per fingerprint, paired bootstrap CIs.

Consumes the pairs parquet from exp1_mces.py. Higher Spearman = the fingerprint's similarity
better tracks graph-based structural similarity (the "meaningful similarity" C7 claim). Paired
across fingerprints (identical pairs); bootstrap over pairs for CIs and Δρ vs a reference.

Deps: scipy + pandas + pyarrow (no torch). Run under the ~/Workspace master pixi env (which
has scipy), or any env with scipy — this step is cheap, so it does not need the cluster.
"""
import argparse, json, os

import numpy as np
import pandas as pd
from scipy.stats import rankdata as _rankdata


def spearman(x, y):
    """Spearman rho via ranks + Pearson, so a paired bootstrap can rank once per draw."""
    rx = _rankdata(x)
    ry = _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/exp1_pairs.parquet")
    ap.add_argument("--ref", default="sherlock")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/exp1_spearman.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.pairs)
    df = df[np.isfinite(df["mces"])].reset_index(drop=True)
    # every column that is not bookkeeping (i/j/mces/timedout) and not a bool flag is an FP
    fp_cols = [c for c in df.columns
               if c not in ("i", "j", "mces", "timedout") and df[c].dtype != bool]
    mces = df["mces"].to_numpy()
    cols = {c: df[c].to_numpy() for c in fp_cols}
    n = len(df)
    print(f"{n} pairs with valid MCES; fingerprints: {fp_cols}", flush=True)

    rho = {c: spearman(cols[c], mces) for c in fp_cols}

    # paired bootstrap over pairs (same resample for every fingerprint)
    boot = {c: np.empty(args.n_boot) for c in fp_cols}
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        m = mces[idx]
        for c in fp_cols:
            boot[c][b] = spearman(cols[c][idx], m)

    out = {"n_pairs": n, "spearman": {}, "delta_vs_" + args.ref: {}}
    for c in fp_cols:
        lo, hi = np.percentile(boot[c], [2.5, 97.5])
        out["spearman"][c] = {"rho": rho[c], "ci95": [float(lo), float(hi)]}
    if args.ref in fp_cols:
        for c in fp_cols:
            if c == args.ref:
                continue
            d = boot[c] - boot[args.ref]
            lo, hi = np.percentile(d, [2.5, 97.5])
            out["delta_vs_" + args.ref][c] = {
                "delta_rho": rho[c] - rho[args.ref],
                "ci95": [float(lo), float(hi)],
                "p_gt_0": float((d > 0).mean()),
            }

    for c in sorted(fp_cols, key=lambda x: -rho[x]):
        ci = out["spearman"][c]["ci95"]
        print(f"  {c:32s} rho={rho[c]:.4f}  95%CI[{ci[0]:.4f},{ci[1]:.4f}]", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
