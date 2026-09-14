"""Concat exp3 shard parquets (from the Nautilus sweep) and print the pooled tail.

    python exp3_concat.py --glob '/path/exp3_*shard*.parquet' --out results/exp3_pooled.parquet
"""
import argparse, glob, json, os
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no shards match {args.glob}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out)

    comp = df[np.isfinite(df["mcs_sim"]) & (~df["partial"])]["mcs_sim"].to_numpy()
    v = df["mcs_sim"].to_numpy(); v = v[np.isfinite(v)]
    print(f"{len(files)} shards -> {len(df)} pairs ({int(df['partial'].sum())} partial)")
    if v.size:
        print(f"MCS median={np.median(v):.3f}  p25={np.quantile(v,.25):.3f}  p5={np.quantile(v,.05):.3f}")
        for thr in (0.3, 0.5, 0.7, 0.9):
            c = float((comp < thr).mean()) if comp.size else float("nan")
            print(f"  far collisions MCS<{thr}: all={float((v<thr).mean()):.3%}  complete-only={c:.3%}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
