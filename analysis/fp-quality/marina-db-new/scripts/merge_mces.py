"""Concatenate the RASCAL MCES shards and join them onto the FP-Tanimoto pair table.

Asserts the shards' (i, j) set is exactly the pair index make_pairs.py wrote (same sampling),
then writes the exp1_analyze.py input: columns i, j, mces, timedout, <one Tanimoto col per FP>.

Usage: python merge_mces.py --shards 'results/mces_shards/shard*.parquet' \
           --fps results/pairs_fp_tanimoto.parquet --out results/exp1_pairs.parquet
"""
import argparse, glob

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--fps", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(a.shards))
    m = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fp = pd.read_parquet(a.fps).drop(columns=["mces"])
    assert len(m) == len(fp) == m[["i", "j"]].drop_duplicates().shape[0], (len(m), len(fp))
    df = m[["i", "j", "mces", "timedout"]].merge(fp, on=["i", "j"], how="inner", validate="1:1")
    assert len(df) == len(fp), "MCES shard pairs != pair index"
    df = df.sort_values(["i", "j"]).reset_index(drop=True)
    df.to_parquet(a.out)
    print(f"{len(files)} shards, {len(df)} pairs; NaN mces {int(np.isnan(df.mces).sum())}; "
          f"timedout {int(df.timedout.sum())}; FP cols {[c for c in df.columns if c not in ('i','j','mces','timedout')]}"
          f" -> {a.out}")


if __name__ == "__main__":
    main()
