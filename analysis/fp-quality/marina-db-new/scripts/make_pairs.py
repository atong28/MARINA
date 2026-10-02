"""Write the Exp 1 pair index (i, j) WITHOUT running MCES, so FP Tanimoto columns can be added
(exp1_add_fps.py) while the RASCAL shards are still running and large rankingsets can be deleted
right after use (disk). Sampling is a verbatim copy of exp1_mces.py (mass-stratified pool, seed,
sorted unique pairs); merge_mces.py asserts the pair set equals the MCES shards' pair set.

Usage: python make_pairs.py --retrieval <pkl> --out pairs_index.parquet [--n-pool 5000 --n-pairs 100000 --seed 0]
"""
import argparse, os, pickle, sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from exp1_mces import _mass_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--n-pool", type=int, default=5000)
    ap.add_argument("--n-pairs", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    with open(a.retrieval, "rb") as f:
        R = pickle.load(f)
    smiles = [(R[i]["smiles"] if isinstance(R[i], dict) else R[i]) for i in range(len(R))]
    with Pool(a.workers) as p:
        masses = np.array(p.map(_mass_one, smiles, chunksize=2000))
    valid = np.where(np.isfinite(masses))[0]
    order = valid[np.argsort(masses[valid])]
    idx = np.unique(order[np.linspace(0, len(order) - 1, a.n_pool).astype(int)])
    pairs = set()
    while len(pairs) < a.n_pairs:
        x, y = rng.choice(idx, 2, replace=False)
        pairs.add((int(min(x, y)), int(max(x, y))))
    pairs = sorted(pairs)
    df = pd.DataFrame({"i": [p[0] for p in pairs], "j": [p[1] for p in pairs],
                       "mces": np.nan})
    df.to_parquet(a.out)
    print(f"pool {len(idx)}; {len(df)} pairs; {len(set(df.i) | set(df.j))} distinct rows; "
          f"{int(((df.i >= 531087) | (df.j >= 531087)).sum())} pairs touch rows >= 531087 -> {a.out}")


if __name__ == "__main__":
    main()
