#!/usr/bin/env python3
"""4_fp_rankingset.py -- build fingerprint vocab + rankingset for config.FP_TYPES.

Structure track: depends ONLY on retrieval.pkl (stage 3), so it runs without any spectral
data or MS/MS. For each fp_type it writes DATA_DATASET/<fp_type>/:

    bitinfo_to_idx.pkl   selected feature -> column index
    rankingset.pt        torch.sparse_csr over the retrieval set, rows L2-normalized

config.FP_TYPES builds all four entropy-selected loaders so the fp-quality analysis can
compare them on one retrieval set:
    RankingEntropyMultiplicityUncapped  (deployed, D8; uncapped thermometer)
    RankingEntropy                      (presence-only Morgan bits; "sherlock")
    RankingEntropyMultiplicity          (thermometer capped at k=5)
    RankingEntropySubstructure          (deduplicated fragment presence)

Selection reuses each loader's own machinery -- _prepare_counts (feature counts over the
retrieval set, cached per-kind) + _filter_by_radius (Morgan filters by the radius in the
BitInfo key; the substructure/multiplicity loaders keep all, radius already bounded at
enumeration) -- then the shared entropy top-K. This is exactly EntropyFPLoader.setup's
vocab logic minus its FragIdx step (the training columns are stage 9, 9_fp_fragidx.py,
which needs the built index/arrow). The per-kind cap is applied inside _prepare_counts,
so the capped-k5 and uncapped variants differ correctly.

    DATASET_ROOT=/workspace pixi run python3 \
        scripts/marina_db/build/4_fp_rankingset.py --num_procs 16
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[3]))

from config import DATA_DATASET, RETRIEVAL_PKL, FP_TYPES, FP_OUT_DIM, FP_RADIUS
from src.modules.data.fp_utils import compute_entropy, select_topk_by_entropy, load_smiles_index
from src.modules.data.fp_loader import FP_LOADERS


def build_one(fp_type, retrieval_path, radius, out_dim, retrieval_size, num_procs):
    loader_class = FP_LOADERS.get(fp_type)
    if loader_class is None:
        raise SystemExit(f"[{fp_type}] unknown fp_type; FP_LOADERS keys: {list(FP_LOADERS)}")
    loader = loader_class(dataset_root=str(DATA_DATASET), retrieval_path=str(retrieval_path))
    loader.max_radius = int(radius)

    # Feature counts over the retrieval set (cached per-kind by COUNTS_PREFIX), then the
    # loader's radius filter (polymorphic: Morgan by key radius; others keep all).
    counts = loader._prepare_counts(int(radius), num_procs)
    filtered = loader._filter_by_radius(counts)
    if not filtered:
        raise RuntimeError(f"[{fp_type}] no features <= radius {radius} in retrieval counts")
    features, cnts = zip(*filtered)

    ent = compute_entropy(np.asarray(cnts), total_dataset_size=retrieval_size)
    k = min(int(out_dim), len(features))
    topk = select_topk_by_entropy(ent, list(features), k)
    mapping = {features[i]: j for j, i in enumerate(topk)}

    loader.bitinfo_to_fp_index_map = mapping
    loader.fp_index_to_bitinfo_map = {v: kk for kk, v in mapping.items()}
    loader.out_dim = len(mapping)
    csr = loader._build_rankingset(fp_type, num_procs=num_procs)   # writes vocab + rankingset

    out_dir = Path(DATA_DATASET) / fp_type
    print(f"[{fp_type}] candidates={len(features)}  selected={len(mapping)}  "
          f"-> {out_dir/'bitinfo_to_idx.pkl'}, {out_dir/'rankingset.pt'} "
          f"({csr.shape[0]} x {csr.shape[1]})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp_types", default=None,
                    help="comma-separated; defaults to config.FP_TYPES")
    ap.add_argument("--out_dim", type=int, default=FP_OUT_DIM)
    ap.add_argument("--radius", type=int, default=FP_RADIUS)
    ap.add_argument("--retrieval", default=str(RETRIEVAL_PKL))
    ap.add_argument("--num_procs", type=int, default=0, help="0 = all cores")
    a = ap.parse_args()

    fp_types = [t for t in a.fp_types.split(",") if t] if a.fp_types else FP_TYPES
    retrieval_size = len(load_smiles_index(a.retrieval))
    print(f"retrieval={a.retrieval}  molecules={retrieval_size}  radius={a.radius}  "
          f"out_dim={a.out_dim}  fp_types={fp_types}", flush=True)

    for fp_type in fp_types:
        build_one(fp_type, a.retrieval, a.radius, a.out_dim, retrieval_size, a.num_procs)


if __name__ == "__main__":
    main()
