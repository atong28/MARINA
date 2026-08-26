#!/usr/bin/env python3
"""9_fp_fragidx.py -- build the training-path FragIdx parquets for config.FP_TYPES.

Spectral/training track: needs the built index (stage 6) + arrow tree (stage 8) and the
vocab written by stage 4 (4_fp_rankingset.py). For each fp_type it writes per-split
DATASET_ROOT/arrow/<split>/<loader FRAGIDX_FILENAME> -- per-molecule feature columns so
the model's build_mfp(idx) trains on that vocabulary. Split out of the old fp_vocab stage
so the vocab + rankingset (structure-only) are not blocked on spectral data.

    DATASET_ROOT=/workspace pixi run python3 \
        scripts/marina_db/build/9_fp_fragidx.py --num_procs 16
"""
import argparse
import pickle
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[3]))

from config import DATA_DATASET, RETRIEVAL_PKL, FP_TYPES, FP_RADIUS
from src.modules.data.fp_utils import build_fragidx_parquets
from src.modules.data.fp_loader import FP_LOADERS


def build_one(fp_type, radius, num_procs):
    loader_class = FP_LOADERS.get(fp_type)
    if loader_class is None:
        raise SystemExit(f"[{fp_type}] unknown fp_type; FP_LOADERS keys: {list(FP_LOADERS)}")
    loader = loader_class(dataset_root=str(DATA_DATASET), retrieval_path=str(RETRIEVAL_PKL))

    vocab_path = Path(DATA_DATASET) / fp_type / "bitinfo_to_idx.pkl"
    if not vocab_path.exists():
        raise SystemExit(f"[{fp_type}] {vocab_path} missing -- run stage 4 (4_fp_rankingset.py) first")
    mapping = pickle.load(open(vocab_path, "rb"))

    build_fragidx_parquets(
        index_path=str(Path(DATA_DATASET) / "index.pkl"),
        out_dir=str(DATA_DATASET),
        bitinfo_to_col=mapping,
        radius=int(radius),
        num_procs=num_procs,
        feature_kind=loader.FEATURE_KIND,
        filename=loader.FRAGIDX_FILENAME,
    )
    print(f"[{fp_type}] wrote FragIdx parquets ({loader.FRAGIDX_FILENAME}) "
          f"under {Path(DATA_DATASET)/'arrow'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp_types", default=None,
                    help="comma-separated; defaults to config.FP_TYPES")
    ap.add_argument("--radius", type=int, default=FP_RADIUS)
    ap.add_argument("--num_procs", type=int, default=0, help="0 = all cores")
    a = ap.parse_args()

    index_path = Path(DATA_DATASET) / "index.pkl"
    if not index_path.exists():
        raise SystemExit(f"{index_path} missing -- run stages 6-8 (index/splits/arrow) first")

    fp_types = [t for t in a.fp_types.split(",") if t] if a.fp_types else FP_TYPES
    print(f"fp_types={fp_types}  radius={a.radius}", flush=True)
    for fp_type in fp_types:
        build_one(fp_type, a.radius, a.num_procs)


if __name__ == "__main__":
    main()
