#!/usr/bin/env python3
"""13_pack_arrow.py -- pack the finalized Arrow shards into memmap-ready arrays.

Spectral/training track, final step: runs after the arrow tree (stage 8), the FragIdx
parquets (stage 9), and the dataset-wide peak collapse (stage 11) that mutates the arrow
shards -- so it packs the finalized data. For every DATASET_ROOT/arrow/<split>/*.parquet
it writes DATASET_ROOT/packed/<split>/<shard>/ flat .npy arrays (see
scripts/dataset/pack_arrow.py). Training auto-selects these memmap stores when packed/ is
present, so the dataset is paged in shared via the OS page cache instead of a full private
copy per DataLoader worker. Shipping packed/ in the dataset means no per-node packing at
train time. Skip-if-exists; CPU-only.

    DATASET_ROOT=/workspace pixi run python3 scripts/marina_db/build/13_pack_arrow.py
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))              # marina_db/ (config)
sys.path.insert(0, str(_HERE.parents[2] / "dataset"))  # scripts/dataset/ (pack_arrow)

from config import DATA_DATASET
from pack_arrow import pack_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Repack shards even if packed output already exists")
    args = ap.parse_args()
    root = str(DATA_DATASET)
    print(f"[13_pack_arrow] packing {root}/arrow -> {root}/packed")
    pack_root(root, force=args.force)


if __name__ == "__main__":
    main()
