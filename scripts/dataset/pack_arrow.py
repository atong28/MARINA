#!/usr/bin/env python3
"""Pack the Parquet Arrow shards into flat, memmap-ready .npy arrays.

The training data pipeline reads each {root}/arrow/{split}/{MOD}.parquet shard
into a Python dict of ndarrays, once per forked DataLoader worker (see
src/modules/data/arrow_store.py). That copy cannot be shared across workers/ranks
and is the dominant RAM term on a multi-GPU node. This script converts every shard
into the flat layout the Memmap* stores read with mmap_mode='r', so the OS page
cache serves one shared mapping to every process.

Output layout, per shard, under {root}/packed/{split}/{MOD}/:
    flat.npy       concatenated row values (float32 tensor / int32 fragidx)
    off.npy        int64 offsets into flat, length R+1
    idx.npy        int64 molecule ids, sorted ascending, length R
    shapeflat.npy  int32 per-row shape tuples concatenated (tensor stores only)
    shapeoff.npy   int64 offsets into shapeflat, length R+1 (tensor stores only)

Skip-if-exists (unless --force). Run once after the Parquet dataset is built:
    python scripts/dataset/pack_arrow.py --root /path/to/data/dataset
"""
import argparse
import glob
import os

import numpy as np
import pyarrow.parquet as pq


def _pack_tensor_shard(parquet_path: str, out_dir: str) -> None:
    table = pq.read_table(parquet_path, columns=["idx", "data", "shape"])
    idx = np.asarray(table["idx"].to_pylist(), dtype=np.int64)
    data = table["data"].to_pylist()
    shape = table["shape"].to_pylist()

    order = np.argsort(idx, kind="stable")
    idx = idx[order]

    flat_parts, off = [], [0]
    sflat_parts, soff = [], [0]
    total, stotal = 0, 0
    for j in order:
        row = np.asarray(data[j], dtype=np.float32)
        flat_parts.append(row)
        total += row.shape[0]
        off.append(total)
        s = np.asarray(shape[j], dtype=np.int32)
        sflat_parts.append(s)
        stotal += s.shape[0]
        soff.append(stotal)

    flat = np.concatenate(flat_parts) if flat_parts else np.zeros(0, np.float32)
    sflat = np.concatenate(sflat_parts) if sflat_parts else np.zeros(0, np.int32)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "flat.npy"), flat)
    np.save(os.path.join(out_dir, "off.npy"), np.asarray(off, dtype=np.int64))
    np.save(os.path.join(out_dir, "idx.npy"), idx)
    np.save(os.path.join(out_dir, "shapeflat.npy"), sflat)
    np.save(os.path.join(out_dir, "shapeoff.npy"), np.asarray(soff, dtype=np.int64))


def _pack_fragidx_shard(parquet_path: str, out_dir: str) -> None:
    table = pq.read_table(parquet_path, columns=["idx", "cols"])
    idx = np.asarray(table["idx"].to_pylist(), dtype=np.int64)
    cols = table["cols"].to_pylist()

    order = np.argsort(idx, kind="stable")
    idx = idx[order]

    flat_parts, off = [], [0]
    total = 0
    for j in order:
        row = np.asarray(cols[j], dtype=np.int32)
        flat_parts.append(row)
        total += row.shape[0]
        off.append(total)

    flat = np.concatenate(flat_parts) if flat_parts else np.zeros(0, np.int32)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "flat.npy"), flat)
    np.save(os.path.join(out_dir, "off.npy"), np.asarray(off, dtype=np.int64))
    np.save(os.path.join(out_dir, "idx.npy"), idx)


def pack_shard(parquet_path: str, out_dir: str, force: bool) -> str:
    if not force and os.path.isfile(os.path.join(out_dir, "flat.npy")):
        return "skip"
    schema_names = pq.ParquetFile(parquet_path).schema_arrow.names
    if "cols" in schema_names:
        _pack_fragidx_shard(parquet_path, out_dir)
    else:
        _pack_tensor_shard(parquet_path, out_dir)
    return "packed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Dataset root containing arrow/{split}/*.parquet")
    ap.add_argument("--force", action="store_true",
                    help="Repack shards even if packed output already exists")
    args = ap.parse_args()

    parquets = sorted(glob.glob(os.path.join(args.root, "arrow", "*", "*.parquet")))
    if not parquets:
        raise SystemExit(f"No parquet shards found under {args.root}/arrow/*/")

    for path in parquets:
        split = os.path.basename(os.path.dirname(path))
        base = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(args.root, "packed", split, base)
        status = pack_shard(path, out_dir, args.force)
        print(f"[{status:6}] {split}/{base} -> {out_dir}")


if __name__ == "__main__":
    main()
