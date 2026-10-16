#!/usr/bin/env python
"""Drop Mnova "no prediction" sentinel peaks from a built MARINA-DB.

Mnova lists F and P atoms inside its 13C / 1H shift lists with the value -100000.0 (no prediction); the
MARINA-DB stage-6 reader took them as real peaks, so ~0.5% of molecules ship with -100000 ppm peaks in
C_NMR / H_NMR / HSQC_NMR, and MARINA2.0's 2D builds carry them into HMBC_NMR / COSY_NMR. This removes every
peak with any coordinate <= MNOVA_SENTINEL_FLOOR. A spectrum left with no peaks is dropped and its index.pkl
has_* flag cleared. Shards are replaced by rename, never rewritten in place: MARINA-DB-OPEN-2D hard-links its 1D
shards to MARINA-DB-OPEN-OLD, and an in-place write would change both trees.

Run on a dataset root (arrow/ + index.pkl), then repack:  pack_arrow.py --root <root> --force
build_open.py calls drop_sentinels() itself before collapse + pack.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from chnmr import MNOVA_SENTINEL_FLOOR  # noqa: E402

SPLITS = ("train", "val", "test")
SHARDS = {"C_NMR": "has_c_nmr", "H_NMR": "has_h_nmr", "HSQC_NMR": "has_hsqc",
          "HMBC_NMR": "has_hmbc", "COSY_NMR": "has_cosy"}


def drop_sentinels(root):
    root = Path(root)
    index = pickle.load(open(root / "index.pkl", "rb"))
    stats = {}
    for shard, flag in SHARDS.items():
        if not (root / "arrow" / SPLITS[0] / f"{shard}.parquet").exists():
            continue
        st = stats[shard] = {"peaks_dropped": 0, "molecules_changed": 0, "spectra_emptied": 0}
        for sp in SPLITS:
            path = root / "arrow" / sp / f"{shard}.parquet"
            tab = pq.read_table(path)
            data = tab.column("data").to_pylist()
            shape = tab.column("shape").to_pylist()
            keep_row = np.ones(len(data), bool)
            changed = False
            for i, (d, s) in enumerate(zip(data, shape)):
                a = np.asarray(d, dtype=np.float64).reshape(s)
                bad = (a <= MNOVA_SENTINEL_FLOOR).reshape(a.shape[0], -1).any(1)
                if not bad.any():
                    continue
                st["peaks_dropped"] += int(bad.sum())
                st["molecules_changed"] += 1
                changed = True
                a = a[~bad]
                if a.shape[0] == 0:
                    keep_row[i] = False
                    st["spectra_emptied"] += 1
                    index[tab.column("idx")[i].as_py()][flag] = False
                data[i], shape[i] = a.reshape(-1).tolist(), list(a.shape)
            if changed:
                new = pa.table({"idx": tab.column("idx"), "data": pa.array(data, tab.schema.field("data").type),
                                "shape": pa.array(shape, tab.schema.field("shape").type)}, schema=tab.schema)
                tmp = path.with_suffix(".parquet.tmp")
                pq.write_table(new.filter(pa.array(keep_row)), tmp)
                os.replace(tmp, path)
    tmp = root / "index.pkl.tmp"
    pickle.dump(index, open(tmp, "wb"))
    os.replace(tmp, root / "index.pkl")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root with arrow/{split}/*.parquet and index.pkl")
    stats = drop_sentinels(ap.parse_args().root)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
