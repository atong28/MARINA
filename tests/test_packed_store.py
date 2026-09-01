"""Verify the memmap-backed stores return exactly what the Parquet stores do.

Packs the real val Arrow shards into a temporary packed layout and asserts every
row matches, for both the tensor stores (idx/data/shape) and the FragIdx stores
(idx/cols). Skips if the local dataset is not present.
"""
import os
import sys

import numpy as np
import pytest
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.data.arrow_store import (  # noqa: E402
    ArrowTensorStore, ArrowFragIdxStore, MemmapTensorStore, MemmapFragIdxStore,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "dataset"))
from pack_arrow import pack_shard  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VAL_ARROW = os.path.join(REPO, "data", "dataset", "arrow", "val")


def _val_parquets():
    if not os.path.isdir(VAL_ARROW):
        return []
    return sorted(p for p in os.listdir(VAL_ARROW) if p.endswith(".parquet"))


@pytest.mark.skipif(not _val_parquets(), reason="local val dataset not present")
@pytest.mark.parametrize("fname", _val_parquets())
def test_packed_matches_parquet(fname, tmp_path):
    parquet_path = os.path.join(VAL_ARROW, fname)
    out_dir = str(tmp_path / "packed")
    pack_shard(parquet_path, out_dir, force=True)

    is_fragidx = "cols" in pq.ParquetFile(parquet_path).schema_arrow.names
    all_idx = pq.read_table(parquet_path, columns=["idx"])["idx"].to_pylist()

    if is_fragidx:
        ref, mm = ArrowFragIdxStore(parquet_path), MemmapFragIdxStore(out_dir)
        for i in all_idx:
            np.testing.assert_array_equal(ref.get_indices(i), mm.get_indices(i))
    else:
        ref, mm = ArrowTensorStore(parquet_path), MemmapTensorStore(out_dir)
        for i in all_idx:
            a = ref.get_tensor(i)
            b = mm.get_tensor(i)
            assert a.shape == b.shape, f"idx {i}: {a.shape} != {b.shape}"
            np.testing.assert_array_equal(a.numpy(), b.numpy())

    # A missing key must raise on both.
    missing = max(all_idx) + 1
    with pytest.raises(KeyError):
        (mm.get_indices if is_fragidx else mm.get_tensor)(missing)
