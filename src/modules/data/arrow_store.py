import os
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import torch


@dataclass
class ArrowTensorStore:
    """
    Read-only tensor store backed by a Parquet file with schema:
      - idx: int64
      - data: list<primitive>
      - shape: list<int32>
    """

    path: str

    def __post_init__(self) -> None:
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"Arrow file not found: {self.path}")
        self._pid = None
        self._by_idx = None

    def _ensure_loaded(self) -> None:
        pid = os.getpid()
        if self._by_idx is not None and self._pid == pid:
            return

        table = pq.read_table(self.path, columns=["idx", "data", "shape"])
        idx_values = table["idx"].to_pylist()
        data_values = table["data"].to_pylist()
        shape_values = table["shape"].to_pylist()

        # Pre-convert each row to a contiguous float32 ndarray once, so the hot
        # path (get_tensor) is a cheap copy instead of a per-item Python-list ->
        # numpy conversion. This also lowers memory vs. keeping Python lists.
        by_idx = {}
        for idx, data, shape in zip(idx_values, data_values, shape_values):
            arr = np.asarray(data, dtype=np.float32)
            if shape:
                arr = arr.reshape(tuple(int(v) for v in shape))
            by_idx[int(idx)] = arr
        self._by_idx = by_idx
        self._pid = pid

    def get_tensor(self, idx: int, dtype: torch.dtype | None = None) -> torch.Tensor:
        self._ensure_loaded()
        arr = self._by_idx[int(idx)]
        # Copy: callers mutate the returned tensor in place (jittering/augment).
        tensor = torch.from_numpy(arr.copy())
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor


@dataclass
class ArrowFragIdxStore:
    """
    Read-only FragIdx store backed by a Parquet file with schema:
      - idx: int64
      - cols: list<int32>
    """

    path: str

    def __post_init__(self) -> None:
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"Arrow file not found: {self.path}")
        self._pid = None
        self._by_idx = None

    def _ensure_loaded(self) -> None:
        pid = os.getpid()
        if self._by_idx is not None and self._pid == pid:
            return

        table = pq.read_table(self.path, columns=["idx", "cols"])
        idx_values = table["idx"].to_pylist()
        cols_values = table["cols"].to_pylist()
        # Pre-convert to int32 ndarrays once (get_indices is called per sample).
        self._by_idx = {
            int(idx): np.asarray(cols, dtype=np.int32)
            for idx, cols in zip(idx_values, cols_values)
        }
        self._pid = pid

    def get_indices(self, idx: int) -> np.ndarray:
        self._ensure_loaded()
        cols = self._by_idx.get(int(idx))
        if cols is None:
            raise KeyError(f"Key {idx} not found in {self.path}")
        return cols


# --------------------------------------------------------------------------- #
# Memmap-backed stores
#
# The ArrowTensorStore/ArrowFragIdxStore above read a whole Parquet shard into a
# Python dict of ndarrays. Because that load is pid-gated (lazy), every forked
# DataLoader worker builds its own full copy that cannot be shared -- the dominant
# RAM term on a 4-GPU node with many persistent workers.
#
# The Memmap* variants read from flat .npy arrays produced by
# scripts/dataset/pack_arrow.py and open every array with mmap_mode='r', so the OS
# page cache serves one shared read-only mapping to all workers and DDP ranks on a
# node. Layout per store (packed_dir):
#     flat.npy       concatenated row values (float32 tensor / int32 fragidx)
#     off.npy        int64 offsets into flat, length R+1
#     idx.npy        int64 molecule ids, sorted ascending, length R
#     shapeflat.npy  int32 per-row shape tuples concatenated (tensor stores only)
#     shapeoff.npy   int64 offsets into shapeflat, length R+1 (tensor stores only)
# A molecule id -> row lookup is np.searchsorted over the sorted idx array.
# --------------------------------------------------------------------------- #


def packed_dir_for(parquet_path: str) -> str:
    """Map {root}/arrow/{split}/{MOD}.parquet -> {root}/packed/{split}/{MOD}."""
    split_dir = os.path.dirname(parquet_path)                 # {root}/arrow/{split}
    base = os.path.splitext(os.path.basename(parquet_path))[0]
    split = os.path.basename(split_dir)
    root = os.path.dirname(os.path.dirname(split_dir))        # {root}
    return os.path.join(root, "packed", split, base)


def _packed_ready(packed_dir: str) -> bool:
    return os.path.isfile(os.path.join(packed_dir, "flat.npy"))


@dataclass
class MemmapTensorStore:
    """Memmap-backed drop-in for ArrowTensorStore. Same public API (get_tensor)."""

    packed_dir: str

    def __post_init__(self) -> None:
        if not _packed_ready(self.packed_dir):
            raise FileNotFoundError(f"Packed store not found: {self.packed_dir}")
        self._pid = None
        self._flat = self._off = self._idx = self._sflat = self._soff = None

    def _ensure_loaded(self) -> None:
        # Reopen memmaps per process. The mapped data pages are shared via the OS
        # page cache regardless of which process opened them, so this is cheap.
        pid = os.getpid()
        if self._flat is not None and self._pid == pid:
            return
        d = self.packed_dir
        self._flat = np.load(os.path.join(d, "flat.npy"), mmap_mode="r")
        self._off = np.load(os.path.join(d, "off.npy"), mmap_mode="r")
        self._idx = np.load(os.path.join(d, "idx.npy"), mmap_mode="r")
        self._sflat = np.load(os.path.join(d, "shapeflat.npy"), mmap_mode="r")
        self._soff = np.load(os.path.join(d, "shapeoff.npy"), mmap_mode="r")
        self._pid = pid

    def _row(self, idx: int) -> int:
        r = int(np.searchsorted(self._idx, idx))
        if r >= self._idx.shape[0] or int(self._idx[r]) != idx:
            raise KeyError(f"Key {idx} not found in {self.packed_dir}")
        return r

    def get_tensor(self, idx: int, dtype: torch.dtype | None = None) -> torch.Tensor:
        self._ensure_loaded()
        r = self._row(int(idx))
        a, b = int(self._off[r]), int(self._off[r + 1])
        # .copy() materializes only this one row out of the memmap into a writable
        # buffer (callers jitter/augment in place). The rest stays mapped/shared.
        flat = np.asarray(self._flat[a:b]).copy()
        sa, sb = int(self._soff[r]), int(self._soff[r + 1])
        shape = tuple(int(v) for v in self._sflat[sa:sb])
        arr = flat.reshape(shape)
        tensor = torch.from_numpy(arr)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor


@dataclass
class MemmapFragIdxStore:
    """Memmap-backed drop-in for ArrowFragIdxStore. Same public API (get_indices)."""

    packed_dir: str

    def __post_init__(self) -> None:
        if not _packed_ready(self.packed_dir):
            raise FileNotFoundError(f"Packed store not found: {self.packed_dir}")
        self._pid = None
        self._flat = self._off = self._idx = None

    def _ensure_loaded(self) -> None:
        pid = os.getpid()
        if self._flat is not None and self._pid == pid:
            return
        d = self.packed_dir
        self._flat = np.load(os.path.join(d, "flat.npy"), mmap_mode="r")
        self._off = np.load(os.path.join(d, "off.npy"), mmap_mode="r")
        self._idx = np.load(os.path.join(d, "idx.npy"), mmap_mode="r")
        self._pid = pid

    def get_indices(self, idx: int) -> np.ndarray:
        self._ensure_loaded()
        r = int(np.searchsorted(self._idx, int(idx)))
        if r >= self._idx.shape[0] or int(self._idx[r]) != int(idx):
            raise KeyError(f"Key {idx} not found in {self.packed_dir}")
        a, b = int(self._off[r]), int(self._off[r + 1])
        return np.asarray(self._flat[a:b]).copy()


def open_tensor_store(parquet_path: str):
    """Return a memmap-backed store if a packed sibling exists, else the Parquet store."""
    packed = packed_dir_for(parquet_path)
    if _packed_ready(packed):
        return MemmapTensorStore(packed)
    return ArrowTensorStore(parquet_path)


def open_fragidx_store(parquet_path: str):
    """Return a memmap-backed store if a packed sibling exists, else the Parquet store."""
    packed = packed_dir_for(parquet_path)
    if _packed_ready(packed):
        return MemmapFragIdxStore(packed)
    return ArrowFragIdxStore(parquet_path)
