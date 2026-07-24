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
