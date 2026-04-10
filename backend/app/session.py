"""
ModelSession: loads a MARINA or SPECTRE model and its associated resources
(fingerprint loader, rankingset, metadata, MW index) from a per-model directory.

Each model directory must contain:
    best.ckpt       – PyTorch Lightning checkpoint
    params.json     – JSON dict of model hyperparameters
    retrieval.pkl   – List[Dict] or Dict[int, Dict] with SMILES entries
    metadata.json   – Dict[str, Dict] keyed by str(index)

Optional (auto-built if absent):
    RankingEntropy/rankingset.pt
    RankingEntropy/bitinfo_to_idx.pkl
"""
from __future__ import annotations

import bisect
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ── Lazy imports from MARINA src ─────────────────────────────────────────────

def _import_marina():
    """Import MARINA model class (deferred until after sys.path is set up)."""
    from src.modules.marina.model import MARINA
    return MARINA


def _import_spectre():
    from src.modules.spectre.model import SPECTRE
    return SPECTRE


def _import_marina_args():
    from src.modules.marina.args import MARINAArgs
    return MARINAArgs


def _import_spectre_args():
    from src.modules.spectre.args import SPECTREArgs
    return SPECTREArgs


def _import_fp_loader():
    from src.modules.data.fp_loader import EntropyFPLoader
    return EntropyFPLoader


# ── Helper: load metadata ────────────────────────────────────────────────────

def _load_json(path: str) -> Any:
    with open(path, "r") as fh:
        return json.load(fh)


def _mw_from_entry(entry: Optional[Dict[str, Any]]) -> Optional[float]:
    if not entry:
        return None
    mw = entry.get("mw")
    if isinstance(mw, (int, float)):
        try:
            return float(mw)
        except Exception:
            pass
    return None


# ── RankingSet wrapper ────────────────────────────────────────────────────────

class RankingSet:
    """
    Thin wrapper around src.modules.core.ranker.RankingSet that exposes
    retrieve_with_scores (scores + indices) and handles sparse CSR row filtering.
    """

    def __init__(self, store: torch.Tensor, metric: str = "cosine") -> None:
        from src.modules.core.ranker import RankingSet as _RS
        self._rs = _RS(store=store, metric=metric)

    @property
    def data(self) -> torch.Tensor:
        return self._rs.data

    def retrieve_with_scores(
        self, query: torch.Tensor, n: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (scores, indices) tensors of shape (n,) for a single query."""
        if query.dim() == 1:
            query = query.unsqueeze(0)
        sims = self._rs._sims(query)          # (N, 1)
        k_eff = min(n, sims.size(0))
        sims_topk, idxs_topk = torch.topk(sims, k=k_eff, dim=0)
        return sims_topk.squeeze(1), idxs_topk.squeeze(1)


def _filter_csr_rows(store: torch.Tensor, kept: List[int]) -> torch.Tensor:
    """Return a new CSR tensor containing only the rows in `kept`."""
    if not kept:
        return torch.sparse_csr_tensor(
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.float32),
            size=(0, store.shape[1]),
        )
    if store.layout != torch.sparse_csr:
        return store[kept]

    crow = store.crow_indices()
    col  = store.col_indices()
    val  = store.values()
    new_crow, new_cols, new_vals = [0], [], []
    nnz = 0
    for idx in kept:
        s, e = int(crow[idx]), int(crow[idx + 1])
        if e > s:
            new_cols.append(col[s:e])
            new_vals.append(val[s:e])
            nnz += e - s
        new_crow.append(nnz)

    if nnz == 0:
        return torch.sparse_csr_tensor(
            torch.tensor(new_crow, dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.float32),
            size=(len(kept), store.shape[1]),
        )
    return torch.sparse_csr_tensor(
        torch.tensor(new_crow, dtype=torch.int64),
        torch.cat(new_cols).to(torch.int64),
        torch.cat(new_vals).to(torch.float32),
        size=(len(kept), store.shape[1]),
    )


# ── ModelSession ──────────────────────────────────────────────────────────────

@dataclass
class ModelSession:
    """Encapsulates a loaded model with all associated retrieval resources."""

    model_type: str                    # "marina" or "spectre"
    model: torch.nn.Module
    fp_loader: Any                     # EntropyFPLoader
    fp_type: str
    model_root: str
    device: torch.device
    _metadata: Dict[str, Any]

    # Lazily built retrieval index
    _rankingset_store:   Optional[torch.Tensor]          = field(default=None, repr=False)
    _rankingset_wrapper: Optional[RankingSet]            = field(default=None, repr=False)
    _filtered_cache:     Dict[Tuple, RankingSet]         = field(default_factory=dict, repr=False)
    _mw_sorted:          Optional[List[Tuple[float,int]]] = field(default=None, repr=False)
    _mw_by_idx:          Optional[Dict[int, float]]      = field(default=None, repr=False)
    _num_rows:           Optional[int]                   = field(default=None, repr=False)
    _lock:               threading.RLock                 = field(default_factory=threading.RLock, repr=False)

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_model_root(
        cls,
        model_root: str,
        model_type: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> "ModelSession":
        from app.marina_import import ensure_marina_importable
        ensure_marina_importable()

        if device is None:
            from app.config import DEVICE
            device = torch.device(DEVICE)

        model_type = (model_type or "marina").lower()

        params_path    = os.path.join(model_root, "params.json")
        ckpt_path      = os.path.join(model_root, "best.ckpt")
        metadata_path  = os.path.join(model_root, "metadata.json")
        retrieval_path = os.path.join(model_root, "retrieval.pkl")

        params = _load_json(params_path)

        if model_type == "spectre":
            Args  = _import_spectre_args()
            Model = _import_spectre()
        else:
            Args  = _import_marina_args()
            Model = _import_marina()

        args = Args(**params)

        EntropyFPLoader = _import_fp_loader()
        fp_loader = EntropyFPLoader(dataset_root=model_root, retrieval_path=retrieval_path)
        fp_loader.setup(args.out_dim, 6, retrieval_path=retrieval_path)

        model = Model(args, fp_loader)

        logger.debug("Loading checkpoint from %s → %s", ckpt_path, device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        torch.set_grad_enabled(False)

        metadata = _load_json(metadata_path)

        return cls(
            model_type=model_type,
            model=model,
            fp_loader=fp_loader,
            fp_type=getattr(args, "fp_type", "RankingEntropy"),
            model_root=model_root,
            device=device,
            _metadata=metadata,
        )

    # ── Metadata helpers ─────────────────────────────────────────────────────

    def get_entry(self, idx: int) -> Optional[Dict[str, Any]]:
        return self._metadata.get(str(idx))

    def get_smiles(self, idx: int) -> Optional[str]:
        entry = self.get_entry(idx)
        if not entry:
            return None
        smi = entry.get("canonical_3d_smiles")
        if smi and smi != "N/A":
            return smi
        return entry.get("canonical_2d_smiles")

    def get_exact_mass(self, idx: int) -> Optional[float]:
        return _mw_from_entry(self.get_entry(idx))

    # ── MW index & filtering ─────────────────────────────────────────────────

    def _ensure_mw_index(self) -> None:
        with self._lock:
            if self._mw_sorted is not None:
                return
            store = self.fp_loader.load_rankingset(self.fp_type)
            self._rankingset_store = store
            n = store.shape[0]
            self._num_rows = n
            mw_by_idx: Dict[int, float] = {}
            for i in range(n):
                mw = _mw_from_entry(self._metadata.get(str(i)))
                if mw is not None:
                    mw_by_idx[i] = mw
            self._mw_by_idx = mw_by_idx
            self._mw_sorted = sorted((mw, idx) for idx, mw in mw_by_idx.items())

    def indices_in_mw_range(
        self,
        mw_min: Optional[float],
        mw_max: Optional[float],
    ) -> List[int]:
        """Return all retrieval indices whose MW is within [mw_min, mw_max]."""
        self._ensure_mw_index()
        assert self._num_rows is not None

        if mw_min is None and mw_max is None:
            return list(range(self._num_rows))

        assert self._mw_sorted is not None and self._mw_by_idx is not None
        if not self._mw_sorted:
            return list(range(self._num_rows))

        mw_vals = [t[0] for t in self._mw_sorted]
        lo = bisect.bisect_left(mw_vals, mw_min)  if mw_min is not None else 0
        hi = bisect.bisect_right(mw_vals, mw_max) if mw_max is not None else len(mw_vals)
        in_range = {self._mw_sorted[i][1] for i in range(lo, hi)}
        no_mw    = {i for i in range(self._num_rows) if i not in self._mw_by_idx}
        return sorted(in_range | no_mw)

    # ── RankingSet access ────────────────────────────────────────────────────

    def get_rankingset(self) -> RankingSet:
        """Lazily load the full retrieval rankingset."""
        if self._rankingset_store is None:
            self._ensure_mw_index()
        if self._rankingset_wrapper is None:
            assert self._rankingset_store is not None
            self._rankingset_wrapper = RankingSet(store=self._rankingset_store, metric="cosine")
        return self._rankingset_wrapper

    def get_filtered_rankingset(
        self,
        mw_min: Optional[float],
        mw_max: Optional[float],
    ) -> Tuple[RankingSet, List[int]]:
        """
        Return a (RankingSet, kept_indices) pair filtered by MW range.
        The kept_indices list maps local row positions back to global indices.
        Results are cached per (mw_min, mw_max) key.
        """
        key = (mw_min, mw_max)
        with self._lock:
            if key in self._filtered_cache:
                kept = self.indices_in_mw_range(mw_min, mw_max)
                return self._filtered_cache[key], kept

            kept  = self.indices_in_mw_range(mw_min, mw_max)
            base  = self.get_rankingset().data
            store = _filter_csr_rows(base, kept)
            rs    = RankingSet(store=store, metric="cosine")
            self._filtered_cache[key] = rs
            return rs, kept

    # ── Fingerprint helpers ──────────────────────────────────────────────────

    def fp_indices_for_smiles(self, smiles: str) -> Optional[List[int]]:
        """Return sorted list of active fingerprint bit indices for a SMILES string."""
        from src.modules.data.fp_utils import count_circular_substructures
        try:
            present = count_circular_substructures(smiles, self.fp_loader.max_radius)
            indices = sorted(
                col
                for b, col in (
                    (b, self.fp_loader.bitinfo_to_fp_index_map.get(b))
                    for b in present
                )
                if col is not None
            )
            return indices
        except Exception as exc:
            logger.warning("fp_indices_for_smiles failed for %r: %s", smiles, exc)
            return None

    def retrieved_fp_indices(self, global_idx: int) -> List[int]:
        """Return the non-zero column indices for a retrieval entry's fingerprint."""
        store = self.get_rankingset().data
        if store.layout == torch.sparse_csr:
            crow = store.crow_indices()
            col  = store.col_indices()
            s, e = int(crow[global_idx]), int(crow[global_idx + 1])
            return col[s:e].cpu().tolist() if e > s else []
        row = store[global_idx].cpu()
        nz  = torch.nonzero(row > 0.5, as_tuple=False).squeeze(-1)
        return nz.tolist() if nz.numel() else []
