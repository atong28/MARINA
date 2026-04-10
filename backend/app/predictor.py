"""
Inference logic: convert raw spectral inputs → predicted fingerprint → top-k retrieval.

This module is also imported inside worker processes (compute_pool.py), so it
must not import anything from the FastAPI layer at module level.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_forward_lock = threading.Lock()

# Canonical order matching src/modules/core/const.py INPUTS_CANONICAL_ORDER
_INPUTS_ORDER = ["hsqc", "c_nmr", "h_nmr", "mass_spec", "mw"]


# ── Input preprocessing ───────────────────────────────────────────────────────

def preprocess_inputs(raw: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Convert a raw dict (lists / scalars) to tensors in the shape the model expects.

    hsqc:      flat list [H1,C1,I1,...] → (N,3) with axes reordered to [C,H,I]
    h_nmr:     flat list [ppm,...]      → (N,1)
    c_nmr:     flat list [ppm,...]      → (N,1)
    mass_spec: flat list [mz,I,...]     → (N,2)
    mw:        scalar                   → (1,1)
    """
    out: Dict[str, torch.Tensor] = {}
    for mod, v in raw.items():
        if mod == "mw":
            out["mw"] = torch.tensor([[float(v)]], dtype=torch.float32)
            continue
        if not isinstance(v, (list, tuple)):
            continue
        t = torch.tensor(v, dtype=torch.float32)
        if mod == "hsqc":
            if t.numel() % 3 != 0:
                logger.warning("HSQC length %d not divisible by 3 – skipping", t.numel())
                continue
            t = t.view(-1, 3)[:, [1, 0, 2]]   # reorder H,C,I → C,H,I
        elif mod in ("h_nmr", "c_nmr"):
            t = t.view(-1, 1)
        elif mod == "mass_spec":
            if t.numel() % 2 != 0:
                logger.warning("mass_spec length %d not divisible by 2 – skipping", t.numel())
                continue
            t = t.view(-1, 2)
        out[mod] = t
    return out


def _collate_marina(processed: Dict[str, torch.Tensor], out_dim: int):
    """Collate a single sample into the padded batch format MARINA expects."""
    dummy_fp = torch.zeros(out_dim, dtype=torch.float32)
    batch_inputs: Dict[str, torch.Tensor] = {}
    for mod in _INPUTS_ORDER:
        seq = processed.get(mod)
        if seq is None:
            continue
        if seq.ndim == 1:
            seq = seq.unsqueeze(0)
        batch_inputs[mod] = seq.unsqueeze(0)   # (1, N, D)
    return batch_inputs, dummy_fp.unsqueeze(0)


def _collate_spectre(processed: Dict[str, torch.Tensor]):
    """
    Convert MARINA-style modality dict into SPECTRE's (inputs, type_indicator).

    Type codes:
        0: HSQC   1: C NMR   2: H NMR   3: MW   4: Mass Spec
    """
    segments: List[torch.Tensor] = []
    types:    List[int]          = []

    hsqc = processed.get("hsqc")
    if hsqc is not None and hsqc.numel() > 0:
        segments.append(hsqc)
        types.extend([0] * hsqc.shape[0])

    c_nmr = processed.get("c_nmr")
    if c_nmr is not None and c_nmr.numel() > 0:
        padded = torch.nn.functional.pad(c_nmr, (0, 2))
        segments.append(padded)
        types.extend([1] * padded.shape[0])

    h_nmr = processed.get("h_nmr")
    if h_nmr is not None and h_nmr.numel() > 0:
        padded = torch.nn.functional.pad(h_nmr, (1, 1))
        segments.append(padded)
        types.extend([2] * padded.shape[0])

    mass_spec = processed.get("mass_spec")
    if mass_spec is not None and mass_spec.numel() > 0:
        padded = torch.nn.functional.pad(mass_spec, (0, 1))
        segments.append(padded)
        types.extend([4] * padded.shape[0])

    mw = processed.get("mw")
    if mw is not None and mw.numel() > 0:
        mw_val = float(mw.view(-1)[0])
        segments.append(torch.tensor([[mw_val, 0.0, 0.0]]))
        types.append(3)

    if not segments:
        raise ValueError("No valid spectral inputs for SPECTRE model")

    stacked = torch.vstack(segments).unsqueeze(0)              # (1, N, 3)
    type_t  = torch.tensor(types, dtype=torch.long).unsqueeze(0)  # (1, N)
    return stacked, type_t


# ── Model forward pass ────────────────────────────────────────────────────────

def run_model(session: Any, processed: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Run the model forward pass and return the raw (pre-sigmoid) prediction vector.
    Thread-safe via a global lock (one forward pass at a time per process).
    """
    device = session.device
    processed = {k: v.to(device) for k, v in processed.items()}

    with _forward_lock, torch.no_grad():
        if session.model_type == "spectre":
            inputs, type_indicator = _collate_spectre(processed)
            out = session.model(inputs.to(device), type_indicator.to(device))
        else:
            batch_inputs, _ = _collate_marina(processed, session.fp_loader.out_dim)
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            out = session.model(batch_inputs)

        return out.detach().cpu()[0]   # first (and only) batch element


# ── Top-k retrieval ───────────────────────────────────────────────────────────

def retrieve_top_k(
    session: Any,
    pred_logits: torch.Tensor,
    k: int,
    mw_min: Optional[float] = None,
    mw_max: Optional[float] = None,
) -> Tuple[List[float], List[int], List[float]]:
    """
    Apply sigmoid, then retrieve top-k from the (optionally MW-filtered) rankingset.

    Returns:
        scores       – cosine similarities, length ≤ k
        global_idxs  – corresponding database indices
        pred_prob    – sigmoid of pred_logits as a Python list
    """
    pred = torch.sigmoid(pred_logits)
    pred_prob = pred.tolist()

    rs, kept = session.get_filtered_rankingset(mw_min, mw_max)
    n = min(k, len(kept))
    if n == 0:
        return [], [], pred_prob

    sims, local_idxs = rs.retrieve_with_scores(pred.unsqueeze(0), n=n)
    sims_list  = sims.tolist()        if isinstance(sims.tolist(), list)  else [sims.tolist()]
    local_list = local_idxs.tolist()  if isinstance(local_idxs.tolist(), list) else [local_idxs.tolist()]

    global_idxs = [int(kept[li]) for li in local_list]
    return [float(s) for s in sims_list], global_idxs, pred_prob


# ── High-level API ────────────────────────────────────────────────────────────

def predict_from_raw(
    raw_inputs: Dict[str, Any],
    k: int = 10,
    model_id: Optional[str] = None,
    mw_min: Optional[float] = None,
    mw_max: Optional[float] = None,
) -> Tuple[List[float], List[int], List[float]]:
    """
    Full pipeline: raw dict → preprocess → model forward → top-k retrieval.
    Returns (scores, global_indices, pred_fp_as_list).
    """
    from app.registry import ensure_loaded
    session = ensure_loaded(model_id or _default_model_id())

    processed   = preprocess_inputs(raw_inputs)
    pred_logits = run_model(session, processed)
    return retrieve_top_k(session, pred_logits, k, mw_min=mw_min, mw_max=mw_max)


def _default_model_id() -> str:
    from app.manifest import get_default_model_id
    return get_default_model_id()
