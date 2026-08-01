"""
Similarity metrics shared by the result builder and the custom-SMILES route.

Both operate on equal-length vectors. Callers are responsible for checking
lengths first — these helpers deliberately do not truncate to the shorter of
the two, because a length mismatch means the caller compared fingerprints from
different models and the resulting score would be meaningless.
"""
from __future__ import annotations

import math

import torch


def _prepare(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = a.detach().float().view(-1)
    b = b.detach().float().view(-1)
    if a.numel() != b.numel():
        raise ValueError(
            f"fingerprint length mismatch: {a.numel()} vs {b.numel()}"
        )
    return a, b


def _clamp01(val: float) -> float:
    return max(0.0, min(1.0, val)) if math.isfinite(val) else 0.0


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity, clamped to [0, 1]."""
    a, b = _prepare(a, b)
    if a.numel() == 0:
        return 0.0
    denom = float(torch.linalg.norm(a)) * float(torch.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return _clamp01(float(torch.dot(a, b)) / denom)


def tanimoto(a: torch.Tensor, b: torch.Tensor) -> float:
    """Generalised (real-valued) Tanimoto similarity, clamped to [0, 1]."""
    a, b = _prepare(a, b)
    if a.numel() == 0:
        return 0.0
    dot   = torch.dot(a, b)
    denom = a.pow(2).sum() + b.pow(2).sum() - dot
    if denom <= 0:
        return 0.0
    return _clamp01((dot / denom).item())
