"""
Assemble ResultCard dicts from retrieval (global_idx, score) pairs.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _clamp(v: Any) -> float:
    try:
        x = float(v)
        return 0.0 if not math.isfinite(x) else max(0.0, min(1.0, x))
    except (TypeError, ValueError):
        return 0.0


def _tanimoto(a: torch.Tensor, b: torch.Tensor) -> float:
    """Generalised Tanimoto similarity for real-valued vectors."""
    a = a.detach().float().view(-1)
    b = b.detach().float().view(-1)
    n = min(a.numel(), b.numel())
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    dot   = torch.dot(a, b)
    denom = a.pow(2).sum() + b.pow(2).sum() - dot
    if denom <= 0:
        return 0.0
    val = (dot / denom).item()
    return max(0.0, min(1.0, val)) if math.isfinite(val) else 0.0


def _dense_from_indices(length: int, indices: List[int]) -> torch.Tensor:
    vec = torch.zeros(length, dtype=torch.float32)
    for j in indices:
        if 0 <= j < length:
            vec[j] = 1.0
    return vec


def _exact_mass(smiles: str, entry: Dict[str, Any]) -> Optional[float]:
    mw = entry.get("mw")
    if isinstance(mw, (int, float)):
        try:
            return float(mw)
        except Exception:
            pass
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Descriptors.ExactMolWt(mol)
    except Exception:
        pass
    return None


def _database_links(entry: Dict[str, Any]) -> Dict[str, Optional[str]]:
    links: Dict[str, Optional[str]] = {"coconut": None, "lotus": None, "npmrd": None}
    coconut = entry.get("coconut") or {}
    lotus   = entry.get("lotus")   or {}
    npmrd   = entry.get("npmrd")   or {}
    if cid := coconut.get("coconut_id"):
        links["coconut"] = f"https://coconut.naturalproducts.net/compounds/{cid}"
    if lid := lotus.get("lotus_id"):
        links["lotus"] = f"https://lotus.naturalproducts.net/compound/lotus_id/{lid}"
    if nid := npmrd.get("npmrd_id"):
        links["npmrd"] = f"https://np-mrd.org/natural_products/{nid}"
    return links


def _primary(entry: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return (name, link) for the most authoritative database."""
    for db, url_template, id_key in [
        ("coconut", "https://coconut.naturalproducts.net/compounds/{}", "coconut_id"),
        ("lotus",   "https://lotus.naturalproducts.net/compound/lotus_id/{}", "lotus_id"),
        ("npmrd",   "https://np-mrd.org/natural_products/{}", "npmrd_id"),
    ]:
        rec = entry.get(db) or {}
        db_id = rec.get(id_key)
        if db_id:
            return rec.get("name"), url_template.format(db_id)
    return None, None


def build_result_cards(
    session: Any,
    pairs: List[Tuple[int, float]],   # (global_idx, cosine_score)
    pred_tensor: torch.Tensor,
    img_size: int = 400,
    max_cards: Optional[int] = None,
) -> List[dict]:
    """
    Build a list of ResultCard dicts from (global_idx, score) pairs.
    """
    from app.renderer import render_plain_svg, render_enhanced_svg

    cards: List[dict] = []
    for global_idx, cosine_score in pairs:
        if max_cards is not None and len(cards) >= max_cards:
            break

        entry = session.get_entry(global_idx)
        if not entry:
            continue
        smiles = session.get_smiles(global_idx)
        if not smiles:
            continue

        cosine_sim  = _clamp(cosine_score)
        fp_indices  = session.retrieved_fp_indices(global_idx)
        tanimoto_sim = 0.0
        if fp_indices and pred_tensor is not None:
            retrieved_vec = _dense_from_indices(pred_tensor.numel(), fp_indices)
            tanimoto_sim  = _tanimoto(pred_tensor, retrieved_vec)

        fp_loader = session.fp_loader
        enhanced_svg = render_enhanced_svg(smiles, pred_tensor, fp_loader, img_size=img_size)
        plain_svg    = render_plain_svg(smiles, img_size=img_size)

        name, primary_link = _primary(entry)
        db_links           = _database_links(entry)
        mass               = _exact_mass(smiles, entry)

        cards.append({
            "index":                         global_idx,
            "smiles":                        smiles,
            "similarity":                    cosine_sim,
            "cosine_similarity":             cosine_sim,
            "tanimoto_similarity":           tanimoto_sim,
            "svg":                           enhanced_svg,
            "plain_svg":                     plain_svg,
            "name":                          name,
            "primary_link":                  primary_link,
            "database_links":                db_links,
            "retrieved_molecule_fp_indices": fp_indices,
            "exact_mass":                    mass,
        })
    return cards
