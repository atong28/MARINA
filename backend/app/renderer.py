"""
Molecule rendering via RDKit.

All methods return None gracefully when RDKit is unavailable or rendering fails,
so callers never need to catch exceptions from this module.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_rdkit_available: Optional[bool] = None
_rdkit_lock = threading.Lock()


def _check_rdkit() -> bool:
    global _rdkit_available
    if _rdkit_available is None:
        with _rdkit_lock:
            if _rdkit_available is None:
                try:
                    from rdkit import Chem  # noqa: F401
                    _rdkit_available = True
                except ImportError:
                    _rdkit_available = False
                    logger.warning("RDKit not available – molecule rendering disabled")
    return _rdkit_available


def render_plain_svg(smiles: str, img_size: int = 300) -> Optional[str]:
    """Render a plain SVG for a SMILES string (no fingerprint highlighting)."""
    if not _check_rdkit():
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        half = img_size // 2
        return Draw.MolToSVG(mol, width=half, height=half)
    except Exception as exc:
        logger.debug("render_plain_svg failed for %r: %s", smiles, exc)
        return None


def render_enhanced_svg(
    smiles: str,
    predicted_fp: torch.Tensor,
    fp_loader: object,
    img_size: int = 400,
) -> Optional[str]:
    """
    Render an SVG with fingerprint-based atom highlighting using the
    similarity map approach from RDKit SimilarityMaps.
    Falls back to plain SVG on any error.
    """
    if not _check_rdkit():
        return render_plain_svg(smiles, img_size)
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Compute per-atom weights from fingerprint similarity
        weights = _atom_weights(mol, predicted_fp, fp_loader)
        if weights is None:
            return render_plain_svg(smiles, img_size)

        drawer = rdMolDraw2D.MolDraw2DSVG(img_size, img_size)
        from rdkit.Chem.Draw import SimilarityMaps
        SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, draw2d=drawer)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as exc:
        logger.debug("render_enhanced_svg failed for %r: %s", smiles, exc)
        return render_plain_svg(smiles, img_size)


def _atom_weights(mol, predicted_fp: torch.Tensor, fp_loader: object):
    """Compute per-atom contribution weights based on the predicted fingerprint."""
    try:
        from app.marina_import import ensure_marina_importable
        ensure_marina_importable()
        from src.modules.data.fp_utils import get_bitinfos

        max_radius = getattr(fp_loader, "max_radius", 6) or 6
        bitinfo_map = getattr(fp_loader, "bitinfo_to_fp_index_map", {})
        smiles = mol.GetAtomWithIdx(0).GetOwningMol() if hasattr(mol, 'GetAtomWithIdx') else None

        from rdkit import Chem
        smi = Chem.MolToSmiles(mol)
        atom_to_bits, _ = get_bitinfos(smi, max_radius)
        if atom_to_bits is None:
            return None

        pred = predicted_fp.detach().float().cpu()
        weights = []
        for atom_idx in range(mol.GetNumAtoms()):
            bits = atom_to_bits.get(atom_idx, [])
            score = 0.0
            for b in bits:
                col = bitinfo_map.get(b)
                if col is not None and 0 <= col < pred.numel():
                    score = max(score, float(pred[col]))
            weights.append(score)
        return weights
    except Exception as exc:
        logger.debug("_atom_weights failed: %s", exc)
        return None
