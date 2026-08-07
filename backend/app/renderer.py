"""
Molecule rendering via RDKit.

All methods return None gracefully when RDKit is unavailable or rendering fails,
so callers never need to catch exceptions from this module.
"""
from __future__ import annotations

import base64
import logging
import threading
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_rdkit_available: Optional[bool] = None
_rdkit_lock = threading.Lock()
_cairo_available: Optional[bool] = None


def _check_rdkit() -> bool:
    global _rdkit_available
    if _rdkit_available is None:
        with _rdkit_lock:
            if _rdkit_available is None:
                from app.config import RDKIT_ENABLED
                if not RDKIT_ENABLED:
                    _rdkit_available = False
                    logger.info("RDKIT_ENABLED=false – molecule rendering disabled")
                    return _rdkit_available
                try:
                    from rdkit import Chem  # noqa: F401
                    _rdkit_available = True
                except ImportError:
                    _rdkit_available = False
                    logger.warning("RDKit not available – molecule rendering disabled")
    return _rdkit_available


def _check_cairo() -> bool:
    """Whether RDKit can rasterise. Falls back to SVG when it cannot."""
    global _cairo_available
    if _cairo_available is None:
        with _rdkit_lock:
            if _cairo_available is None:
                try:
                    from rdkit.Chem.Draw import rdMolDraw2D
                    rdMolDraw2D.MolDraw2DCairo(8, 8)
                    _cairo_available = True
                except Exception as exc:
                    _cairo_available = False
                    logger.warning("RDKit Cairo unavailable (%s) – falling back to SVG", exc)
    return _cairo_available


def render_plain_svg(smiles: str, img_size: int = 300) -> Optional[str]:
    """
    Render a plain molecule depiction (no fingerprint highlighting).

    Returns an SVG string: line drawings have no contour fill, so they are a
    few kilobytes and stay vector. Only the highlighted view is rasterised.
    """
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


def render_fragment_svg(
    fragment_smiles: str,
    atom_symbol: str = "",
    img_size: int = 120,
) -> Optional[str]:
    """
    Draw a fingerprint bit's substructure on its own, for the panel thumbnails.

    These come from `Chem.PathToSubmol`, so they are *partial* structures: an
    aromatic environment clipped out of its ring arrives as "cc(C)oc(c)c", whose
    lowercase atoms are non-ring and fail a normal sanitize. Kekulisation and
    aromaticity perception are therefore skipped and the fragment is drawn as
    the open substructure it actually is.

    Radius-0 bits carry no fragment at all — they are one atom, so the element
    symbol is drawn instead.
    """
    if not _check_rdkit():
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D

        smiles = fragment_smiles or atom_symbol
        if not smiles:
            return None

        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            catchErrors=True,
        )

        # These fragments are cut out of a larger molecule, so their open
        # valences are bonds to the rest of the structure — not hydrogens.
        # Left alone RDKit fills them in, drawing a lone oxygen as "H2O" and an
        # ether oxygen as "OH", which asserts atoms the parent may not have.
        for atom in mol.GetAtoms():
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)

        rdDepictor.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DSVG(img_size, img_size)
        opts = drawer.drawOptions()
        # The mol is already prepared; re-preparing would re-run the kekulisation
        # that these partial fragments cannot survive.
        opts.prepareMolsBeforeDrawing = False
        opts.clearBackground = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as exc:
        logger.debug("render_fragment_svg failed for %r: %s", fragment_smiles, exc)
        return None


def render_bit_svg(
    smiles: str,
    atoms: list,
    bonds: list,
    img_size: int = 300,
) -> Optional[str]:
    """
    Render a depiction with one fingerprint bit's environment picked out.

    Stays vector: unlike the similarity map there is no contour fill, so the
    payload is a few kilobytes and the atom highlights are crisp at any zoom.
    Out-of-range indices are dropped rather than raising — the client may hold
    a bit list from a previous molecule while a new one is loading.
    """
    if not _check_rdkit():
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        n_atoms, n_bonds = mol.GetNumAtoms(), mol.GetNumBonds()
        hl_atoms = [int(a) for a in atoms if 0 <= int(a) < n_atoms]
        hl_bonds = [int(b) for b in bonds if 0 <= int(b) < n_bonds]

        half = img_size // 2
        drawer = rdMolDraw2D.MolDraw2DSVG(half, half)
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol, highlightAtoms=hl_atoms, highlightBonds=hl_bonds,
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as exc:
        logger.debug("render_bit_svg failed for %r: %s", smiles, exc)
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

    Weights are signed (see _atom_weights), so RDKit's PiWG colour map renders
    supporting atoms green and contradicting atoms pink.
    """
    if not _check_rdkit():
        return render_plain_svg(smiles, img_size)
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Pass the same SMILES string that produced `mol`: _atom_weights re-parses
        # it to extract bit environments, and canonicalizing in between would
        # renumber the atoms and shift every highlight.
        weights = _atom_weights(mol, smiles, predicted_fp, fp_loader)
        if weights is None:
            return render_plain_svg(smiles, img_size)

        weights, _ = SimilarityMaps.GetStandardizedWeights(weights)

        # Rasterise, like SPECTRE does. The contour fill turns into one <rect>
        # per grid cell in SVG — ~1 MB per card at the default resolution and
        # 2.7 MB at SPECTRE's 0.06 — where the PNG is ~100 KB at the finer grid.
        # It also keeps this payload out of dangerouslySetInnerHTML on the client.
        if _check_cairo():
            drawer = rdMolDraw2D.MolDraw2DCairo(img_size, img_size)
            SimilarityMaps.GetSimilarityMapFromWeights(
                mol, weights, draw2d=drawer,
                contourLines=0, gridResolution=0.06, extraGridPadding=0.5,
            )
            drawer.FinishDrawing()
            png = base64.b64encode(drawer.GetDrawingText()).decode("ascii")
            return f"data:image/png;base64,{png}"

        drawer = rdMolDraw2D.MolDraw2DSVG(img_size, img_size)
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol, weights, draw2d=drawer, contourLines=0,
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as exc:
        logger.debug("render_enhanced_svg failed for %r: %s", smiles, exc)
        return render_plain_svg(smiles, img_size)


def _mfp_from_bitinfo(atom_to_bits: dict, bitinfo_map: dict, out_dim: int,
                      ignore_atoms: tuple = ()) -> torch.Tensor:
    """
    Build a dense fingerprint from per-atom Morgan bit environments.

    Mirrors SPECTRE's build_mfp_from_bitInfo: every atom's bits are set, then the
    ignored atoms' bits are zeroed afterwards — so a bit an ignored atom touches
    drops out even when a different atom also contributes it. This is stronger
    than fp_loader.build_mfp_from_bitinfo, which only skips the ignored atom's
    own contribution, and it is what makes the ablation deltas large enough to
    swing negative.
    """
    import numpy as np

    fp = np.zeros(out_dim, dtype=np.float32)
    for bits in atom_to_bits.values():
        for b in bits:
            col = bitinfo_map.get(b)
            if col is not None and 0 <= col < out_dim:
                fp[col] = 1.0
    for atom_idx in ignore_atoms:
        for b in atom_to_bits.get(atom_idx, ()):
            col = bitinfo_map.get(b)
            if col is not None and 0 <= col < out_dim:
                fp[col] = 0.0
    return torch.from_numpy(fp)


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.norm(a) * torch.norm(b)
    return 0.0 if denom == 0 else float((a @ b) / denom)


def _atom_weights(mol, smiles: str, predicted_fp: torch.Tensor, fp_loader: object):
    """
    Per-atom contribution weights, as a leave-one-out ablation of the retrieved
    molecule's fingerprint (ported from SPECTRE's
    show_retrieved_mol_with_highlighted_frags).

    weight[i] = cos(FP_retrieved, FP_pred) - cos(FP_retrieved without atom i, FP_pred)

    Positive means removing the atom hurts the match, so the atom supports the
    retrieval; negative means removing it improves the match. The sign is the
    whole point — a non-negative weight vector renders green-only.
    """
    try:
        from app.marina_import import ensure_marina_importable
        ensure_marina_importable()
        from src.modules.data.fp_utils import get_bitinfos

        max_radius = getattr(fp_loader, "max_radius", 6) or 6
        bitinfo_map = getattr(fp_loader, "bitinfo_to_fp_index_map", {})
        if not bitinfo_map:
            return None

        atom_to_bits, _ = get_bitinfos(smiles, max_radius)
        if not atom_to_bits:
            return None

        pred = predicted_fp.detach().float().cpu().flatten()
        out_dim = pred.numel()

        base_fp = _mfp_from_bitinfo(atom_to_bits, bitinfo_map, out_dim)
        base_sim = _cos_sim(base_fp, pred)

        return [
            base_sim - _cos_sim(
                _mfp_from_bitinfo(atom_to_bits, bitinfo_map, out_dim, (atom_idx,)), pred)
            for atom_idx in range(mol.GetNumAtoms())
        ]
    except Exception as exc:
        logger.debug("_atom_weights failed: %s", exc)
        return None
