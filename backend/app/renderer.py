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


def highlighting_available() -> bool:
    """
    Whether the server will produce highlighted depictions at all.

    False when RDKit is missing or HIGHLIGHT_ENABLED=false, in which case every
    card carries `plain_svg` only and a client-side highlight toggle has nothing
    to switch to.
    """
    from app.config import HIGHLIGHT_ENABLED
    return HIGHLIGHT_ENABLED and _check_rdkit()


# Margin around the molecule's bounding box, in molecule coordinate units (~Å),
# applied identically to every depiction mode.
_FIT_PADDING = 0.9

_BOND_WIDTH = 2.0

# Similarity-map wash. sigma is a fraction of a bond length: 0.45 blends
# neighbouring atoms into regions rather than a dot per atom. The gamma lifts
# the weaker contributions, which standardising to the single strongest atom
# otherwise leaves nearly invisible on a large molecule; it is monotone, so a
# stronger colour still means a larger contribution.
_MAP_SIGMA = 0.45
_MAP_GAMMA = 0.7
_MAP_GRID = 0.05


def _prepare_mol(smiles: str):
    """
    Parse a SMILES into a molecule ready to draw, or None.

    Stereochemistry is stripped first: MARINA's fingerprints are 2-D, so a wedge
    or E/Z geometry on the depiction would claim information the model never saw.
    The 2-D layout is computed here, once, so every depiction of the molecule
    shares the same coordinates.
    """
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return rdMolDraw2D.PrepareMolForDrawing(mol)


def _fit(drawer, mol, canvas: int) -> None:
    """
    Pin the drawing transform to the molecule's bounding box plus a fixed margin.

    RDKit otherwise picks a scale per call — the similarity map fits its contour
    grid, a plain drawing fits the atoms — so the same molecule landed at
    different sizes and positions depending on the view mode. With one explicit
    fit, the skeleton sits in the same place in every depiction and an overlay
    drawn from `depiction_geometry` lines up with all of them.
    """
    from rdkit.Geometry import Point2D

    conf = mol.GetConformer()
    pts = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    xs, ys = [p.x for p in pts], [p.y for p in pts]
    drawer.SetScale(
        canvas, canvas,
        Point2D(min(xs) - _FIT_PADDING, min(ys) - _FIT_PADDING),
        Point2D(max(xs) + _FIT_PADDING, max(ys) + _FIT_PADDING),
    )


def render_plain_svg(smiles: str, img_size: int = 300) -> Optional[str]:
    """
    Render the molecule as a line drawing, with a transparent background.

    Returns an SVG string: a few kilobytes, and vector, so it stays crisp at
    any size. This is the top layer in every view mode — the similarity map is
    a colour wash the client places underneath it — which is why it carries no
    background of its own.
    """
    if not _check_rdkit():
        return None
    try:
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = _prepare_mol(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DSVG(img_size, img_size)
        opts = drawer.drawOptions()
        opts.bondLineWidth = _BOND_WIDTH
        opts.clearBackground = False
        _fit(drawer, mol, img_size)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as exc:
        logger.debug("render_plain_svg failed for %r: %s", smiles, exc)
        return None


def depiction_geometry(smiles: str, img_size: int = 300) -> Optional[dict]:
    """
    Where each atom lands on the depiction, in canvas pixels, plus the bond list.

    Uses the same preparation and fit as the depictions themselves, so a client
    can draw its own highlights over any of them. Coordinates are for a square
    canvas of `img_size`; an overlay scaled to the displayed image needs no
    other correction.
    """
    if not _check_rdkit():
        return None
    try:
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = _prepare_mol(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DSVG(img_size, img_size)
        _fit(drawer, mol, img_size)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        atoms = []
        for i in range(mol.GetNumAtoms()):
            p = drawer.GetDrawCoords(i)
            atoms.append([round(p.x, 2), round(p.y, 2)])
        bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
        return {"size": img_size, "atoms": atoms, "bonds": bonds}
    except Exception as exc:
        logger.debug("depiction_geometry failed for %r: %s", smiles, exc)
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


def render_enhanced_svg(
    smiles: str,
    predicted_fp: torch.Tensor,
    fp_loader: object,
    img_size: int = 400,
) -> Optional[str]:
    """
    Render the similarity-map colour wash for a molecule: the Gaussian fill of
    RDKit's SimilarityMaps, on a transparent background, with no molecule
    drawn in it. The client layers `render_plain_svg` on top, so the skeleton
    stays vector-crisp and lands in exactly the same place as in the plain
    view (both use `_fit`), instead of this being a second, rasterised
    picture of the molecule at its own scale.

    Weights are signed (see _atom_weights), so RDKit's PiWG colour map renders
    supporting atoms green and contradicting atoms pink.

    Returns None when highlighting is switched off (HIGHLIGHT_ENABLED=false),
    when RDKit is unavailable, or when the molecule cannot be weighted: the
    card carries `plain_svg` regardless, and a wash without weights would be
    an empty image.
    """
    from app.config import HIGHLIGHT_ENABLED
    if not HIGHLIGHT_ENABLED or not _check_rdkit():
        return None
    try:
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
        from rdkit.Geometry import Point2D

        mol = _prepare_mol(smiles)
        if mol is None:
            return None

        # Pass the same SMILES string that produced `mol`: _atom_weights re-parses
        # it to extract bit environments, and canonicalizing in between would
        # renumber the atoms and shift every highlight.
        weights = _atom_weights(mol, smiles, predicted_fp, fp_loader)
        if weights is None:
            return None

        weights, _ = SimilarityMaps.GetStandardizedWeights(weights)
        weights = [(1 if w >= 0 else -1) * abs(w) ** _MAP_GAMMA for w in weights]

        # SimilarityMaps.GetSimilarityMapFromWeights lets the contour grid set the
        # scale, which is what made the map sit at a different size from the plain
        # drawing. This is the same Gaussian fill with the shared fit instead.
        conf = mol.GetConformer()
        locs = [Point2D(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y)
                for i in range(mol.GetNumAtoms())]
        bond = mol.GetBondWithIdx(0) if mol.GetNumBonds() else None
        a, b = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) if bond else (0, 1)
        sigma = round(_MAP_SIGMA * (conf.GetAtomPosition(a) - conf.GetAtomPosition(b)).Length(), 2)
        params = Draw.ContourParams()
        params.fillGrid = True
        params.gridResolution = _MAP_GRID
        params.extraGridPadding = 0.5
        params.setScale = False

        def draw(drawer):
            drawer.drawOptions().setBackgroundColour((1.0, 1.0, 1.0, 0.0))
            _fit(drawer, mol, img_size)
            drawer.ClearDrawing()
            Draw.ContourAndDrawGaussians(
                drawer, locs, weights, [sigma] * len(locs), nContours=0, params=params,
            )
            drawer.FinishDrawing()

        # Rasterise, like SPECTRE does. The fill is one <rect> per grid cell in
        # SVG — megabytes per card — where the PNG stays around 100 KB. It also
        # keeps this payload out of dangerouslySetInnerHTML on the client.
        if _check_cairo():
            drawer = rdMolDraw2D.MolDraw2DCairo(img_size, img_size)
            draw(drawer)
            png = base64.b64encode(drawer.GetDrawingText()).decode("ascii")
            return f"data:image/png;base64,{png}"

        drawer = rdMolDraw2D.MolDraw2DSVG(img_size, img_size)
        draw(drawer)
        return drawer.GetDrawingText()
    except Exception as exc:
        logger.debug("render_enhanced_svg failed for %r: %s", smiles, exc)
        return None


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
        from src.modules.data.fp_utils import get_feature_locations

        max_radius = getattr(fp_loader, "max_radius", 6) or 6
        bitinfo_map = getattr(fp_loader, "bitinfo_to_fp_index_map", {})
        if not bitinfo_map:
            return None

        feature_kind = getattr(fp_loader, "FEATURE_KIND", "morgan")
        located = get_feature_locations(smiles, max_radius, kind=feature_kind)
        if not located:
            return None
        # The ablation only needs which features each atom carries, not where they sit.
        atom_to_bits = {a: [f for f, _ in feats] for a, feats in located.items()}

        pred = predicted_fp.detach().float().cpu().flatten()
        out_dim = pred.numel()

        base_fp = _mfp_from_bitinfo(atom_to_bits, bitinfo_map, out_dim)
        base_sim = _cos_sim(base_fp, pred)

        if feature_kind in _MULTIPLICITY_KINDS:
            drops = _multiplicity_drops(atom_to_bits, feature_kind)
        else:
            drops = {a: set(bits) for a, bits in atom_to_bits.items()}

        weights = []
        for atom_idx in range(mol.GetNumAtoms()):
            fp = base_fp.clone()
            for b in drops.get(atom_idx, ()):
                col = bitinfo_map.get(b)
                if col is not None and 0 <= col < out_dim:
                    fp[col] = 0.0
            weights.append(base_sim - _cos_sim(fp, pred))
        return weights
    except Exception as exc:
        logger.debug("_atom_weights failed: %s", exc)
        return None


_MULTIPLICITY_KINDS = ("multiplicity", "multiplicity_uncapped", "unique_multiplicity")


def _multiplicity_drops(atom_to_bits: dict, feature_kind: str) -> dict:
    """
    Which (fragment, bucket) features ablating each atom removes, for a counting
    vocabulary.

    SPECTRE's rule — zero every bit centred on the atom — is right for Morgan bits,
    where one bit is one environment. Here a fragment's cumulative buckets (≥1×,
    ≥2×, …) are separate columns and an atom centred on ONE occurrence of a
    fragment counted n times carries all n of them, so zeroing them all pretends
    the fragment vanished from the molecule. Removal then always destroys wanted
    bits, every weight comes out positive and the map can only ever be green.

    Removing an occurrence lowers the count from n to n-1, which turns off only
    the ≥n× bucket, so that is what an atom's ablation drops (one bucket per
    occurrence it is the centre of, from the top down).
    """
    from collections import Counter
    from src.modules.data.fp_utils import _multiplicity_cap

    cap = _multiplicity_cap(feature_kind)
    # One (frag, 1) entry per occurrence, keyed by its centre atom.
    centred = {a: [f for f, k in bits if k == 1] for a, bits in atom_to_bits.items()}
    counts = Counter(f for frags in centred.values() for f in frags)
    drops = {}
    for atom, frags in centred.items():
        removed = set()
        for frag, m in Counter(frags).items():
            n = counts[frag]
            top, new_top = min(n, cap), min(n - m, cap)
            removed |= {(frag, k) for k in range(new_top + 1, top + 1)}
        drops[atom] = removed
    return drops
