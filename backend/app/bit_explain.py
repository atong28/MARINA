"""
Per-bit explanation of a predicted fingerprint against a candidate structure.

Each of the 16,384 entropy-fingerprint bits is a circular substructure, and
`fp_loader.fp_index_to_bitinfo_map` already stores what it is:
(bit_id, centre atom symbol, fragment SMILES, radius). Re-running the Morgan
environment extraction on the candidate SMILES says *where* each bit sits, so a
bit can be highlighted on the depiction.

The panel is ordered by disagreement, because that is where the information is.
A molecule has ~56 bits predicted above 0.9, so a confidence-sorted list is
almost all confident matches; the short list of confident bits the candidate
*lacks* is what actually discriminates between candidates.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Bits below this predicted probability are not worth a row unless the candidate
# actually has them. ~16,318 of 16,384 bits per molecule sit under 0.01.
MIN_CONFIDENCE = 0.01

# Group ordering: confident misses first, then confirmations, then the tail.
GROUP_MISSING = "missing"        # predicted present, candidate lacks it
GROUP_MATCH = "match"            # predicted present, candidate has it
GROUP_UNEXPECTED = "unexpected"  # not predicted, candidate has it anyway
GROUP_UNCERTAIN = "uncertain"    # model is unsure, candidate lacks it
_GROUP_RANK = {GROUP_MISSING: 0, GROUP_MATCH: 1, GROUP_UNEXPECTED: 2, GROUP_UNCERTAIN: 3}

CONFIDENT = 0.5


def band(p: float) -> str:
    """Qualitative label. Thresholds are on the *calibrated* probability."""
    if p >= 0.9:
        return "Very likely"
    if p >= 0.7:
        return "Likely"
    if p >= 0.4:
        return "Possible"
    return "Unlikely"


def _locations(smiles: str, max_radius: int, bitinfo_to_col: dict
               ) -> Optional[Dict[int, Tuple[List[int], List[int]]]]:
    """
    Map fingerprint column -> (atom indices, bond indices) in this molecule.

    A bit may fire at several centres (aspirin's C=O bit hits both carbonyls),
    so the footprints of all occurrences are unioned.
    """
    from app.marina_import import ensure_marina_importable
    ensure_marina_importable()
    from src.modules.data.fp_utils import get_bitinfos
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_to_bits, _ = get_bitinfos(smiles, max_radius)
    if not atom_to_bits:
        return None

    out: Dict[int, Tuple[set, set]] = {}
    for atom_idx, bitinfos in atom_to_bits.items():
        for bi in bitinfos:
            col = bitinfo_to_col.get(bi)
            if col is None:
                continue
            atoms, bonds = out.setdefault(col, (set(), set()))
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, bi[3], atom_idx)
            for bond_idx in env:
                bond = mol.GetBondWithIdx(bond_idx)
                bonds.add(bond_idx)
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())
            if not env:
                # Radius-0 bit: the centre atom itself, no bonds.
                atoms.add(atom_idx)
    return {c: (sorted(a), sorted(b)) for c, (a, b) in out.items()}


def explain_bits(session, smiles: str, pred_fp: List[float], limit: int,
                 calibrator=None, include_fragment_svg: bool = False) -> dict:
    """
    Join predicted confidence, substructure identity and in-molecule location.

    Returns the rows plus per-group totals, so the UI can say "+N more" without
    the whole 16,384-row table crossing the wire.

    `include_fragment_svg` attaches a drawing of each substructure. It is opt-in
    because it is only worth the payload in the expanded view — the detail
    overlay asks for it, and nothing else does.
    """
    fp_loader = session.fp_loader
    index_to_bitinfo = getattr(fp_loader, "fp_index_to_bitinfo_map", {})
    bitinfo_to_col = getattr(fp_loader, "bitinfo_to_fp_index_map", {})
    max_radius = getattr(fp_loader, "max_radius", 6) or 6

    locs = _locations(smiles, max_radius, bitinfo_to_col) or {}
    present = set(locs)

    # Candidate bits always get a row; predicted bits only above the floor.
    considered = present | {
        i for i, p in enumerate(pred_fp) if p >= MIN_CONFIDENCE
    }

    rows = []
    for col in considered:
        raw = float(pred_fp[col]) if col < len(pred_fp) else 0.0
        conf = calibrator(raw) if calibrator else raw
        in_mol = col in present
        if conf >= CONFIDENT:
            group = GROUP_MATCH if in_mol else GROUP_MISSING
        else:
            group = GROUP_UNEXPECTED if in_mol else GROUP_UNCERTAIN

        info = index_to_bitinfo.get(col)
        bit_id, atom_symbol, frag_smiles, radius = (
            info if info else (None, None, None, None)
        )
        atoms, bonds = locs.get(col, ([], []))
        rows.append({
            "index": col,
            "fragment_smiles": frag_smiles or "",
            "atom_symbol": atom_symbol or "",
            "radius": radius if radius is not None else -1,
            "raw_confidence": raw,
            "confidence": conf,
            "band": band(conf),
            "present": in_mol,
            "group": group,
            "atoms": atoms,
            "bonds": bonds,
        })

    rows.sort(key=lambda r: (_GROUP_RANK[r["group"]], -r["confidence"], r["index"]))

    if include_fragment_svg:
        from app.renderer import render_fragment_svg
        # Only the returned rows are drawn, and identical fragments are drawn
        # once: the same substructure string names many distinct bits, so a
        # 60-row panel typically holds far fewer unique pictures.
        drawn: Dict[tuple, Optional[str]] = {}
        for row in rows[:limit]:
            key = (row["fragment_smiles"], row["atom_symbol"])
            if key not in drawn:
                drawn[key] = render_fragment_svg(key[0], key[1])
            row["fragment_svg"] = drawn[key]

    totals: Dict[str, int] = {g: 0 for g in _GROUP_RANK}
    for r in rows:
        totals[r["group"]] += 1

    return {
        "smiles": smiles,
        "calibrated": calibrator is not None,
        "bits": rows[:limit],
        "totals": totals,
        "total_shown": min(limit, len(rows)),
        "total_available": len(rows),
    }
