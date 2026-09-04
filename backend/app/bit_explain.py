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

# Multiplicity thermometer floor: a cumulative level is worth a bar only if the
# candidate reaches it or the model predicts it at least this strongly. Trailing
# levels that are neither (all sit at ~0) are dropped, and a whole fragment that is
# neither present nor predicted above this is not shown at all.
BUCKET_FLOOR = 0.05

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


def _locations(smiles: str, max_radius: int, bitinfo_to_col: dict,
               feature_kind: str = "morgan"
               ) -> Optional[Dict[int, Tuple[List[int], List[int], int]]]:
    """
    Map fingerprint column -> (atom indices, bond indices, radius) in this molecule.

    A bit may fire at several centres (aspirin's C=O bit hits both carbonyls),
    so the footprints of all occurrences are unioned.

    The radius is carried out because a substructure feature is keyed on the fragment
    SMILES alone and has no radius in its key, unlike a Morgan BitInfo tuple.
    """
    from app.marina_import import ensure_marina_importable
    ensure_marina_importable()
    from src.modules.data.fp_utils import get_feature_locations
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_to_feats = get_feature_locations(smiles, max_radius, kind=feature_kind)
    if not atom_to_feats:
        return None

    out: Dict[int, Tuple[set, set, int]] = {}
    for atom_idx, feats in atom_to_feats.items():
        for feat, radius in feats:
            col = bitinfo_to_col.get(feat)
            if col is None:
                continue
            atoms, bonds, seen_r = out.setdefault(col, (set(), set(), radius))
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
            for bond_idx in env:
                bond = mol.GetBondWithIdx(bond_idx)
                bonds.add(bond_idx)
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())
            if not env:
                # Radius-0 bit: the centre atom itself, no bonds.
                atoms.add(atom_idx)
            # Keep the smallest radius that produced this feature: the substructure
            # enumeration re-emits a saturated environment at every larger radius.
            out[col] = (atoms, bonds, min(seen_r, radius))
    return {c: (sorted(a), sorted(b), r) for c, (a, b, r) in out.items()}


_MULTIPLICITY_KINDS = ("multiplicity", "multiplicity_uncapped")


def _fragment_radius(frag: str) -> int:
    """
    Approximate Morgan radius of a substructure from its fragment SMILES: the graph
    radius (min over atoms of the eccentricity). A multiplicity feature carries no
    radius in its key, so this recovers a stable one for ordering small→large, and
    it agrees with the Morgan radius for the symmetric environments Morgan emits.
    """
    if not frag:
        return 0
    from rdkit import Chem
    mol = Chem.MolFromSmiles(frag, sanitize=False)
    if mol is None or mol.GetNumAtoms() <= 1:
        return 0
    dm = Chem.GetDistanceMatrix(mol)
    n = mol.GetNumAtoms()
    return int(min(max(dm[i][j] for j in range(n)) for i in range(n)))


def _collapse_multiplicity(pred_fp, considered, present, locs, index_to_bitinfo, calibrator):
    """
    Collapse a multiplicity vocabulary's per-bucket bits into one row per fragment.

    A fragment's cumulative buckets (≥1×, ≥2×, …) are separate columns; here they
    become one row carrying the whole thermometer in `buckets` (every vocabulary
    level with its predicted confidence, monotonic or not) plus `true_count`, the
    count the candidate actually reaches. The row's own confidence is the ≥1×
    bucket — "does this fragment appear at all" — which is what sorting/grouping key on.
    """
    frag_buckets: Dict[str, List[Tuple[int, int]]] = {}
    for col, info in index_to_bitinfo.items():
        if isinstance(info, tuple) and len(info) == 2:
            frag_buckets.setdefault(info[0], []).append((int(info[1]), col))
    for lst in frag_buckets.values():
        lst.sort()

    frags, seen = [], set()
    for col in considered:
        info = index_to_bitinfo.get(col)
        if isinstance(info, tuple) and len(info) == 2 and info[0] not in seen:
            seen.add(info[0])
            frags.append(info[0])

    def conf_of(col):
        raw = float(pred_fp[col]) if col < len(pred_fp) else 0.0
        return raw, (calibrator(raw) if calibrator else raw)

    rows = []
    for frag in frags:
        buckets, present_levels = [], []
        rep_atoms, rep_bonds, rep_radius = [], [], -1
        one_raw = one_conf = 0.0
        one_col = -1
        for lvl, col in frag_buckets.get(frag, []):
            raw, conf = conf_of(col)
            in_mol = col in present
            atoms, bonds, r = locs.get(col, ([], [], -1))
            if in_mol:
                present_levels.append(lvl)
                if not rep_atoms:                       # first present bucket = fragment location
                    rep_atoms, rep_bonds, rep_radius = list(atoms), list(bonds), r
            if lvl == 1:
                one_raw, one_conf, one_col = raw, conf, col
            buckets.append({"level": lvl, "index": col, "raw_confidence": raw,
                            "confidence": conf, "band": band(conf), "present": in_mol})
        if not buckets:
            continue
        if one_col == -1:                                # no explicit ≥1 bucket → lowest level
            b0 = buckets[0]
            one_raw, one_conf, one_col = b0["raw_confidence"], b0["confidence"], b0["index"]
        true_count = max(present_levels) if present_levels else 0
        is_present = true_count >= 1

        # Drop a fragment that is neither present nor predicted above the floor — an
        # all-zero thermometer carries no information.
        max_conf = max((b["confidence"] for b in buckets), default=0.0)
        if not is_present and max_conf < BUCKET_FLOOR:
            continue
        # Trim trailing buckets that are neither present nor predicted, so the strip
        # shows only the meaningful range (up to the true count or the last prediction).
        kmax = max((b["level"] for b in buckets if b["present"] or b["confidence"] >= BUCKET_FLOOR),
                   default=1)
        buckets = [b for b in buckets if b["level"] <= kmax]

        group = ((GROUP_MATCH if is_present else GROUP_MISSING) if one_conf >= CONFIDENT
                 else (GROUP_UNEXPECTED if is_present else GROUP_UNCERTAIN))
        rows.append({
            "index": one_col, "fragment_smiles": frag or "", "atom_symbol": "",
            "radius": _fragment_radius(frag), "multiplicity": None,
            "raw_confidence": one_raw, "confidence": one_conf, "band": band(one_conf),
            "present": is_present, "group": group, "atoms": rep_atoms, "bonds": rep_bonds,
            "buckets": buckets, "true_count": true_count,
            # entropy rank within a radius = the vocabulary column order (lower = higher
            # entropy), taken from the fragment's ≥1× bucket.
            "entropy_rank": frag_buckets.get(frag, [(0, one_col)])[0][1],
        })
    return rows


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
    feature_kind = getattr(fp_loader, "FEATURE_KIND", "morgan")

    locs = _locations(smiles, max_radius, bitinfo_to_col, feature_kind) or {}
    present = set(locs)

    # Candidate bits always get a row; predicted bits only above the floor.
    considered = present | {
        i for i, p in enumerate(pred_fp) if p >= MIN_CONFIDENCE
    }

    if feature_kind in _MULTIPLICITY_KINDS:
        # One row per fragment, carrying its whole ≥1×, ≥2×, … thermometer.
        rows = _collapse_multiplicity(pred_fp, considered, present, locs,
                                      index_to_bitinfo, calibrator)
    else:
        rows = []
        for col in considered:
            raw = float(pred_fp[col]) if col < len(pred_fp) else 0.0
            conf = calibrator(raw) if calibrator else raw
            in_mol = col in present
            if conf >= CONFIDENT:
                group = GROUP_MATCH if in_mol else GROUP_MISSING
            else:
                group = GROUP_UNEXPECTED if in_mol else GROUP_UNCERTAIN

            atoms, bonds, found_radius = locs.get(col, ([], [], -1))
            info = index_to_bitinfo.get(col)
            if isinstance(info, str):
                # Substructure vocabulary: the feature *is* the fragment SMILES. There is no
                # centre-atom symbol, and the radius is only known from where it was found --
                # so a feature this structure lacks reports -1 rather than a made-up radius.
                frag_smiles, atom_symbol, radius = info, "", found_radius
            elif info:
                _bit_id, atom_symbol, frag_smiles, radius = info
            else:
                frag_smiles, atom_symbol, radius = None, None, None
            rows.append({
                "index": col,
                "fragment_smiles": frag_smiles or "",
                "atom_symbol": atom_symbol or "",
                "radius": radius if radius is not None else -1,
                "multiplicity": None,
                "raw_confidence": raw,
                "confidence": conf,
                "band": band(conf),
                "present": in_mol,
                "group": group,
                "atoms": atoms,
                "bonds": bonds,
            })

    if feature_kind in _MULTIPLICITY_KINDS:
        # Small fragments first, then by vocabulary (entropy) order within a radius.
        rows.sort(key=lambda r: (r["radius"], r.get("entropy_rank", r["index"])))
    else:
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
