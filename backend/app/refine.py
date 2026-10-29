"""
Fragment deletion retrieval for hybrid compounds.

A hybrid A+B tends to retrieve only A-like compounds: A's bits dominate the
prediction and B never surfaces. The chemist picks a region of one retrieved
candidate (atoms S) and deletes it: retrieval then runs against what the
prediction says is NOT S.

The multiplicity vocabularies make this arithmetic rather than masking. A bit is
"fragment f occurs at least k times", so a fragment's bits form a cumulative
ladder. If the candidate's region S accounts for c(f) occurrences of f, the
probability that the rest of the true structure still has at least k of them is
the predicted probability of at least k + c(f):

    p'(f, >=k) = p(f, >=k + c(f))          (0 when that rung is not in the vocabulary)

Zeroing f's bits instead would also erase the occurrences B has of small, common
fragments (methyls, esters), which are exactly what B still needs.

c(f) counts every occurrence that *touches* S -- inside it, or spanning the
junction between S and the rest -- because a compound resembling the remainder
alone cannot carry the junction environments either.

The bank rows are stored L2-normalised (binary rows divided by sqrt(nnz)), so a
binary product is the stored product times sqrt(nnz); no second copy of the
bank is needed.

Prototype and the Rhodomollein LVII case study: wiki
MARINA/experiments/hybrid-fragment-deletion.md.
"""
from __future__ import annotations

import functools
import logging
import threading
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

MULTIPLICITY_KINDS = ("multiplicity", "multiplicity_uncapped", "unique_multiplicity")

# Containment scoring: Tversky index with alpha weighing the query bits a
# candidate misses and beta the candidate bits the query does not have. A small
# beta means "contains the remainder" without penalising the candidate's other
# parts -- for when the missing piece only exists embedded in larger molecules.
TVERSKY_ALPHA = 1.0
TVERSKY_BETA = 0.2

# Remaining bits worth listing in the "left unexplained" panel.
UNEXPLAINED_MIN = 0.5
UNEXPLAINED_LIMIT = 60

# Shared-core suggestions: share of the compounds that must contain the core.
# Below half, a core stops describing "what these retrievals agree on".
CORE_COVERAGES = (1.0, 0.8, 0.6, 0.5)
CORE_TIMEOUT_S = 4
CORE_MIN_ATOMS = 4
MAX_CORE_MOLECULES = 50


class RefineUnsupported(ValueError):
    """The model's fingerprint has no multiplicity ladder to shift."""


# ── Vocabulary ────────────────────────────────────────────────────────────────

_ladder_lock = threading.Lock()


def _ladders(session) -> Dict[str, Dict[int, int]]:
    """fragment SMILES -> {k: column} for a multiplicity vocabulary, cached on the session."""
    cached = getattr(session, "_refine_ladders", None)
    if cached is not None:
        return cached
    with _ladder_lock:
        cached = getattr(session, "_refine_ladders", None)
        if cached is None:
            fp_loader = session.fp_loader
            if getattr(fp_loader, "FEATURE_KIND", None) not in MULTIPLICITY_KINDS:
                raise RefineUnsupported(
                    "Fragment deletion needs a multiplicity fingerprint; this model does not use one.")
            cached = defaultdict(dict)
            for key, col in fp_loader.bitinfo_to_fp_index_map.items():
                frag, k = key
                cached[frag][int(k)] = int(col)
            cached = dict(cached)
            session._refine_ladders = cached
    return cached


# ── Which occurrences the selection accounts for ──────────────────────────────

def _occurrences(mol, radius: int, kind: str):
    """(fragment SMILES, atom set) per occurrence, exactly as the vocabulary counts them."""
    from rdkit import Chem
    from app.marina_import import ensure_marina_importable
    ensure_marina_importable()
    from src.modules.data.fp_utils import _substructure_occurrences

    for atom_idx, frag, r in _substructure_occurrences(
            mol, radius, (), dedup=kind == "unique_multiplicity"):
        atoms = {atom_idx}
        for b in Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx):
            bond = mol.GetBondWithIdx(b)
            atoms.add(bond.GetBeginAtomIdx())
            atoms.add(bond.GetEndAtomIdx())
        yield frag, frozenset(atoms)


def _parse(smiles: str):
    """Parse with the atom order the depiction uses, so client atom indices line up."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES {smiles!r}")
    Chem.RemoveStereochemistry(mol)
    return mol


def selection_counts(session, smiles: str, atoms: Iterable[int]) -> Counter:
    """Per-fragment count of the molecule's occurrences that touch the selected atoms."""
    ladders = _ladders(session)
    mol = _parse(smiles)
    n = mol.GetNumAtoms()
    sel = {int(a) for a in atoms}
    bad = [a for a in sel if not 0 <= a < n]
    if bad:
        raise ValueError(f"Atom indices {bad[:5]} are out of range for a {n}-atom molecule")
    fp_loader = session.fp_loader
    radius = int(getattr(fp_loader, "max_radius", 10) or 10)
    counts: Counter = Counter()
    for frag, occ_atoms in _occurrences(mol, radius, fp_loader.FEATURE_KIND):
        if frag in ladders and occ_atoms & sel:
            counts[frag] += 1
    return counts


def selection_smiles(smiles: str, atoms: Iterable[int]) -> str:
    """The selected atoms as a SMILES fragment, for the edit's thumbnail."""
    from rdkit import Chem
    mol = _parse(smiles)
    sel = sorted({int(a) for a in atoms})
    if not sel:
        return ""
    try:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=sel, canonical=True, isomericSmiles=False)
    except Exception:
        return ""


def selection_query(smiles: str, atoms: Iterable[int]):
    """
    The selected atoms as a substructure query (for "require this region"): their
    elements, aromaticity and the bonds between them, with open valences, so a
    compound matches whatever its substituents are. A selection in several pieces
    must match as several pieces.
    """
    from rdkit import Chem
    mol = _parse(smiles)
    sel = sorted({int(a) for a in atoms})
    bad = [a for a in sel if not 0 <= a < mol.GetNumAtoms()]
    if not sel or bad:
        raise ValueError(f"Atom indices {bad[:5] or sel} are not valid for this molecule")
    smarts = Chem.MolFragmentToSmarts(mol, atomsToUse=sel, isomericSmarts=False)
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise ValueError("The selected region could not be turned into a substructure query")
    return smarts, query


# ── Fingerprint arithmetic ────────────────────────────────────────────────────

def shift_query(p: torch.Tensor, counts: Dict[str, int], ladders) -> torch.Tensor:
    """p'(f, >=k) = p(f, >=k + c(f)): the occurrences the selection does not explain."""
    q = p.clone()
    for frag, c in counts.items():
        if c <= 0:
            continue
        lad = ladders[frag]
        for k, col in lad.items():
            src = lad.get(k + c)
            q[col] = p[src] if src is not None else 0.0
    return q


def _bank(session) -> Tuple[torch.Tensor, torch.Tensor]:
    """(L2-normalised CSR bank, sqrt(nnz) per row), the latter cached on the session."""
    store = session.get_rankingset().data
    sq = getattr(session, "_refine_sqrt_nnz", None)
    if sq is None or sq.numel() != store.size(0):
        if store.layout == torch.sparse_csr:
            nnz = (store.crow_indices()[1:] - store.crow_indices()[:-1]).float()
        else:
            nnz = (store != 0).sum(dim=1).float()
        sq = nnz.sqrt()
        session._refine_sqrt_nnz = sq
    return store, sq


def _binary_dot(store: torch.Tensor, sq: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Row-wise binary(bank) . v, using the normalised rows: binary = normalised * sqrt(nnz)."""
    out = (store @ v.reshape(-1, 1).to(store.dtype)).reshape(-1)
    return out * sq


def score_bank(session, q: torch.Tensor, metric: str) -> torch.Tensor:
    """Similarity of every bank row to the edited query."""
    store, sq = _bank(session)
    inter = _binary_dot(store, sq, q)
    size = sq * sq
    if metric == "containment":
        missed = q.sum() - inter
        extra = size - inter
        return inter / (inter + TVERSKY_ALPHA * missed + TVERSKY_BETA * extra).clamp_min(1e-9)
    return inter / (q.norm() * size.clamp_min(0).sqrt()).clamp_min(1e-9)


def unexplained_bits(session, q: torch.Tensor, calibrator=None) -> List[dict]:
    """Confident bits of the edited query: what is left for the rest of the structure to explain."""
    index_to_bitinfo = session.fp_loader.fp_index_to_bitinfo_map
    rows = []
    for col in torch.nonzero(q >= 0.05).reshape(-1).tolist():
        raw = float(q[col])
        conf = calibrator(raw) if calibrator else raw
        if conf < UNEXPLAINED_MIN:
            continue
        frag, k = index_to_bitinfo[col]
        rows.append({"index": col, "fragment_smiles": frag, "multiplicity": int(k),
                     "raw_confidence": raw, "confidence": conf})
    # Larger fragments say more about what is missing; within one, the highest rung.
    rows.sort(key=lambda r: (-len(r["fragment_smiles"]), -r["multiplicity"]))
    # One row per fragment: its highest confident rung already implies the lower ones.
    seen, out = set(), []
    for r in rows:
        if r["fragment_smiles"] in seen:
            continue
        seen.add(r["fragment_smiles"])
        out.append(r)
    return out[:UNEXPLAINED_LIMIT]


# ── Shared-core suggestions ───────────────────────────────────────────────────

@functools.lru_cache(maxsize=64)
def _cores(smiles: Tuple[str, ...]) -> Tuple[Tuple[float, str, int, int, bool], ...]:
    """(coverage, SMARTS, atoms, molecules containing it, timed out) per coverage level, deduplicated."""
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    mols = [m for m in (Chem.MolFromSmiles(s) for s in smiles) if m is not None]
    if len(mols) < 2:
        return ()
    out, seen = [], set()
    for cov in CORE_COVERAGES:
        res = rdFMCS.FindMCS(
            mols, threshold=cov, timeout=CORE_TIMEOUT_S,
            ringMatchesRingOnly=True, completeRingsOnly=True,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
        )
        if not res.smartsString or res.numAtoms < CORE_MIN_ATOMS or res.smartsString in seen:
            continue
        seen.add(res.smartsString)
        query = Chem.MolFromSmarts(res.smartsString)
        found = sum(m.HasSubstructMatch(query) for m in mols)
        out.append((cov, res.smartsString, res.numAtoms, found, bool(res.canceled)))
    return tuple(out)


# A compound can hold a core more than once (a dimer); every copy is highlighted, up to this many.
MAX_CORE_COPIES = 4


def shared_cores(smiles: Sequence[str], target: Optional[str] = None,
                 matches: bool = False) -> Tuple[int, List[dict]]:
    """
    Cores shared by the given compounds. Each is mapped onto the target's atoms
    when a target is given and has it; with `matches`, also onto every compound
    of the pool (one atom list per copy, in depiction atom order, or None).
    """
    from rdkit import Chem

    pool = tuple(smiles[:MAX_CORE_MOLECULES])
    target_mol = _parse(target) if target else None
    pool_mols = [_parse(s) for s in pool] if matches else []
    cores = []
    for cov, smarts, n_atoms, found, timed_out in _cores(pool):
        query = Chem.MolFromSmarts(smarts)
        core = {
            "coverage": cov, "smarts": smarts, "num_atoms": n_atoms,
            "molecules_with_core": found, "partial": timed_out, "atoms": None,
        }
        if target_mol is not None:
            match = target_mol.GetSubstructMatch(query)
            core["atoms"] = sorted(match) if match else None
        if matches:
            core["matches"] = [
                [sorted(m) for m in mol.GetSubstructMatches(query, uniquify=True, maxMatches=MAX_CORE_COPIES)]
                or None
                for mol in pool_mols
            ]
        cores.append(core)
    return len(pool), cores
