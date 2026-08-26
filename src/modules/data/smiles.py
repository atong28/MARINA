"""
Canonical SMILES for dataset construction.

There used to be two copies of this function: process_smiles.py canonicalized twice,
generate_dataset.py once. build_retrieval_set unions the key sets from both, so the
retrieval database ended up holding some molecules under a pass-1 spelling and some
under a pass-2 spelling — the same compound in two rows, and every string comparison
between the two sides reporting a spurious mismatch. Both now import from here.
"""
from typing import Optional

from rdkit import Chem

MAX_PASSES = 5


def canonicalize_smiles(
    smiles: str,
    keep_stereo: bool = False,
    largest_fragment: bool = True,
) -> Optional[str]:
    """
    Canonicalize to a fixed point, i.e. until re-parsing stops changing the string.

    This is the single source of truth for SMILES canonicalization across the whole
    MARINA-DB build (index, retrieval, splits, benchmarks) and for fingerprint-identity
    keys. Every pipeline stage imports this function; do not reintroduce a local copy.

    RDKit canonicalization is not idempotent in one pass. An explicit [H] that the
    parser preserves because it carries stereo or isotope information is written as a
    bare atom under isomericSmiles=False, and absorbed into its neighbour's implicit
    count on re-parse. Measured over MARINA1's 518,901 molecules this affects 159
    (0.031%), all by that one mechanism.

    Two passes was empirically enough for every molecule in MARINA1 under four RDKit
    versions, but the canonical form of metal-coordinated natural products has already
    changed once between releases (O[Fe] -> [O][Fe] between 2024.09 and 2025.03), so
    this loops to a fixed point rather than assuming a pass count.

    largest_fragment: reduce a multi-component input (salts, counter-ions) to its
    largest fragment before canonicalizing, matching the dataset spec. Applied before
    the fixed-point loop; the loop then also stabilizes the stripped form.

    Returns None for anything RDKit cannot parse, at any pass.
    """
    if largest_fragment and "." in smiles:
        smiles = max(smiles.split("."), key=len)
    seen = set()
    for _ in range(MAX_PASSES):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        folded = Chem.MolToSmiles(mol, isomericSmiles=keep_stereo, canonical=True)
        if folded == smiles:
            return smiles
        seen.add(folded)
        smiles = folded
    # Non-convergence. The 2D key path (keep_stereo=False) is not expected to oscillate
    # (stereo stripped), and a non-fixed-point *key* would split one compound across two
    # rows -- the defect this function exists to prevent -- so that stays a hard error.
    # The stereo-preserving path DOES oscillate for a few natural products: RDKit flips the
    # directional bonds around a ring-closure double bond between two equivalent spellings
    # each pass (a 2-cycle, emitting "Conflicting single bond directions" warnings). That
    # SMILES is provenance (canonical_3d_smiles / original_smiles), never a dedup key, so
    # collapse the cycle to a deterministic representative rather than aborting the build.
    if keep_stereo:
        return min(seen)
    raise ValueError(
        f"SMILES did not reach a canonical fixed point in {MAX_PASSES} passes: {smiles}"
    )
