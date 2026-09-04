"""
UniqueMultiplicity fingerprint: uncapped multiplicity de-duplicated by atom set, with
fragments re-canonicalised so one substructure is one bit.

Fixes two over-counts in the plain multiplicity vocabulary:
  * per-atom-environment counting (symmetric groups counted once per equivalent atom),
  * MolFragmentToSmiles spelling the same fragment differently from different centres
    (e.g. an amide as CC(N)=O and CC(=O)N), splitting it across two bits.
"""
import os
os.environ.setdefault("DATASET_ROOT", "/tmp")

from rdkit import Chem

from src.modules.data.fp_utils import (
    extract_features, _substructure_occurrences,
    MULTIPLICITY_UNCAPPED, UNIQUE_MULTIPLICITY,
)

# Kavaratamide A (2D) — has an amide (two spellings under uncapped) and isopropyls (symmetric).
KAVA = "CCCCCCCC(O)CC(=O)NC(C(=O)N(C)C(C)C(=O)OC(C(=O)N1C(=O)C=C(OC)C1C(C)C)C(C)C)C(C)C"
R = 10


def _frags(kind):
    return {f for (f, _k) in extract_features(KAVA, R, kind=kind)}


def test_unique_keys_are_all_canonical():
    for frag in _frags(UNIQUE_MULTIPLICITY):
        assert Chem.CanonSmiles(frag) == frag, f"{frag!r} is not canonical"


def test_amide_spelling_collapses_to_one_bit():
    uncapped = _frags(MULTIPLICITY_UNCAPPED)
    unique = _frags(UNIQUE_MULTIPLICITY)
    # The uncapped vocabulary carries the amide under both spellings; unique under one.
    assert {"CC(N)=O", "CC(=O)N"} <= uncapped
    assert ("CC(=O)N" not in unique) and ("CC(N)=O" in unique)


def test_unique_has_fewer_features_than_uncapped():
    assert len(_frags(UNIQUE_MULTIPLICITY)) < len(_frags(MULTIPLICITY_UNCAPPED))


def test_atom_set_dedup_collapses_symmetric_and_multi_centre_counts():
    mol = Chem.MolFromSmiles(KAVA)
    from collections import Counter
    plain = Counter(f for _, f, _ in _substructure_occurrences(mol, R, dedup=False))
    uniq = Counter(f for _, f, _ in _substructure_occurrences(mol, R, dedup=True))
    # Per-atom counting over-counts one isopropyl 3x and the amide fragment 2x.
    assert plain["CC(N)C(C)C"] == 3
    assert plain["CC(=O)N"] == 2
    # dedup+re-canon collapses to distinct groups under the canonical spelling: the two
    # N-adjacent valines, and the four amide groups (spelling variants merged in).
    assert uniq[Chem.CanonSmiles("CC(N)C(C)C")] == 2
    assert uniq[Chem.CanonSmiles("CC(=O)N")] == 4
    # and always fewer total than the per-atom count.
    assert sum(uniq.values()) < sum(plain.values())


def test_deployed_multiplicity_uncapped_is_unchanged():
    # dedup defaults off, so the existing (deployed) vocabulary is byte-for-byte the same.
    mol = Chem.MolFromSmiles(KAVA)
    a = list(_substructure_occurrences(mol, R))                 # default dedup=False
    b = list(_substructure_occurrences(mol, R, dedup=False))
    assert a == b
