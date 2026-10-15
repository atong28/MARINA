"""
app.refine: fragment deletion / hold-fixed retrieval for hybrid compounds.

The arithmetic is checked against its definition on a real unique-multiplicity
vocabulary: deleting a region must leave exactly the occurrences that do not
touch it, and the hold-fixed shortcut must equal shifting every candidate.
"""
from collections import Counter

import pytest
import torch

pytestmark = pytest.mark.usefixtures("marina_src")

# A hybrid: a p-coumaroyl ester on a cyclohexanol (the two "halves" share an ester O).
HYBRID = "OC1CCCCC1OC(=O)C=Cc1ccc(O)cc1"
COUMARIC = "OC(=O)C=Cc1ccc(O)cc1"
CYCLOHEXANOL = "OC1CCCCC1"
RADIUS = 10


@pytest.fixture
def session(marina_src):
    from src.modules.data.fp_utils import count_substructure_multiplicities, UNIQUE_MULTIPLICITY

    cols: dict = {}
    for smi in (HYBRID, COUMARIC, CYCLOHEXANOL):
        for feat in sorted(count_substructure_multiplicities(smi, RADIUS, cap=1 << 30, dedup=True)):
            cols.setdefault(feat, len(cols))

    bank = []
    for smi in (HYBRID, COUMARIC, CYCLOHEXANOL):
        row = torch.zeros(len(cols))
        for feat in count_substructure_multiplicities(smi, RADIUS, cap=1 << 30, dedup=True):
            row[cols[feat]] = 1.0
        bank.append(row / row.norm())
    store = torch.stack(bank).to_sparse_csr()

    class _Loader:
        FEATURE_KIND = UNIQUE_MULTIPLICITY
        max_radius = RADIUS
        bitinfo_to_fp_index_map = cols
        fp_index_to_bitinfo_map = {v: k for k, v in cols.items()}
        out_dim = len(cols)

    class _RS:
        data = store

    class _Session:
        fp_loader = _Loader()

        def get_rankingset(self):
            return _RS()

    return _Session()


def _fp(session, smiles):
    from src.modules.data.fp_utils import count_substructure_multiplicities
    cols = session.fp_loader.bitinfo_to_fp_index_map
    v = torch.zeros(len(cols))
    for feat in count_substructure_multiplicities(smiles, RADIUS, cap=1 << 30, dedup=True):
        v[cols[feat]] = 1.0
    return v


def _hybrid_atoms(pattern):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(HYBRID)
    return set(mol.GetSubstructMatch(Chem.MolFromSmiles(pattern)))


def test_deleting_everything_explains_every_bit(session):
    from app.refine import _ladders, selection_counts, shift_query
    from rdkit import Chem
    n = Chem.MolFromSmiles(HYBRID).GetNumAtoms()
    p = _fp(session, HYBRID)
    q = shift_query(p, selection_counts(session, HYBRID, range(n)), _ladders(session))
    assert float(q.sum()) == 0.0


def test_deletion_leaves_exactly_the_untouched_occurrences(session):
    """Residual ladder f: >=k is on iff at least k occurrences of f avoid the selection."""
    from app.refine import _ladders, _occurrences, _parse, selection_counts, shift_query
    sel = _hybrid_atoms("C=Cc1ccc(O)cc1")
    p = _fp(session, HYBRID)
    q = shift_query(p, selection_counts(session, HYBRID, sel), _ladders(session))

    untouched = Counter(f for f, atoms in _occurrences(_parse(HYBRID), RADIUS, "unique_multiplicity")
                        if not atoms & sel)
    for (frag, k), col in session.fp_loader.bitinfo_to_fp_index_map.items():
        assert float(q[col]) == (1.0 if untouched[frag] >= k else 0.0), (frag, k)
    assert 0 < float(q.sum()) < float(p.sum())


def test_shift_reads_a_missing_rung_as_absent(session):
    from app.refine import _ladders, shift_query
    ladders = _ladders(session)
    frag, lad = next((f, l) for f, l in ladders.items() if len(l) == 1)
    p = torch.zeros(session.fp_loader.out_dim)
    p[lad[1]] = 0.9
    q = shift_query(p, {frag: 1}, ladders)
    assert float(q[lad[1]]) == 0.0


def test_hold_vectors_equal_shifting_every_candidate(session):
    from app.refine import _ladders, hold_vectors, selection_counts, shift_query
    ladders = _ladders(session)
    counts = selection_counts(session, HYBRID, _hybrid_atoms(CYCLOHEXANOL))
    q = shift_query(torch.rand(session.fp_loader.out_dim), counts, ladders)
    w, m = hold_vectors(q, counts, ladders)
    for smi in (HYBRID, COUMARIC, CYCLOHEXANOL):
        x = _fp(session, smi)
        shifted = shift_query(x, counts, ladders)          # the same ladder shift, on the candidate
        assert torch.isclose(x @ w, shifted @ q, atol=1e-5)
        assert torch.isclose(x @ m, shifted.sum(), atol=1e-5)


def test_cosine_without_edits_matches_plain_cosine(session):
    from app.refine import score_bank
    p = _fp(session, HYBRID) * 0.8
    s = score_bank(session, p, None, "cosine")
    bank = session.get_rankingset().data.to_dense()
    expected = bank @ (p / p.norm())
    assert torch.allclose(s, expected, atol=1e-5)


def test_deleting_one_half_retrieves_the_other(session):
    """Delete the coumaroyl side of the hybrid: cyclohexanol must beat coumaric acid."""
    from app.refine import _ladders, score_bank, selection_counts, shift_query
    p = _fp(session, HYBRID)
    q = shift_query(p, selection_counts(session, HYBRID, _hybrid_atoms("C(=O)C=Cc1ccc(O)cc1")),
                    _ladders(session))
    for metric in ("cosine", "containment"):
        s = score_bank(session, q, None, metric)
        assert float(s[2]) > float(s[1]), metric      # rows: hybrid, coumaric, cyclohexanol
        assert bool(((s >= 0) & (s <= 1 + 1e-6)).all())


def test_selection_rejects_out_of_range_atoms(session):
    from app.refine import selection_counts
    with pytest.raises(ValueError):
        selection_counts(session, HYBRID, [999])


def test_unsupported_for_presence_vocabularies(session):
    from app.refine import RefineUnsupported, _ladders
    session.fp_loader.FEATURE_KIND = "morgan"
    with pytest.raises(RefineUnsupported):
        _ladders(session)


def test_selection_smiles(session):
    from app.refine import selection_smiles
    assert selection_smiles(HYBRID, sorted(_hybrid_atoms(CYCLOHEXANOL))) == "OC1CCCCC1"


def test_shared_core_maps_onto_target_or_reports_absence():
    from app.refine import shared_cores
    pool = ["CCOC(=O)C=Cc1ccc(O)cc1", "OC1CCCCC1OC(=O)C=Cc1ccc(O)cc1", "COC(=O)C=Cc1ccc(O)cc1"]
    n, cores = shared_cores(pool, HYBRID)
    assert n == 3 and cores
    full = cores[0]
    assert full["coverage"] == 1.0 and full["molecules_with_core"] == 3
    assert full["atoms"] and len(full["atoms"]) == full["num_atoms"]
    _, cores = shared_cores(pool, "CCCCCC")
    assert all(c["atoms"] is None for c in cores)
