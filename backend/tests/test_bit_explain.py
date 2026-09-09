"""
app.bit_explain — joins predicted confidence, substructure identity and location.

The ordering is the point: a molecule has ~56 bits predicted above 0.9, so a
confidence-sorted list is almost entirely confident matches. The confident bits
the candidate *lacks* are what discriminate between candidates, and they have to
come first.
"""
import pytest

from app.bit_explain import (
    CONFIDENT, GROUP_MATCH, GROUP_MISSING, GROUP_UNCERTAIN, GROUP_UNEXPECTED,
    band, explain_bits,
)

pytestmark = pytest.mark.usefixtures("marina_src")

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
OTHER = "CCCCCCOc1ccccc1C(=O)NC"
RADIUS = 6


@pytest.fixture
def session(marina_src):
    """A session stub carrying only what explain_bits reads off the fp_loader."""
    from src.modules.data.fp_utils import get_bitinfos

    cols: dict = {}
    for smi in (ASPIRIN, OTHER):
        for bit in sorted(get_bitinfos(smi, RADIUS)[1]):
            cols.setdefault(bit, len(cols))

    class _Loader:
        max_radius = RADIUS
        bitinfo_to_fp_index_map = cols
        fp_index_to_bitinfo_map = {v: k for k, v in cols.items()}
        out_dim = len(cols)

    class _Session:
        fp_loader = _Loader()

    return _Session()


def _cols_of(session, smiles):
    from src.modules.data.fp_utils import get_bitinfos
    m = session.fp_loader.bitinfo_to_fp_index_map
    return {m[b] for b in get_bitinfos(smiles, RADIUS)[1]}


# ── Bands ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("p,expected", [
    (0.99, "Very likely"), (0.90, "Very likely"),
    (0.80, "Likely"), (0.70, "Likely"),
    (0.50, "Possible"), (0.40, "Possible"),
    (0.20, "Unlikely"), (0.00, "Unlikely"),
])
def test_band_thresholds(p, expected):
    assert band(p) == expected


# ── Location ──────────────────────────────────────────────────────────────────

def test_present_bits_carry_atom_locations(session):
    n = session.fp_loader.out_dim
    fp = [0.0] * n
    for c in _cols_of(session, ASPIRIN):
        fp[c] = 0.99

    out = explain_bits(session, ASPIRIN, fp, limit=500)
    located = [b for b in out["bits"] if b["present"]]
    assert located, "aspirin's own bits should be found in aspirin"
    for b in located:
        assert b["atoms"], f"bit {b['index']} present but has no atoms"


def test_absent_bits_have_no_location(session):
    """A bit the candidate lacks cannot be highlighted on it."""
    n = session.fp_loader.out_dim
    only_other = _cols_of(session, OTHER) - _cols_of(session, ASPIRIN)
    assert only_other, "fixture molecules must differ"

    fp = [0.0] * n
    for c in only_other:
        fp[c] = 0.99

    out = explain_bits(session, ASPIRIN, fp, limit=500)
    for b in out["bits"]:
        if not b["present"]:
            assert b["atoms"] == [] and b["bonds"] == []


def test_multi_occurrence_bit_unions_all_sites(session):
    """Aspirin's C=O environment fires at both carbonyls; both must be highlighted."""
    from src.modules.data.fp_utils import get_bitinfos

    atom_to_bits, _ = get_bitinfos(ASPIRIN, RADIUS)
    cols = session.fp_loader.bitinfo_to_fp_index_map
    counts: dict = {}
    for bits in atom_to_bits.values():
        for b in bits:
            counts.setdefault(cols[b], set())
    for atom, bits in atom_to_bits.items():
        for b in bits:
            counts[cols[b]].add(atom)
    multi = [c for c, atoms in counts.items() if len(atoms) > 1]
    assert multi, "expected at least one bit firing at several centres"

    n = session.fp_loader.out_dim
    fp = [0.0] * n
    for c in multi:
        fp[c] = 0.99
    out = explain_bits(session, ASPIRIN, fp, limit=500)
    by_index = {b["index"]: b for b in out["bits"]}
    for c in multi:
        assert len(by_index[c]["atoms"]) >= len(counts[c])
        # Each site is also kept apart, so overlapping instances can be told apart.
        occ = by_index[c]["occurrences"]
        assert len(occ) >= 2
        assert set().union(*(set(o["atoms"]) for o in occ)) == set(by_index[c]["atoms"])


# ── Grouping and ordering ─────────────────────────────────────────────────────

def test_groups_are_assigned_by_confidence_and_presence(session):
    n = session.fp_loader.out_dim
    aspirin_cols = sorted(_cols_of(session, ASPIRIN))
    absent = sorted(set(range(n)) - set(aspirin_cols))
    assert len(aspirin_cols) >= 2 and len(absent) >= 2

    fp = [0.0] * n
    fp[aspirin_cols[0]] = 0.99   # confident + present  -> match
    fp[aspirin_cols[1]] = 0.02   # unconfident + present -> unexpected
    fp[absent[0]] = 0.99         # confident + absent   -> missing
    fp[absent[1]] = 0.30         # unconfident + absent -> uncertain

    got = {b["index"]: b["group"] for b in explain_bits(session, ASPIRIN, fp, limit=500)["bits"]}
    assert got[aspirin_cols[0]] == GROUP_MATCH
    assert got[aspirin_cols[1]] == GROUP_UNEXPECTED
    assert got[absent[0]] == GROUP_MISSING
    assert got[absent[1]] == GROUP_UNCERTAIN


def test_missing_bits_are_ranked_above_matches(session):
    n = session.fp_loader.out_dim
    aspirin_cols = sorted(_cols_of(session, ASPIRIN))
    absent = sorted(set(range(n)) - set(aspirin_cols))

    fp = [0.0] * n
    for c in aspirin_cols:
        fp[c] = 0.999            # many very confident matches
    fp[absent[0]] = 0.60         # one *less* confident miss

    bits = explain_bits(session, ASPIRIN, fp, limit=500)["bits"]
    assert bits[0]["index"] == absent[0], "the confident miss must lead despite lower confidence"
    assert bits[0]["group"] == GROUP_MISSING


def test_within_group_sorted_by_confidence(session):
    n = session.fp_loader.out_dim
    fp = [0.0] * n
    for i, c in enumerate(sorted(_cols_of(session, ASPIRIN))):
        fp[c] = 0.90 + 0.001 * i

    matches = [b for b in explain_bits(session, ASPIRIN, fp, limit=500)["bits"]
               if b["group"] == GROUP_MATCH]
    confs = [b["confidence"] for b in matches]
    assert confs == sorted(confs, reverse=True)


# ── Truncation and calibration ────────────────────────────────────────────────

def test_totals_count_all_rows_not_just_returned(session):
    n = session.fp_loader.out_dim
    fp = [0.0] * n
    for c in _cols_of(session, ASPIRIN):
        fp[c] = 0.99

    full = explain_bits(session, ASPIRIN, fp, limit=500)
    clipped = explain_bits(session, ASPIRIN, fp, limit=3)
    assert len(clipped["bits"]) == 3
    assert clipped["total_available"] == full["total_available"]
    assert clipped["totals"] == full["totals"]


def test_low_confidence_absent_bits_are_dropped(session):
    """Otherwise every molecule reports all 16,384 rows."""
    n = session.fp_loader.out_dim
    fp = [0.001] * n              # below MIN_CONFIDENCE, and mostly absent
    out = explain_bits(session, ASPIRIN, fp, limit=500)
    assert all(b["present"] for b in out["bits"])


def test_calibrator_is_applied_and_flagged(session):
    n = session.fp_loader.out_dim
    col = sorted(_cols_of(session, ASPIRIN))[0]
    fp = [0.0] * n
    fp[col] = 0.9

    raw = explain_bits(session, ASPIRIN, fp, limit=500)
    cal = explain_bits(session, ASPIRIN, fp, limit=500, calibrator=lambda p: p * 0.5)

    assert raw["calibrated"] is False
    assert cal["calibrated"] is True
    row_raw = next(b for b in raw["bits"] if b["index"] == col)
    row_cal = next(b for b in cal["bits"] if b["index"] == col)
    assert row_raw["confidence"] == pytest.approx(0.9)
    assert row_cal["confidence"] == pytest.approx(0.45)
    assert row_cal["raw_confidence"] == pytest.approx(0.9), "raw value stays available"


def test_calibration_can_move_a_bit_across_the_confident_threshold(session):
    """Grouping keys off the calibrated value, so recalibration changes the panel."""
    n = session.fp_loader.out_dim
    absent = sorted(set(range(n)) - _cols_of(session, ASPIRIN))[0]
    fp = [0.0] * n
    fp[absent] = 0.8

    raw = explain_bits(session, ASPIRIN, fp, limit=500)
    cal = explain_bits(session, ASPIRIN, fp, limit=500, calibrator=lambda p: p * 0.5)
    assert next(b for b in raw["bits"] if b["index"] == absent)["group"] == GROUP_MISSING
    assert next(b for b in cal["bits"] if b["index"] == absent)["group"] == GROUP_UNCERTAIN
    assert CONFIDENT == 0.5


def test_invalid_smiles_yields_no_locations(session):
    n = session.fp_loader.out_dim
    fp = [0.9] * 3 + [0.0] * (n - 3)
    out = explain_bits(session, "not-a-smiles", fp, limit=500)
    assert all(not b["present"] for b in out["bits"])
