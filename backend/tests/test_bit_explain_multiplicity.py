"""
app.bit_explain against the multiplicity vocabulary.

A multiplicity model keys its feature map on `(fragment SMILES, occurrence bucket)` —
a 2-tuple, so neither the Morgan 4-tuple unpack nor the substructure `isinstance(str)`
branch covers it. Pointed at such a model the panel raised
`ValueError: not enough values to unpack (expected 4, got 2)` and every explain request
returned 500.
"""
import pytest

from app.bit_explain import GROUP_MATCH, explain_bits

pytestmark = pytest.mark.usefixtures("marina_src")

# Hexanoic acid: one fragment (the chain CH2) recurs, so cumulative buckets >1 exist.
CHAIN = "CCCCCC(=O)O"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
RADIUS = 6


@pytest.fixture
def mult_session(marina_src):
    """Session stub whose feature map is keyed on (fragment SMILES, bucket)."""
    from src.modules.data.fp_utils import (
        count_substructure_multiplicities, MULTIPLICITY,
    )

    cols: dict = {}
    for smi in (CHAIN, ASPIRIN):
        for feat in sorted(count_substructure_multiplicities(smi, RADIUS)):
            cols.setdefault(feat, len(cols))

    class _Loader:
        FEATURE_KIND = MULTIPLICITY
        max_radius = RADIUS
        bitinfo_to_fp_index_map = cols
        fp_index_to_bitinfo_map = {v: k for k, v in cols.items()}
        out_dim = len(cols)

    class _Session:
        fp_loader = _Loader()

    return _Session()


def test_feature_map_keys_are_two_tuples(mult_session):
    keys = list(mult_session.fp_loader.bitinfo_to_fp_index_map)
    assert keys
    assert all(isinstance(k, tuple) and len(k) == 2 for k in keys)


def test_explain_does_not_raise_on_two_tuple_features(mult_session):
    """The regression: a 2-tuple feature key used to blow up the 4-tuple unpack."""
    n = mult_session.fp_loader.out_dim
    fp = [0.0] * n
    for col, feat in mult_session.fp_loader.fp_index_to_bitinfo_map.items():
        fp[col] = 0.99

    out = explain_bits(mult_session, CHAIN, fp, limit=500)
    assert out["bits"], "expected bits for a fully-predicted fingerprint"


def test_rows_carry_the_fragment_and_no_centre_atom(mult_session):
    n = mult_session.fp_loader.out_dim
    fp = [0.0] * n
    for col, feat in mult_session.fp_loader.fp_index_to_bitinfo_map.items():
        fp[col] = 0.99

    rows = explain_bits(mult_session, CHAIN, fp, limit=500)["bits"]
    matched = [r for r in rows if r["group"] == GROUP_MATCH]
    assert matched, "expected some present-and-confident bits"
    for r in matched:
        # The fragment SMILES comes from the feature key, never the bucket integer.
        assert isinstance(r["fragment_smiles"], str) and r["fragment_smiles"]
        assert r["atom_symbol"] == ""
        assert r["radius"] >= 0


def test_cumulative_buckets_share_a_fragment_across_columns(mult_session):
    """A recurring fragment lights several columns that differ only in bucket."""
    cols = mult_session.fp_loader.bitinfo_to_fp_index_map
    by_frag: dict = {}
    for (frag, bucket) in cols:
        by_frag.setdefault(frag, set()).add(bucket)
    assert any(len(b) > 1 for b in by_frag.values()), (
        "test molecule should contain a fragment occurring more than once"
    )


def test_rows_collapse_buckets_into_one_thermometer_per_fragment(mult_session):
    """Multiplicity rows collapse a fragment's cumulative buckets into a single row
    carrying the whole ≥1×, ≥2×, … thermometer (buckets) plus its true_count."""
    n = mult_session.fp_loader.out_dim
    fp = [0.99] * n

    rows = explain_bits(mult_session, CHAIN, fp, limit=500)["bits"]
    assert rows
    # Collapsed: one row per fragment, the bare per-bit bucket gone, thermometer present.
    assert all(r["multiplicity"] is None for r in rows)
    assert all(r["buckets"] is not None and r["true_count"] is not None for r in rows)
    frags = [r["fragment_smiles"] for r in rows]
    assert len(frags) == len(set(frags)), "each fragment appears once (collapsed)"

    multi = [r for r in rows if len(r["buckets"]) > 1]
    assert multi, "a recurring fragment should carry several buckets in one row"
    for r in multi:
        levels = sorted(b["level"] for b in r["buckets"])
        assert levels == list(range(1, max(levels) + 1)), (
            f"buckets for {r['fragment_smiles']!r} should be cumulative from 1, got {levels}"
        )
        # true_count is how far up the thermometer the candidate actually reaches.
        present_levels = [b["level"] for b in r["buckets"] if b["present"]]
        assert r["true_count"] == (max(present_levels) if present_levels else 0)


def test_multiplicity_rows_carry_a_sortable_radius(mult_session):
    """
    A multiplicity feature has no radius in its key, so one is recovered from the
    fragment SMILES (graph radius) for *every* fragment — present or absent — so the
    panel can order small fragments before large ones. Single-atom fragments are r0.
    """
    n = mult_session.fp_loader.out_dim
    fp = [0.99] * n

    rows = explain_bits(mult_session, ASPIRIN, fp, limit=500)["bits"]
    assert any(not r["present"] for r in rows), "predicting every bit should leave some absent"
    assert all(r["radius"] >= 0 for r in rows), "every fragment reports a computed radius"
    # rows come back ordered by radius ascending
    radii = [r["radius"] for r in rows]
    assert radii == sorted(radii)
