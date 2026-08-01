"""
app.session internals: CSR row filtering, the MW index, and the bounded
MW-filter cache.

_filter_csr_rows is property-tested against a straightforward reference because
the shipped version is a vectorised gather — the readable loop it replaced cost
seconds on a database-sized rankingset.
"""
import logging

import pytest
import torch

from app.session import MAX_FILTERED_CACHE, _filter_csr_rows, _warn_if_not_normalized


def _reference_filter(store, kept):
    """The obvious per-row implementation, used as the oracle."""
    if not kept:
        return torch.zeros(0, store.shape[1])
    return store.to_dense()[kept]


@pytest.mark.parametrize("seed", range(25))
def test_filter_csr_rows_matches_the_reference(seed):
    torch.manual_seed(seed)
    n_rows = int(torch.randint(1, 12, (1,)))
    n_cols = int(torch.randint(2, 20, (1,)))
    dense = (torch.rand(n_rows, n_cols) < 0.35).float() * torch.rand(n_rows, n_cols)
    if seed % 5 == 0:                                  # exercise empty rows
        dense[int(torch.randint(0, n_rows, (1,)))] = 0.0

    store = dense.to_sparse_csr()
    n_keep = int(torch.randint(0, n_rows + 1, (1,)))
    kept = sorted(torch.randperm(n_rows)[:n_keep].tolist())

    got = _filter_csr_rows(store, kept)
    assert got.shape == (len(kept), n_cols)
    assert torch.allclose(got.to_dense(), _reference_filter(store, kept))


def test_filter_csr_rows_with_no_kept_indices():
    store = torch.rand(4, 5).to_sparse_csr()
    assert _filter_csr_rows(store, []).shape == (0, 5)


def test_filter_csr_rows_with_all_empty_rows():
    store = torch.zeros(4, 5).to_sparse_csr()
    out = _filter_csr_rows(store, [0, 2])
    assert out.shape == (2, 5)
    assert out.to_dense().sum() == 0


def test_filter_csr_rows_preserves_order_and_duplicates():
    dense = torch.eye(4)
    out = _filter_csr_rows(dense.to_sparse_csr(), [2, 0])
    assert torch.allclose(out.to_dense(), dense[[2, 0]])


def test_filter_accepts_dense_stores_too():
    dense = torch.rand(5, 6)
    assert torch.allclose(_filter_csr_rows(dense, [1, 3]), dense[[1, 3]])


# ── Rankingset normalisation guard ────────────────────────────────────────────

def test_unnormalized_rankingset_is_flagged(caplog):
    """
    RankingSet's cosine metric normalises only the query, so raw 0/1 rows
    inflate every score past 1.0 where it is clamped — every card reads 1.000.
    """
    raw = (torch.rand(6, 20) < 0.4).float()
    with caplog.at_level(logging.WARNING):
        _warn_if_not_normalized(raw.to_sparse_csr(), "RankingEntropy", "/model")
    assert "unit-norm" in caplog.text


def test_normalized_rankingset_is_silent(caplog):
    rows = torch.nn.functional.normalize((torch.rand(6, 20) < 0.4).float(), dim=1)
    with caplog.at_level(logging.WARNING):
        _warn_if_not_normalized(rows.to_sparse_csr(), "RankingEntropy", "/model")
    assert "unit-norm" not in caplog.text


def test_normalisation_check_tolerates_an_empty_store(caplog):
    with caplog.at_level(logging.WARNING):
        _warn_if_not_normalized(torch.zeros(0, 5).to_sparse_csr(), "RankingEntropy", "/m")
    assert "unit-norm" not in caplog.text


# ── MW filtering and the bounded cache ────────────────────────────────────────

pytestmark_marina = pytest.mark.usefixtures("marina_src")


@pytest.mark.slow
def test_mw_filter_keeps_entries_without_a_recorded_mass(spectre_session):
    """Documented behaviour: a sparse metadata field must not hide candidates."""
    sess = spectre_session
    sess._ensure_mw_index()
    sess._mw_by_idx = {0: 100.0}
    sess._mw_sorted = [(100.0, 0)]
    sess._mw_values = [100.0]
    sess._no_mw = [1, 2]

    kept = sess.indices_in_mw_range(90.0, 110.0)
    assert kept == [0, 1, 2]


@pytest.mark.slow
def test_mw_filter_excludes_out_of_range_entries(spectre_session):
    sess = spectre_session
    sess._ensure_mw_index()
    sess._mw_by_idx = {0: 100.0, 1: 900.0}
    sess._mw_sorted = [(100.0, 0), (900.0, 1)]
    sess._mw_values = [100.0, 900.0]
    sess._no_mw = []

    assert sess.indices_in_mw_range(90.0, 110.0) == [0]


@pytest.mark.slow
def test_unfiltered_query_does_not_copy_the_rankingset(spectre_session):
    """The (None, None) case must reuse the base store, not clone it."""
    rs, kept = spectre_session.get_filtered_rankingset(None, None)
    assert rs is spectre_session.get_rankingset()
    assert len(kept) == spectre_session._num_rows
    assert spectre_session._filtered_cache == {}


@pytest.mark.slow
def test_mw_filter_cache_is_bounded(spectre_session):
    """
    Regression: the cache key comes straight from user input and each entry is
    a full copy of the rankingset, so an unbounded cache was an OOM lever.
    """
    for i in range(MAX_FILTERED_CACHE + 6):
        spectre_session.get_filtered_rankingset(float(i), float(i + 500))
    assert len(spectre_session._filtered_cache) <= MAX_FILTERED_CACHE


@pytest.mark.slow
def test_mw_filter_cache_returns_consistent_results(spectre_session):
    a_rs, a_kept = spectre_session.get_filtered_rankingset(50.0, 5000.0)
    b_rs, b_kept = spectre_session.get_filtered_rankingset(50.0, 5000.0)
    assert a_kept == b_kept
    assert b_rs is a_rs
