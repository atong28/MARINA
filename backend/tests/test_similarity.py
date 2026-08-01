"""app.similarity — the shared metric helpers."""
import math

import pytest
import torch

from app.similarity import cosine, tanimoto


def test_cosine_of_identical_vectors_is_one():
    v = torch.tensor([1.0, 0.0, 1.0, 1.0])
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_matches_the_closed_form():
    a, b = torch.tensor([1.0, 0.0, 1.0]), torch.tensor([1.0, 1.0, 1.0])
    assert cosine(a, b) == pytest.approx(2 / (math.sqrt(2) * math.sqrt(3)), abs=1e-6)


def test_tanimoto_of_identical_binary_vectors_is_one():
    v = torch.tensor([1.0, 0.0, 1.0])
    assert tanimoto(v, v) == pytest.approx(1.0)


def test_tanimoto_matches_the_closed_form():
    # |A∩B| = 1, |A∪B| = 3  ->  1/3
    a, b = torch.tensor([1.0, 1.0, 0.0]), torch.tensor([0.0, 1.0, 1.0])
    assert tanimoto(a, b) == pytest.approx(1 / 3, abs=1e-6)


@pytest.mark.parametrize("fn", [cosine, tanimoto])
def test_length_mismatch_is_rejected_not_truncated(fn):
    """
    Regression: both helpers used to slice to the shorter vector, so a
    reference_fp from a different model scored against a truncated prefix
    instead of being rejected.
    """
    with pytest.raises(ValueError, match="length mismatch"):
        fn(torch.rand(8), torch.rand(16))


@pytest.mark.parametrize("fn", [cosine, tanimoto])
def test_zero_vector_gives_zero_not_nan(fn):
    a, b = torch.zeros(4), torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert fn(a, b) == 0.0


@pytest.mark.parametrize("fn", [cosine, tanimoto])
def test_result_is_clamped_to_unit_interval(fn):
    a, b = torch.rand(32) * 10, torch.rand(32) * 10
    assert 0.0 <= fn(a, b) <= 1.0


def test_negative_cosine_is_clamped_to_zero():
    assert cosine(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])) == 0.0
