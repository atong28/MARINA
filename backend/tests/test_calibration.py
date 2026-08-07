"""
app.calibration — the monotone lookup table that turns raw sigmoid outputs into
probabilities a user can be shown.

The measured correction is two-sided (over-confident above ~0.45, under-confident
below), so tests that only check "the number goes down" would pass on a broken
one-sided curve.
"""
import json

import pytest

from app.calibration import Calibrator, load_calibrator


@pytest.fixture(autouse=True)
def clear_cache():
    from app import calibration
    calibration._cache.clear()
    yield
    calibration._cache.clear()


def _write(root, payload):
    (root / "calibration.json").write_text(json.dumps(payload))
    return str(root)


# ── Interpolation ─────────────────────────────────────────────────────────────

def test_interpolates_between_grid_points():
    cal = Calibrator([0.0, 0.5, 1.0], [0.0, 0.25, 1.0])
    assert cal(0.25) == pytest.approx(0.125)
    assert cal(0.75) == pytest.approx(0.625)


def test_grid_points_are_exact():
    cal = Calibrator([0.0, 0.5, 1.0], [0.0, 0.25, 1.0])
    for g, c in zip(cal.grid, cal.curve):
        assert cal(g) == pytest.approx(c)


def test_clamps_outside_the_fitted_range():
    cal = Calibrator([0.2, 0.8], [0.1, 0.9])
    assert cal(0.0) == pytest.approx(0.1)
    assert cal(1.0) == pytest.approx(0.9)


def test_never_exceeds_the_curve_maximum():
    """The fitted curve tops out below 1.0; the UI must not be able to claim certainty."""
    cal = Calibrator([0.0, 1.0], [0.0, 0.9984])
    assert cal(1.0) <= 0.9984
    assert cal(0.999999) <= 0.9984


def test_preserves_ordering():
    """Monotone by construction — recalibration must never reorder bits."""
    cal = Calibrator([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.35, 0.47, 0.68, 0.998])
    xs = [i / 200 for i in range(201)]
    ys = [cal(x) for x in xs]
    assert all(b >= a for a, b in zip(ys, ys[1:]))


def test_correction_is_two_sided():
    """Raises low probabilities and lowers high ones, as measured on seed-2."""
    cal = Calibrator([0.0, 0.1, 0.5, 0.9, 1.0], [0.0, 0.207, 0.466, 0.773, 0.998])
    assert cal(0.1) > 0.1      # under-confident at the low end
    assert cal(0.9) < 0.9      # over-confident at the high end


# ── Loading ───────────────────────────────────────────────────────────────────

def test_loads_a_well_formed_curve(tmp_path):
    root = _write(tmp_path, {"ckpt": "final2", "grid": [0.0, 1.0], "curve": [0.0, 0.99]})
    cal = load_calibrator(root)
    assert cal is not None and cal.ckpt == "final2"
    assert cal(0.5) == pytest.approx(0.495)


def test_missing_file_returns_none(tmp_path):
    """A model without a calibration curve still serves, with raw confidences."""
    assert load_calibrator(str(tmp_path)) is None


@pytest.mark.parametrize("payload", [
    {"grid": [0.0, 1.0], "curve": [0.0]},              # length mismatch
    {"grid": [0.0], "curve": [0.0]},                   # too short to interpolate
    {"grid": [0.0, 1.0], "curve": [0.9, 0.1]},         # non-monotone
    {"grid": [0.0, 1.0]},                              # missing curve
])
def test_unusable_curves_degrade_to_none(tmp_path, payload):
    """A malformed artifact must not take the endpoint down."""
    assert load_calibrator(_write(tmp_path, payload)) is None


def test_result_is_cached(tmp_path):
    root = _write(tmp_path, {"grid": [0.0, 1.0], "curve": [0.0, 1.0]})
    first = load_calibrator(root)
    (tmp_path / "calibration.json").unlink()
    assert load_calibrator(root) is first


# ── The shipped artifact ──────────────────────────────────────────────────────

def test_shipped_artifact_matches_measured_values(tmp_path):
    """
    Guards the numbers the write-up quotes. If the curve is ever refitted these
    move, and the wiki page and any UI copy have to move with them.
    """
    import os
    artifact = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "analysis", "bit-confidence", "results", "calibration_artifact.json",
    )
    if not os.path.isfile(artifact):
        pytest.skip("calibration artifact not present in this checkout")

    with open(artifact) as f:
        d = json.load(f)
    cal = Calibrator(d["grid"], d["curve"])
    assert cal(0.9) == pytest.approx(0.773, abs=0.02)
    assert cal(0.8) == pytest.approx(0.682, abs=0.02)
    assert cal(0.1) == pytest.approx(0.207, abs=0.02)
    assert max(d["curve"]) < 1.0
