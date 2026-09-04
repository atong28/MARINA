"""
The fp radius must be read from the model's own count file, not hardcoded — a
radius-6 backend serving a radius-10 model builds query fingerprints at the wrong
radius, so a molecule fails to self-retrieve at cosine 1.0.
"""
from app.session import _infer_fp_radius, FP_RADIUS_DEFAULT


def test_infers_radius_from_count_filename(tmp_path):
    (tmp_path / "count_multiplicity_uncapped_under_radius_10.pkl").write_bytes(b"")
    assert _infer_fp_radius(str(tmp_path)) == 10


def test_infers_radius_6(tmp_path):
    (tmp_path / "count_hashes_under_radius_6.pkl").write_bytes(b"")
    assert _infer_fp_radius(str(tmp_path)) == 6


def test_picks_max_when_several_present(tmp_path):
    (tmp_path / "count_hashes_under_radius_6.pkl").write_bytes(b"")
    (tmp_path / "count_multiplicity_uncapped_under_radius_10.pkl").write_bytes(b"")
    assert _infer_fp_radius(str(tmp_path)) == 10


def test_falls_back_to_default_when_absent(tmp_path):
    assert _infer_fp_radius(str(tmp_path)) == FP_RADIUS_DEFAULT
