"""
Input preprocessing and SPECTRE collation.

The collation contract is asserted against src/modules/core/const.py rather than
hard-coded numbers, so a change to the type codes on the model side fails here
instead of silently producing garbage predictions.
"""
import pytest
import torch

from app.predictor import _collate_spectre, preprocess_inputs

pytestmark = pytest.mark.usefixtures("marina_src")


@pytest.fixture
def codes(marina_src):
    from src.modules.core.const import INPUT_MAP
    return INPUT_MAP


RAW = {
    "hsqc": [5.83, 78.42, 4007870.8, 2.37, 43.03, -1524417.4],
    "h_nmr": [5.83, 2.37],
    "c_nmr": [180.01, 43.03],
    "mass_spec": [104.0712, 12000.0],
    "mw": 609.0,
}


# ── preprocess_inputs ─────────────────────────────────────────────────────────

def test_hsqc_columns_are_reordered_to_carbon_first():
    """
    The API takes [¹H, ¹³C, intensity]; the encoder reads coordinate 0 with
    c_wavelength_bounds and coordinate 1 with h_wavelength_bounds.
    """
    out = preprocess_inputs({"hsqc": [5.83, 78.42, 4007870.8]})
    assert out["hsqc"].shape == (1, 3)
    assert out["hsqc"][0].tolist() == pytest.approx([78.42, 5.83, 4007870.8], rel=1e-5)


def test_one_dimensional_modalities_become_column_vectors():
    out = preprocess_inputs({"h_nmr": [1.0, 2.0, 3.0]})
    assert out["h_nmr"].shape == (3, 1)


def test_mass_spec_becomes_mz_intensity_pairs():
    out = preprocess_inputs({"mass_spec": [104.0, 12.0, 227.0, 96.0]})
    assert out["mass_spec"].shape == (2, 2)


def test_malformed_hsqc_raises_rather_than_being_dropped():
    """
    Regression: a bad modality used to be skipped with only a log line, so the
    caller got a confident prediction computed from less data than they sent.
    """
    with pytest.raises(ValueError, match="divisible by 3"):
        preprocess_inputs({"hsqc": [1.0, 2.0]})


def test_malformed_mass_spec_raises():
    with pytest.raises(ValueError, match="divisible by 2"):
        preprocess_inputs({"mass_spec": [1.0, 2.0, 3.0]})


# ── SPECTRE collation ─────────────────────────────────────────────────────────

def test_collate_produces_one_padded_three_column_stream():
    inputs, types = _collate_spectre(preprocess_inputs(RAW))
    assert inputs.ndim == 3 and inputs.shape[0] == 1 and inputs.shape[2] == 3
    assert types.shape == (1, inputs.shape[1])


def test_collate_emits_the_documented_type_codes(codes):
    inputs, types = _collate_spectre(preprocess_inputs(RAW))
    assert set(types.flatten().tolist()) == {
        codes["hsqc"], codes["c_nmr"], codes["h_nmr"], codes["mw"], codes["mass_spec"]
    }


def _rows_of_type(inputs, types, code):
    mask = types[0] == code
    return inputs[0][mask]


def test_hsqc_rows_are_carbon_proton_intensity(codes):
    inputs, types = _collate_spectre(preprocess_inputs(RAW))
    row = _rows_of_type(inputs, types, codes["hsqc"])[0]
    assert row.tolist() == pytest.approx([78.42, 5.83, 4007870.8], rel=1e-5)


def test_carbon_shifts_occupy_coordinate_zero(codes):
    inputs, types = _collate_spectre(preprocess_inputs(RAW))
    row = _rows_of_type(inputs, types, codes["c_nmr"])[0]
    assert row.tolist() == pytest.approx([180.01, 0.0, 0.0], rel=1e-5)


def test_proton_shifts_occupy_coordinate_one(codes):
    inputs, types = _collate_spectre(preprocess_inputs(RAW))
    row = _rows_of_type(inputs, types, codes["h_nmr"])[0]
    assert row.tolist() == pytest.approx([0.0, 5.83, 0.0], rel=1e-5)


def test_mass_spec_rows_are_mz_intensity_zero(codes):
    inputs, types = _collate_spectre(preprocess_inputs(RAW))
    row = _rows_of_type(inputs, types, codes["mass_spec"])[0]
    assert row.tolist() == pytest.approx([104.0712, 12000.0, 0.0], rel=1e-5)


def test_molecular_weight_is_a_single_row(codes):
    inputs, types = _collate_spectre(preprocess_inputs(RAW))
    rows = _rows_of_type(inputs, types, codes["mw"])
    assert rows.shape[0] == 1
    assert rows[0].tolist() == pytest.approx([609.0, 0.0, 0.0])


def test_peak_count_matches_the_input():
    inputs, _ = _collate_spectre(preprocess_inputs(RAW))
    # 2 HSQC + 2 C + 2 H + 1 MS + 1 MW
    assert inputs.shape[1] == 8


@pytest.mark.parametrize("subset", [
    {"hsqc": RAW["hsqc"]},
    {"h_nmr": RAW["h_nmr"]},
    {"c_nmr": RAW["c_nmr"]},
    {"mass_spec": RAW["mass_spec"]},
    {"mw": RAW["mw"]},
])
def test_every_modality_collates_on_its_own(subset):
    """The site lets users supply any subset of the spreadsheet columns."""
    inputs, types = _collate_spectre(preprocess_inputs(subset))
    assert inputs.shape[0] == 1 and inputs.shape[2] == 3
    assert inputs.shape[1] == types.shape[1] >= 1


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="No valid spectral inputs"):
        _collate_spectre({})


def test_a_peak_summing_to_zero_would_be_masked_as_padding():
    """
    Documents a real limitation: SPECTRE.encode derives its padding mask from
    ~x.sum(dim=2).bool(), so a ¹H peak at exactly 0.0 ppm (TMS reference)
    becomes [0, 0, 0] and drops out. This matches training behaviour, so it is
    asserted rather than worked around — changing it would create train/serve
    skew and needs a retrain.
    """
    inputs, _ = _collate_spectre(preprocess_inputs({"h_nmr": [0.0, 2.37]}))
    mask = ~inputs.sum(dim=2).bool()
    assert bool(mask[0, 0]) is True
    assert bool(mask[0, 1]) is False
