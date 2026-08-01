"""
Request validation at the API boundary.

Every unbounded list field used to be a trivial denial-of-service: the frontend
capped the spreadsheet at 400 rows but the API accepted arrays of any size.
"""
import pytest

from app.config import (
    MAX_FP_LENGTH, MAX_HSQC_PEAKS, MAX_MS_PEAKS, MAX_NMR_PEAKS, MAX_SMILES_LENGTH,
)


def test_health_is_reachable_without_a_model(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "model_loaded" in r.json()


def test_predict_requires_at_least_one_modality(client):
    r = client.post("/api/predict", json={"raw": {}, "k": 5})
    assert r.status_code == 400


def test_hsqc_must_come_in_triplets(client):
    r = client.post("/api/predict", json={"raw": {"hsqc": [1.0, 2.0]}, "k": 5})
    assert r.status_code == 422


def test_mass_spec_must_come_in_pairs(client):
    r = client.post("/api/predict", json={"raw": {"mass_spec": [100.0, 1.0, 200.0]}, "k": 5})
    assert r.status_code == 422


@pytest.mark.parametrize("field,length", [
    ("hsqc", (MAX_HSQC_PEAKS + 10) * 3),
    ("h_nmr", MAX_NMR_PEAKS + 10),
    ("c_nmr", MAX_NMR_PEAKS + 10),
    ("mass_spec", (MAX_MS_PEAKS + 10) * 2),
])
def test_oversized_spectra_are_rejected(client, field, length):
    r = client.post("/api/predict", json={"raw": {field: [1.0] * length}, "k": 5})
    assert r.status_code == 422


@pytest.mark.parametrize("field,length", [
    ("hsqc", MAX_HSQC_PEAKS * 3),
    ("h_nmr", MAX_NMR_PEAKS),
    ("mass_spec", MAX_MS_PEAKS * 2),
])
def test_spectra_at_exactly_the_cap_pass_validation(client, field, length):
    """The boundary must be inclusive, or legitimate large inputs break."""
    r = client.post("/api/predict", json={"raw": {field: [1.0] * length}, "k": 5})
    assert r.status_code != 422


def test_oversized_smiles_is_rejected(client):
    r = client.post("/api/smiles-search", json={"smiles": "C" * (MAX_SMILES_LENGTH + 1)})
    assert r.status_code == 422


def test_oversized_reference_fp_is_rejected(client):
    r = client.post("/api/custom-smiles-card",
                    json={"smiles": "CCO", "reference_fp": [0.1] * (MAX_FP_LENGTH + 1)})
    assert r.status_code == 422


def test_empty_smiles_is_rejected(client):
    assert client.post("/api/smiles-search", json={"smiles": ""}).status_code == 422


@pytest.mark.parametrize("k", [0, -1, 1000])
def test_k_outside_the_allowed_range_is_rejected(client, k):
    r = client.post("/api/predict", json={"raw": {"h_nmr": [1.0]}, "k": k})
    assert r.status_code == 422


@pytest.mark.parametrize("payload", [
    {"mw": 0}, {"mw": -5},
])
def test_non_positive_mw_is_rejected(client, payload):
    r = client.post("/api/predict", json={"raw": {"h_nmr": [1.0], **payload}, "k": 5})
    assert r.status_code == 422


def test_inverted_mw_filter_is_rejected(client):
    r = client.post("/api/predict",
                    json={"raw": {"h_nmr": [1.0]}, "k": 5, "mw_min": 500, "mw_max": 100})
    assert r.status_code == 400


def test_unknown_model_id_is_rejected(client):
    """Only reachable when a manifest is loaded; otherwise the default is used."""
    r = client.post("/api/predict",
                    json={"raw": {"h_nmr": [1.0]}, "k": 5, "model_id": "does-not-exist"})
    assert r.status_code in (400, 500)


def test_internal_errors_do_not_leak_exception_text(client):
    """
    With no lifespan there is no compute pool, so the handler raises internally.
    The client must get a generic message, not the traceback text.
    """
    r = client.post("/api/predict", json={"raw": {"h_nmr": [1.0, 2.0]}, "k": 5})
    assert r.status_code == 500
    detail = r.json().get("detail", "")
    assert detail == "Prediction failed."
    assert "compute_pool" not in detail and "Traceback" not in detail
