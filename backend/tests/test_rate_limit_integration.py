"""
Throttling as wired into the app.

The middleware reads its config once at construction, so these tests build a
fresh app with the limits set rather than mutating the shared one.
"""
import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def throttled_client(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PREDICT", "5 per minute")
    monkeypatch.setenv("RATE_LIMIT_SMILES", "3 per minute")

    import app.config
    import app.rate_limit
    import app.main
    for mod in (app.config, app.rate_limit, app.main):
        importlib.reload(mod)
    yield TestClient(app.main.app)

    # Restore the shared modules for the rest of the session.
    monkeypatch.undo()
    for mod in (app.config, app.rate_limit, app.main):
        importlib.reload(mod)


def _predict(c):
    return c.post("/api/predict", json={"raw": {"h_nmr": [1.0, 2.0]}, "k": 5})


def test_requests_are_throttled_after_the_limit(throttled_client):
    codes = [_predict(throttled_client).status_code for _ in range(9)]
    assert 429 in codes
    assert codes.index(429) <= 5, f"limiter engaged too late: {codes}"


def test_throttled_response_carries_retry_after(throttled_client):
    for _ in range(9):
        r = _predict(throttled_client)
        if r.status_code == 429:
            assert int(r.headers["Retry-After"]) >= 1
            return
    pytest.fail("limiter never engaged")


def test_endpoints_have_independent_budgets(throttled_client):
    for _ in range(9):
        _predict(throttled_client)
    assert _predict(throttled_client).status_code == 429
    # A different rule, so its budget is untouched.
    assert throttled_client.get("/api/health").status_code == 200


def test_separate_clients_are_not_throttled_together(throttled_client):
    for _ in range(9):
        _predict(throttled_client)
    assert _predict(throttled_client).status_code == 429

    other = throttled_client.post(
        "/api/predict",
        json={"raw": {"h_nmr": [1.0, 2.0]}, "k": 5},
        headers={"X-Forwarded-For": "198.51.100.77"},
    )
    assert other.status_code != 429


def test_limits_are_off_when_unset(client):
    """The default config ships with limits disabled for local development."""
    codes = {_predict(client).status_code for _ in range(20)}
    assert 429 not in codes


def test_smiles_endpoints_share_one_budget(throttled_client):
    """Spreading requests over the SMILES endpoints must not multiply the limit (3/min)."""
    # No model is loaded here, so the endpoints themselves fail; only the limiter matters.
    c = TestClient(throttled_client.app, raise_server_exceptions=False)
    body = {"smiles": "CCO", "pred_fp": [0.5], "reference_fp": [0.5]}
    paths = ["/api/smiles-search", "/api/custom-smiles-card", "/api/fingerprints/explain"]
    codes = [c.post(p, json=body).status_code for p in paths]
    assert 429 not in codes
    assert c.post("/api/fingerprints/indices", json=body).status_code == 429
