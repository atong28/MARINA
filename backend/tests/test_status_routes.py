"""/api/queue, /api/queue/{id} and /api/stats."""
import pytest


def test_queue_reports_idle_when_no_pool_is_configured(client):
    r = client.get("/api/queue")
    assert r.status_code == 200
    body = r.json()
    assert body["pooled"] is False
    assert body["queued"] == 0


def test_queue_position_for_an_unknown_id_is_answerable(client):
    """Polling must not 404 once the job finishes and leaves the registry."""
    r = client.get("/api/queue/never-submitted")
    assert r.status_code == 200
    assert "state" in r.json()


def test_stats_shape(client):
    body = client.get("/api/stats").json()
    assert set(body) == {"queries_total", "by_kind", "counting_since"}
    assert isinstance(body["queries_total"], int)


def test_rejected_requests_do_not_inflate_the_counters(client):
    before = client.get("/api/stats").json()["queries_total"]
    client.post("/api/predict", json={"raw": {"hsqc": [1.0]}, "k": 5})   # 422
    client.post("/api/smiles-search", json={"smiles": ""})               # 422
    assert client.get("/api/stats").json()["queries_total"] == before


@pytest.mark.parametrize("path", ["/api/queue", "/api/stats"])
def test_status_endpoints_are_not_rate_limited(client, path):
    """The browser polls /api/queue every 1.5 s while a prediction is running."""
    assert {client.get(path).status_code for _ in range(40)} == {200}
