"""app.rate_limit — spec parsing, client identification, and the window."""
import pytest

from app.rate_limit import _FixedWindow, client_ip, parse_limit


class _Req:
    """Minimal stand-in for a Starlette Request."""
    def __init__(self, headers=None, host="10.0.0.1"):
        self.headers = headers or {}
        self.client = type("C", (), {"host": host})() if host else None


@pytest.mark.parametrize("spec,expected", [
    ("30 per minute", (30, 60.0)),
    ("5 per second", (5, 1.0)),
    ("100 per hour", (100, 3600.0)),
    ("  7  per  minute  ", (7, 60.0)),
    ("12 PER MINUTE", (12, 60.0)),
])
def test_parse_limit_accepts_valid_specs(spec, expected):
    assert parse_limit(spec) == expected


@pytest.mark.parametrize("spec", ["", "   ", "nonsense", "30 per fortnight", "per minute", None])
def test_parse_limit_rejects_junk_without_raising(spec):
    """A malformed spec disables the limit rather than crashing startup."""
    assert parse_limit(spec) is None


def test_client_ip_prefers_the_first_forwarded_entry():
    """
    The edge proxy overwrites X-Forwarded-For with $remote_addr and the inner
    hop appends, so entry 0 is the real client.
    """
    req = _Req({"x-forwarded-for": "203.0.113.7, 172.18.0.5"})
    assert client_ip(req) == "203.0.113.7"


def test_client_ip_falls_back_to_the_socket_peer():
    assert client_ip(_Req({}, host="192.168.1.9")) == "192.168.1.9"


def test_client_ip_handles_a_missing_peer():
    assert client_ip(_Req({}, host=None)) == "unknown"


def test_window_allows_up_to_the_limit_then_blocks():
    w = _FixedWindow()
    assert all(w.allow("k", 3, 60.0, 100.0)[0] for _ in range(3))
    allowed, retry_after = w.allow("k", 3, 60.0, 100.0)
    assert not allowed
    assert retry_after > 0


def test_window_resets_after_the_period_elapses():
    w = _FixedWindow()
    for _ in range(3):
        w.allow("k", 3, 60.0, 100.0)
    assert not w.allow("k", 3, 60.0, 100.0)[0]
    assert w.allow("k", 3, 60.0, 161.0)[0]


def test_window_keys_are_independent():
    w = _FixedWindow()
    for _ in range(3):
        w.allow("a", 3, 60.0, 100.0)
    assert not w.allow("a", 3, 60.0, 100.0)[0]
    assert w.allow("b", 3, 60.0, 100.0)[0]


def test_window_sweeps_expired_entries():
    """Idle clients must not accumulate forever."""
    w = _FixedWindow()
    for i in range(5000):
        w.allow(f"client-{i}", 10, 60.0, 100.0)
    w.allow("trigger", 10, 60.0, 100_000.0)
    assert len(w._hits) < 5000
