"""app.stats — usage counters, privacy, and persistence."""
import json

import pytest

from app.stats import UsageStats


def test_counts_are_split_by_kind_and_totalled(tmp_path):
    s = UsageStats(None)
    for _ in range(3):
        s.record("predict", "1.2.3.4")
    s.record("smiles_search", "1.2.3.4")
    s.record("custom_card", "5.6.7.8")

    snap = s.snapshot()
    assert snap["queries_total"] == 5
    assert snap["by_kind"] == {"predict": 3, "smiles_search": 1, "custom_card": 1}


def test_unknown_kinds_are_ignored():
    s = UsageStats(None)
    s.record("not_a_real_kind", "1.2.3.4")
    assert s.snapshot()["queries_total"] == 0


def test_unique_clients_counts_distinct_addresses():
    s = UsageStats(None)
    for ip in ["1.1.1.1", "1.1.1.1", "2.2.2.2", "3.3.3.3", "2.2.2.2"]:
        s.record("predict", ip)
    assert s.snapshot()["unique_clients"] == 3
    assert s.snapshot()["queries_total"] == 5


def test_missing_client_still_counts_the_query():
    s = UsageStats(None)
    s.record("predict", None)
    assert s.snapshot()["queries_total"] == 1
    assert s.snapshot()["unique_clients"] == 0


def test_addresses_are_never_written_to_disk(tmp_path):
    path = tmp_path / "stats.json"
    s = UsageStats(str(path), flush_interval=0.0)
    s.record("predict", "203.0.113.42")

    raw = json.loads(path.read_text())
    assert "203.0.113.42" not in path.read_text()
    assert all(len(c) == 16 and all(ch in "0123456789abcdef" for ch in c)
               for c in raw["clients"])


def test_counters_survive_a_restart(tmp_path):
    path = tmp_path / "stats.json"
    first = UsageStats(str(path), flush_interval=0.0)
    for ip in ["1.1.1.1", "2.2.2.2"]:
        first.record("predict", ip)

    second = UsageStats(str(path))
    assert second.snapshot()["queries_total"] == 2
    assert second.snapshot()["unique_clients"] == 2


def test_returning_visitor_is_not_double_counted_after_restart(tmp_path):
    """The salt persists, so the same address hashes to the same id."""
    path = tmp_path / "stats.json"
    first = UsageStats(str(path), flush_interval=0.0)
    first.record("predict", "1.1.1.1")

    second = UsageStats(str(path))
    second.record("predict", "1.1.1.1")
    assert second.snapshot()["unique_clients"] == 1
    assert second.snapshot()["queries_total"] == 2


def test_unwritable_path_degrades_to_memory_instead_of_raising():
    """A read-only mount is a normal deployment, not a request-failing error."""
    s = UsageStats("/proc/nonexistent/marina/stats.json", flush_interval=0.0)
    s.record("predict", "1.1.1.1")
    assert s.snapshot()["queries_total"] == 1


def test_corrupt_stats_file_does_not_prevent_startup(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text("{ this is not json")
    s = UsageStats(str(path))
    assert s.snapshot()["queries_total"] == 0


def test_client_set_is_bounded(monkeypatch):
    """Past the cap the unique count keeps rising but memory stays bounded."""
    import app.stats as stats_mod
    monkeypatch.setattr(stats_mod, "MAX_TRACKED_CLIENTS", 10)

    s = UsageStats(None)
    for i in range(50):
        s.record("predict", f"10.0.0.{i}")

    assert len(s._clients) == 10
    assert s.snapshot()["unique_clients"] == 50
