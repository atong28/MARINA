"""app.stats — usage counters and persistence."""
import json

from app.stats import UsageStats


def test_counts_are_split_by_kind_and_totalled():
    s = UsageStats(None)
    for _ in range(3):
        s.record("predict")
    s.record("smiles_search")
    s.record("custom_card")

    snap = s.snapshot()
    assert snap["queries_total"] == 5
    assert snap["by_kind"] == {"predict": 3, "smiles_search": 1, "custom_card": 1}


def test_unknown_kinds_are_ignored():
    s = UsageStats(None)
    s.record("not_a_real_kind")
    assert s.snapshot()["queries_total"] == 0


def test_nothing_about_the_caller_is_reported():
    """Only counters are exposed — there is no per-visitor state to leak."""
    s = UsageStats(None)
    s.record("predict")
    assert set(s.snapshot()) == {"queries_total", "by_kind", "counting_since"}


def test_only_counters_are_written_to_disk(tmp_path):
    path = tmp_path / "stats.json"
    s = UsageStats(str(path), flush_interval=0.0)
    s.record("predict")

    raw = json.loads(path.read_text())
    assert set(raw) == {"counts", "started_at"}
    assert raw["counts"]["predict"] == 1


def test_counters_survive_a_restart(tmp_path):
    path = tmp_path / "stats.json"
    first = UsageStats(str(path), flush_interval=0.0)
    first.record("predict")
    first.record("predict")

    second = UsageStats(str(path))
    assert second.snapshot()["queries_total"] == 2


def test_unwritable_path_degrades_to_memory_instead_of_raising():
    """A read-only mount is a normal deployment, not a request-failing error."""
    s = UsageStats("/proc/nonexistent/marina/stats.json", flush_interval=0.0)
    s.record("predict")
    assert s.snapshot()["queries_total"] == 1


def test_corrupt_stats_file_does_not_prevent_startup(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text("{ this is not json")
    s = UsageStats(str(path))
    assert s.snapshot()["queries_total"] == 0
