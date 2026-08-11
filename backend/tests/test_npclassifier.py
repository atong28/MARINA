"""
NPClassifier annotations: loading the interned file, de-interning a lookup, and the
result-card shape.

The file interns its label strings, so most of what can go wrong is an id that does not
resolve to a name. A wrong class name on a result card is worse than no annotation, so
the loader rejects a payload it cannot read and the accessor drops out-of-range ids.
"""
import json

import pytest

from app import session as session_module
from app.result_builder import _npclassifier as card_npclassifier
from app.session import ModelSession, _load_npclassifier


def _payload(entries, labels=None):
    return {
        "schema": "npclassifier/1",
        "tiers": ["pathway", "superclass", "class"],
        "labels": labels or {
            "pathway": ["Terpenoids", "Alkaloids"],
            "superclass": ["Triterpenoids"],
            "class": ["Cucurbitane triterpenoids", "Isocoumarins"],
        },
        "entries": entries,
    }


def _session(payload):
    """A ModelSession with only the fields these accessors touch."""
    return ModelSession(
        model_type="marina", model=None, fp_loader=None, fp_type="RankingEntropy",
        model_root="/nonexistent", device=None, _metadata={}, _npclassifier=payload,
    )


# ── loading ───────────────────────────────────────────────────────────────────

def test_missing_file_disables_the_feature(tmp_path):
    assert _load_npclassifier(str(tmp_path / "absent.json")) is None


def test_unreadable_file_is_ignored_rather_than_fatal(tmp_path):
    p = tmp_path / "npclassifier.json"
    p.write_text("{not json")
    assert _load_npclassifier(str(p)) is None


def test_payload_missing_a_label_table_is_rejected(tmp_path):
    """A tier present in `tiers` but absent from `labels` would raise at lookup time."""
    bad = _payload({"0": [[0], [], [], 0]})
    del bad["labels"]["superclass"]
    p = tmp_path / "npclassifier.json"
    p.write_text(json.dumps(bad))
    assert _load_npclassifier(str(p)) is None


def test_a_worker_that_disabled_annotations_skips_a_present_file(tmp_path, monkeypatch):
    """Compute workers never build cards, so they must not pay the ~270 MB."""
    # Patched first so the flag is restored even though disable_annotations is global.
    monkeypatch.setattr(session_module, "_annotations_enabled", True)
    p = tmp_path / "npclassifier.json"
    p.write_text(json.dumps(_payload({"7": [[0], [0], [0], 1]})))
    assert _load_npclassifier(str(p)) is not None      # loads by default

    session_module.disable_annotations()
    assert _load_npclassifier(str(p)) is None


def test_round_trip(tmp_path):
    p = tmp_path / "npclassifier.json"
    p.write_text(json.dumps(_payload({"7": [[0], [0], [0], 1]})))
    loaded = _load_npclassifier(str(p))
    assert loaded is not None
    assert _session(loaded).get_npclassifier(7) == {
        "pathway": ["Terpenoids"],
        "superclass": ["Triterpenoids"],
        "class": ["Cucurbitane triterpenoids"],
        "isglycoside": True,
    }


# ── lookup ────────────────────────────────────────────────────────────────────

def test_no_annotations_loaded_returns_none():
    assert _session(None).get_npclassifier(0) is None


def test_index_absent_from_the_file_returns_none():
    assert _session(_payload({"0": [[0], [], [], 0]})).get_npclassifier(999) is None


def test_multi_label_tiers_are_preserved():
    got = _session(_payload({"3": [[0, 1], [], [1], 0]})).get_npclassifier(3)
    assert got["pathway"] == ["Terpenoids", "Alkaloids"]
    assert got["class"] == ["Isocoumarins"]
    assert got["isglycoside"] is False


def test_declined_classification_is_empty_not_none():
    """NPClassifier returning nothing is a fact about the molecule, not a missing row."""
    got = _session(_payload({"5": [[], [], [], 0]})).get_npclassifier(5)
    assert got == {"pathway": [], "superclass": [], "class": [], "isglycoside": False}


def test_out_of_range_label_id_is_dropped():
    got = _session(_payload({"1": [[0, 99], [], [], 0]})).get_npclassifier(1)
    assert got["pathway"] == ["Terpenoids"]


# ── result card ───────────────────────────────────────────────────────────────

def test_card_renames_class_to_npclass():
    """`class` is reserved in the TypeScript client, so the wire name differs."""
    card = card_npclassifier(_session(_payload({"7": [[0], [0], [0], 1]})), 7)
    assert card == {
        "pathway": ["Terpenoids"],
        "superclass": ["Triterpenoids"],
        "npclass": ["Cucurbitane triterpenoids"],
        "isglycoside": True,
    }
    assert "class" not in card


def test_card_is_none_without_annotations():
    assert card_npclassifier(_session(None), 0) is None


def test_card_tolerates_a_session_without_the_accessor():
    """SPECTRE sessions and older test doubles predate get_npclassifier."""
    class Bare:
        pass
    assert card_npclassifier(Bare(), 0) is None


@pytest.mark.parametrize("idx", [0, 7, 123456])
def test_lookup_is_keyed_on_the_string_index(idx):
    """metadata.json is keyed by str(index); this file must match or every lookup misses."""
    got = _session(_payload({str(idx): [[1], [], [], 0]})).get_npclassifier(idx)
    assert got["pathway"] == ["Alkaloids"]


# ── end to end ────────────────────────────────────────────────────────────────

@pytest.fixture
def model_dir_with_annotations(spectre_model_dir):
    """
    Drop an npclassifier.json into the shared synthetic model dir, then take it away.

    The directory is session-scoped, so leaving the file behind would silently change
    what every later test that loads this model sees.
    """
    path = spectre_model_dir / "npclassifier.json"
    path.write_text(json.dumps(_payload({"0": [[0], [0], [0], 1], "1": [[], [], [], 0]})))
    yield spectre_model_dir
    path.unlink()


@pytest.mark.slow
def test_annotations_reach_the_result_card(model_dir_with_annotations):
    """Through a real ModelSession and the shared card builder, not a stub."""
    import torch
    from app.result_builder import build_result_cards
    from app.session import ModelSession

    session = ModelSession.from_model_root(str(model_dir_with_annotations),
                                           model_type="spectre")
    fp = torch.zeros(session.fp_loader.out_dim, dtype=torch.float32)
    cards = build_result_cards(session, [(0, 0.9), (1, 0.8)], fp, img_size=100)

    assert cards[0]["npclassifier"] == {
        "pathway": ["Terpenoids"],
        "superclass": ["Triterpenoids"],
        "npclass": ["Cucurbitane triterpenoids"],
        "isglycoside": True,
    }
    # Classified, but NPClassifier assigned nothing — distinct from having no record.
    assert cards[1]["npclassifier"] == {
        "pathway": [], "superclass": [], "npclass": [], "isglycoside": False,
    }


@pytest.mark.slow
def test_card_carries_null_when_the_model_dir_has_no_file(spectre_model_dir):
    """The file is optional; without it the response shape is unchanged but null."""
    import torch
    from app.result_builder import build_result_cards
    from app.session import ModelSession

    assert not (spectre_model_dir / "npclassifier.json").exists()
    session = ModelSession.from_model_root(str(spectre_model_dir), model_type="spectre")
    fp = torch.zeros(session.fp_loader.out_dim, dtype=torch.float32)
    cards = build_result_cards(session, [(0, 0.9)], fp, img_size=100)

    assert cards[0]["npclassifier"] is None
