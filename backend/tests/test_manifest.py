"""
models.json validation.

A manifest entry pointing at a directory that does not exist used to load
cleanly, because nothing checked the resolved root. Startup preload re-raises
on failure, so the mistake surfaced as a crash-looping container with a
traceback from the checkpoint loader rather than from the manifest.

The concrete regression: scripts/website/download_model.sh wrote models.json
from a quoted heredoc, so "$MODEL" reached the file unexpanded.
"""
import json

import pytest

from app import manifest


@pytest.fixture(autouse=True)
def isolated_manifest(monkeypatch, tmp_path):
    """Point DATA_DIR at a scratch dir and reset the module-level cache."""
    monkeypatch.setattr(manifest, "_manifest", None)
    monkeypatch.setattr("app.config.DATA_DIR", str(tmp_path))
    return tmp_path


def write_manifest(tmp_path, models):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": models}))
    return str(path)


def test_entry_with_valid_root_loads(isolated_manifest):
    (isolated_manifest / "marina_best").mkdir()
    path = write_manifest(isolated_manifest, [
        {"id": "marina_best", "root": "marina_best", "type": "marina", "default": True},
    ])

    entries = manifest.load_models_json(path)

    assert [e.id for e in entries] == ["marina_best"]
    assert entries[0].root == str(isolated_manifest / "marina_best")
    assert entries[0].root_rel == "marina_best"


def test_entry_with_missing_root_is_skipped(isolated_manifest):
    """The download_model.sh regression: root arrives as the literal "$MODEL"."""
    path = write_manifest(isolated_manifest, [
        {"id": "$MODEL", "root": "$MODEL", "type": "marina", "default": True},
    ])

    assert manifest.load_models_json(path) == []


def test_missing_root_does_not_hide_a_usable_sibling(isolated_manifest):
    """A broken entry must not take the working model down with it."""
    (isolated_manifest / "marina_best").mkdir()
    path = write_manifest(isolated_manifest, [
        {"id": "$MODEL", "root": "$MODEL", "type": "marina", "default": True},
        {"id": "marina_best", "root": "marina_best", "type": "marina", "default": False},
    ])

    entries = manifest.load_models_json(path)

    assert [e.id for e in entries] == ["marina_best"]
    # Sole survivor is promoted, so preload has something to resolve.
    assert entries[0].default is True
    assert manifest.get_default_model_id() == "marina_best"


def test_absolute_root_outside_data_dir_is_honoured(isolated_manifest, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    path = write_manifest(isolated_manifest, [
        {"id": "external", "root": str(elsewhere), "type": "marina", "default": True},
    ])

    entries = manifest.load_models_json(path)

    assert [e.id for e in entries] == ["external"]
    assert entries[0].root == str(elsewhere)
