"""
Dynamic model selection: the site auto-picks the checkpoint that scores best on
the exact modalities a request carries, gated by what each model accepts.
"""
import json

import pytest

from app import manifest, model_selection


def _model_dir(root, name, input_types, measurements=None):
    """Create a fake checkpoint dir with params.json (+ optional metrics.json)."""
    d = root / name
    d.mkdir()
    (d / "params.json").write_text(json.dumps({"input_types": input_types}))
    if measurements is not None:
        (d / "metrics.json").write_text(json.dumps({
            "model_id": name, "input_types": input_types, "measurements": measurements,
        }))
    return d


def _m(modalities, mean_cos, eval_set="test", split="test_all7pop"):
    return {"modalities": modalities, "eval_set": eval_set, "split": split,
            "n": 100, "metrics": {"mean_cos": mean_cos,
                                  "strict_top1": 0, "strict_top5": 0, "strict_top10": 0}}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "_manifest", None)
    monkeypatch.setattr("app.config.DATA_DIR", str(tmp_path))
    model_selection.clear_cache()
    return tmp_path


def _load(tmp_path, models):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": models}))
    manifest.load_models_json(str(path))


def test_combo_key_is_order_independent_and_canonical():
    assert model_selection.combo_key(["h_nmr", "hsqc"]) == ("hsqc", "h_nmr")
    assert model_selection.combo_key(["mw", "hsqc"]) == ("hsqc", "mw")
    assert model_selection.combo_key({"hsqc"}) == ("hsqc",)


def test_input_types_are_read_from_params_json(isolated):
    _model_dir(isolated, "full", ["hsqc", "c_nmr", "mw"])
    _load(isolated, [{"id": "full", "root": "full", "type": "marina", "default": True}])
    assert manifest.get_model_info("full").input_types == ["hsqc", "c_nmr", "mw"]


def test_best_scoring_eligible_model_wins_for_exact_combo(isolated):
    # Two models both accept HSQC; the one that scores higher on [hsqc] is chosen.
    _model_dir(isolated, "weak",   ["hsqc", "c_nmr", "h_nmr"], [_m(["hsqc"], 0.60)])
    _model_dir(isolated, "strong", ["hsqc", "c_nmr", "h_nmr"], [_m(["hsqc"], 0.80)])
    _load(isolated, [
        {"id": "weak",   "root": "weak",   "type": "marina", "default": True},
        {"id": "strong", "root": "strong", "type": "marina", "default": False},
    ])
    sel = model_selection.select_model(["hsqc"])
    assert sel.model_id == "strong"
    assert sel.reason == "exact combo"
    assert sel.score == pytest.approx(0.80)
    assert sel.auto_selected is True


def test_model_ineligible_when_it_lacks_a_requested_modality(isolated):
    # nmr_only scores higher on HSQC, but the request adds MS which it can't accept,
    # so the MS-capable model is chosen even with a lower score.
    _model_dir(isolated, "nmr_only", ["hsqc", "c_nmr", "h_nmr"],
               [_m(["hsqc"], 0.9), _m(["hsqc", "c_nmr"], 0.9)])
    _model_dir(isolated, "with_ms", ["hsqc", "mass_spec"],
               [_m(["hsqc", "mass_spec"], 0.5)])
    _load(isolated, [
        {"id": "nmr_only", "root": "nmr_only", "type": "marina", "default": True},
        {"id": "with_ms",  "root": "with_ms",  "type": "marina", "default": False},
    ])
    sel = model_selection.select_model(["hsqc", "mass_spec"])
    assert sel.model_id == "with_ms"


def test_benchmark_both_is_preferred_over_simulated_test(isolated):
    # Same combo carries a benchmark/both and a test row; benchmark/both wins.
    _model_dir(isolated, "m", ["hsqc", "mw"], [
        _m(["hsqc", "mw"], 0.70, eval_set="benchmark", split="both"),
        _m(["hsqc", "mw"], 0.99, eval_set="test", split="test_all7pop"),
    ])
    _load(isolated, [{"id": "m", "root": "m", "type": "marina", "default": True}])
    sel = model_selection.select_model(["hsqc", "mw"])
    assert sel.score == pytest.approx(0.70)


def test_falls_back_to_largest_measured_subset(isolated):
    # No exact [hsqc,c_nmr,mw] row; the largest measured subset ([hsqc,c_nmr]) is used.
    _model_dir(isolated, "m", ["hsqc", "c_nmr", "mw"],
               [_m(["hsqc"], 0.5), _m(["hsqc", "c_nmr"], 0.7)])
    _load(isolated, [{"id": "m", "root": "m", "type": "marina", "default": True}])
    sel = model_selection.select_model(["hsqc", "c_nmr", "mw"])
    assert sel.reason == "subset fallback"
    assert sel.score == pytest.approx(0.7)


def test_falls_back_to_default_when_no_metrics(isolated):
    _model_dir(isolated, "a", ["hsqc"])            # eligible, but no metrics.json
    _model_dir(isolated, "b", ["hsqc"])
    _load(isolated, [
        {"id": "a", "root": "a", "type": "marina", "default": True},
        {"id": "b", "root": "b", "type": "marina", "default": False},
    ])
    sel = model_selection.select_model(["hsqc"])
    assert sel.model_id in {"a", "b"}
    assert sel.reason == "eligible, no metrics"
