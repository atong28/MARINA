"""
End-to-end hosting of a SPECTRE model.

Runs against a synthetic, randomly-initialised checkpoint: this proves the
architecture, collation, retrieval and card-building paths line up. It cannot
say anything about output *quality* — that needs a trained checkpoint.
"""
import pytest
import torch

from app.predictor import predict_from_raw, retrieve_top_k, run_model, preprocess_inputs
from app.result_builder import build_result_cards

pytestmark = pytest.mark.slow

RAW = {
    "hsqc": [5.83, 78.42, 4007870.8, 2.37, 43.03, -1524417.4, 1.09, 18.75, 6450184.0],
    "h_nmr": [5.83, 5.29, 2.37, 1.09],
    "c_nmr": [180.01, 78.42, 43.03, 18.75],
    "mass_spec": [104.0712, 12000.0, 227.1754, 96000.0],
    "mw": 609.0,
}


@pytest.fixture
def registered_spectre(spectre_model_dir, monkeypatch):
    """Register the synthetic model so predict_from_raw can resolve it by id."""
    import app.registry as registry
    from app.session import ModelSession

    session = ModelSession.from_model_root(str(spectre_model_dir), model_type="spectre")
    monkeypatch.setitem(registry._registry, "spectre_synth", session)
    return session


def test_forward_pass_returns_one_logit_per_fingerprint_bit(registered_spectre):
    out = run_model(registered_spectre, preprocess_inputs(RAW))
    assert out.shape == (registered_spectre.fp_loader.out_dim,)
    assert torch.isfinite(out).all()


def test_retrieval_returns_ranked_hits(registered_spectre):
    logits = run_model(registered_spectre, preprocess_inputs(RAW))
    scores, idxs, fp = retrieve_top_k(registered_spectre, logits, k=3)

    assert len(scores) == len(idxs) == 3
    assert scores == sorted(scores, reverse=True)
    assert len(fp) == registered_spectre.fp_loader.out_dim


def test_scores_are_genuine_cosine_similarities(registered_spectre):
    """
    Guards the normalisation contract: RankingSet normalises only the query, so
    an unnormalised rankingset silently pushes every score past 1.0 where the
    card builder clamps it and all results read 1.000.
    """
    scores, _, _ = predict_from_raw(RAW, k=5, model_id="spectre_synth")
    assert all(-1.0001 <= s <= 1.0001 for s in scores), scores
    assert len(set(round(s, 4) for s in scores)) > 1, "all scores identical — clamped?"


def test_predicted_fingerprint_is_a_probability_vector(registered_spectre):
    _, _, fp = predict_from_raw(RAW, k=2, model_id="spectre_synth")
    assert all(0.0 <= v <= 1.0 for v in fp)


@pytest.mark.parametrize("name,subset", [
    ("hsqc only", {"hsqc": RAW["hsqc"]}),
    ("h_nmr only", {"h_nmr": RAW["h_nmr"]}),
    ("c_nmr only", {"c_nmr": RAW["c_nmr"]}),
    ("ms only", {"mass_spec": RAW["mass_spec"]}),
    ("mw only", {"mw": RAW["mw"]}),
    ("nmr without mw", {k: RAW[k] for k in ("hsqc", "h_nmr", "c_nmr")}),
    ("everything", RAW),
])
def test_every_modality_subset_predicts(registered_spectre, name, subset):
    """The spreadsheet lets users fill any combination of columns."""
    scores, idxs, _ = predict_from_raw(subset, k=2, model_id="spectre_synth")
    assert len(scores) == len(idxs) == 2


def test_mw_filter_narrows_the_result_set(registered_spectre):
    """
    Only ethanol (46.04 Da) of the five fixture structures is under 50 Da, so a
    working filter returns exactly one candidate for k=5. The old assertion was
    len(narrow) <= len(wide), which also passed while the filter was a no-op.
    """
    wide, _, _ = predict_from_raw(RAW, k=5, model_id="spectre_synth")
    narrow, idxs, _ = predict_from_raw(
        RAW, k=5, model_id="spectre_synth", mw_min=1.0, mw_max=50.0)

    assert len(wide) == 5
    assert len(narrow) == 1
    assert registered_spectre.get_smiles(idxs[0]) == "CCO"


def test_mw_filter_matching_nothing_returns_no_results(registered_spectre):
    """
    Only reachable now that the filter works: an over-narrow range leaves an
    empty candidate set, which must come back empty rather than crash.
    """
    scores, idxs, _ = predict_from_raw(
        RAW, k=5, model_id="spectre_synth", mw_min=10_000.0, mw_max=10_001.0)
    assert len(scores) == 0
    assert len(idxs) == 0


def test_result_cards_build_for_a_spectre_prediction(registered_spectre):
    """The renderer and card builder are shared with MARINA; prove they run."""
    scores, idxs, fp = predict_from_raw(RAW, k=2, model_id="spectre_synth")
    cards = build_result_cards(
        registered_spectre, list(zip(idxs, scores)),
        torch.tensor(fp, dtype=torch.float32), img_size=200, max_cards=2)

    assert len(cards) == 2
    for card in cards:
        assert card["smiles"]
        assert 0.0 <= card["similarity"] <= 1.0
        assert card["svg"], "no depiction rendered"
        assert card["exact_mass"] is not None
