"""
ModelSession.from_model_root guards.

Each of these failure modes is silent without a check, and all three are live
risks when importing SPECTRE artefacts produced by a different codebase.
"""
import json
import shutil

import pytest
import torch

from app.session import ModelSession
from tests.conftest import build_model_dir

pytestmark = pytest.mark.slow


@pytest.fixture
def model_copy(spectre_model_dir, tmp_path):
    dst = tmp_path / "copy"
    shutil.copytree(spectre_model_dir, dst)
    return dst


def _params(root):
    return json.loads((root / "params.json").read_text())


def _write_params(root, params):
    (root / "params.json").write_text(json.dumps(params))


def test_a_valid_spectre_directory_loads(spectre_model_dir):
    sess = ModelSession.from_model_root(str(spectre_model_dir), model_type="spectre")
    assert sess.model_type == "spectre"
    assert sess.fp_loader.out_dim > 0
    assert sess.fp_type == "RankingEntropy"


def test_unknown_params_keys_are_rejected(model_copy):
    """
    Pydantic dataclasses DISCARD unrecognised keys, so a params.json written
    against the published SPECTRE code would silently build the model from this
    codebase's defaults instead of the settings it was trained with.
    """
    p = _params(model_copy)
    p["encoder_hidden_layers"] = 4
    _write_params(model_copy, p)

    with pytest.raises(RuntimeError, match="does not define"):
        ModelSession.from_model_root(str(model_copy), model_type="spectre")


def test_the_rejection_names_the_offending_keys(model_copy):
    p = _params(model_copy)
    p["totally_bogus"] = 1
    _write_params(model_copy, p)

    with pytest.raises(RuntimeError, match="totally_bogus"):
        ModelSession.from_model_root(str(model_copy), model_type="spectre")


def test_out_dim_mismatch_is_rejected(model_copy):
    """The output layer and the retrieval matmul must agree on width."""
    p = _params(model_copy)
    p["out_dim"] = p["out_dim"] + 1000
    _write_params(model_copy, p)

    with pytest.raises(RuntimeError, match="[Ff]ingerprint size mismatch"):
        ModelSession.from_model_root(str(model_copy), model_type="spectre")


def test_checkpoint_missing_parameters_is_rejected(model_copy):
    """
    Regression: load_state_dict(strict=False) used to discard its result, so a
    mismatched checkpoint served randomly-initialised weights.
    """
    ckpt = torch.load(model_copy / "best.ckpt", weights_only=True)
    state = ckpt["state_dict"]
    for key in list(state)[:3]:
        del state[key]
    torch.save({"state_dict": state}, model_copy / "best.ckpt")

    with pytest.raises(RuntimeError, match="missing"):
        ModelSession.from_model_root(str(model_copy), model_type="spectre")


def test_unexpected_checkpoint_keys_are_tolerated(model_copy, caplog):
    """Lightning checkpoints carry buffers the inference model does not declare."""
    ckpt = torch.load(model_copy / "best.ckpt", weights_only=True)
    ckpt["state_dict"]["some.extra.buffer"] = torch.zeros(3)
    torch.save(ckpt, model_copy / "best.ckpt")

    sess = ModelSession.from_model_root(str(model_copy), model_type="spectre")
    assert sess.model_type == "spectre"


def test_malformed_params_json_reports_the_file(model_copy):
    p = _params(model_copy)
    p["heads"] = "not-an-integer"
    _write_params(model_copy, p)

    with pytest.raises(RuntimeError, match="params.json"):
        ModelSession.from_model_root(str(model_copy), model_type="spectre")


def test_marina_directory_also_loads(marina_src, tmp_path):
    """The SPECTRE work must not have regressed the default model type."""
    root = build_model_dir(tmp_path / "marina_synth", "marina")
    sess = ModelSession.from_model_root(str(root), model_type="marina")
    assert sess.model_type == "marina"
