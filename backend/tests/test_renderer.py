"""
app.renderer — fingerprint-weighted molecule depictions.

The headline regression here is polarity: the original weight function was
floored at zero, so RDKit's symmetric PiWG colour map could only ever reach its
green half and the pink (contradicting-substructure) highlights never appeared.
"""
import base64
import io

import pytest
import torch

from app.renderer import (
    _atom_weights, _cos_sim, _mfp_from_bitinfo, render_enhanced_svg, render_plain_svg,
)

pytestmark = pytest.mark.usefixtures("marina_src")

RETRIEVED = "CC(=O)Oc1ccccc1C(=O)O"          # aspirin
PREDICTED = "CCCCCCOc1ccccc1C(=O)NC"          # shares the aryl ether, differs elsewhere
RADIUS = 6


@pytest.fixture
def stub_loader(marina_src):
    """A fingerprint loader spanning both molecules' bit environments."""
    from src.modules.data.fp_utils import get_bitinfos

    cols: dict = {}
    for smi in (RETRIEVED, PREDICTED):
        for bit in sorted(get_bitinfos(smi, RADIUS)[1]):
            cols.setdefault(bit, len(cols))

    class _Loader:
        max_radius = RADIUS
        bitinfo_to_fp_index_map = cols
        out_dim = len(cols)

    return _Loader()


@pytest.fixture
def predicted_fp(stub_loader, marina_src):
    from src.modules.data.fp_utils import get_bitinfos
    fp = torch.zeros(stub_loader.out_dim)
    for bit in get_bitinfos(PREDICTED, RADIUS)[1]:
        fp[stub_loader.bitinfo_to_fp_index_map[bit]] = 1.0
    return fp


def _mol(smiles):
    from rdkit import Chem
    return Chem.MolFromSmiles(smiles)


# ── Weights ───────────────────────────────────────────────────────────────────

def test_weights_are_produced_per_atom(stub_loader, predicted_fp):
    mol = _mol(RETRIEVED)
    weights = _atom_weights(mol, RETRIEVED, predicted_fp, stub_loader)
    assert len(weights) == mol.GetNumAtoms()


def test_weights_are_signed(stub_loader, predicted_fp):
    """Without negative weights the pink half of the colour map is unreachable."""
    weights = _atom_weights(_mol(RETRIEVED), RETRIEVED, predicted_fp, stub_loader)
    assert any(w < 0 for w in weights), "no atom contradicts the prediction"
    assert any(w > 0 for w in weights), "no atom supports the prediction"


def test_identical_molecule_gives_no_negative_weights(stub_loader, marina_src):
    """Every atom of a perfect match supports the match."""
    from src.modules.data.fp_utils import get_bitinfos
    fp = torch.zeros(stub_loader.out_dim)
    for bit in get_bitinfos(RETRIEVED, RADIUS)[1]:
        fp[stub_loader.bitinfo_to_fp_index_map[bit]] = 1.0

    weights = _atom_weights(_mol(RETRIEVED), RETRIEVED, fp, stub_loader)
    assert all(w >= -1e-9 for w in weights)


def test_weights_are_none_without_a_feature_map(predicted_fp):
    class _Empty:
        max_radius = RADIUS
        bitinfo_to_fp_index_map = {}
    assert _atom_weights(_mol(RETRIEVED), RETRIEVED, predicted_fp, _Empty()) is None


def test_ablation_zeroes_every_bit_the_atom_touches(stub_loader, marina_src):
    """
    SPECTRE's semantics: an ignored atom's bits drop out even when another atom
    also contributes them. That is what makes the deltas large enough to swing
    negative.
    """
    from src.modules.data.fp_utils import get_bitinfos
    atom_to_bits, _ = get_bitinfos(RETRIEVED, RADIUS)
    cols = stub_loader.bitinfo_to_fp_index_map
    dim = stub_loader.out_dim

    full = _mfp_from_bitinfo(atom_to_bits, cols, dim)
    ablated = _mfp_from_bitinfo(atom_to_bits, cols, dim, (0,))
    assert ablated.sum() < full.sum()
    for bit in atom_to_bits.get(0, ()):
        assert ablated[cols[bit]] == 0.0


def test_cos_sim_handles_zero_vectors():
    assert _cos_sim(torch.zeros(4), torch.ones(4)) == 0.0


# ── Rendering ─────────────────────────────────────────────────────────────────

def test_enhanced_render_returns_a_png_data_uri(stub_loader, predicted_fp):
    out = render_enhanced_svg(RETRIEVED, predicted_fp, stub_loader, img_size=300)
    assert out.startswith("data:image/png;base64,")
    assert base64.b64decode(out.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"


def test_enhanced_render_shows_both_polarities(stub_loader, predicted_fp):
    """The regression this whole path exists for: pink must actually appear."""
    from PIL import Image

    out = render_enhanced_svg(RETRIEVED, predicted_fp, stub_loader, img_size=300)
    raw = base64.b64decode(out.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    pixels = list(img.getdata()) if not hasattr(img, "get_flattened_data") else list(img.get_flattened_data())

    pink = sum(1 for r, g, b in pixels if r > g + 20 and b > g + 10)
    green = sum(1 for r, g, b in pixels if g > r + 20 and g > b + 20)
    assert pink > 0, "negative-weight highlights are missing"
    assert green > 0, "positive-weight highlights are missing"


def test_enhanced_render_payload_stays_small(stub_loader, predicted_fp):
    """
    The contour fill is one rect per grid cell in SVG (~1 MB/card); rasterising
    keeps it around 100 KB. Guard against a silent revert to vector output.
    """
    out = render_enhanced_svg(RETRIEVED, predicted_fp, stub_loader, img_size=400)
    assert len(out) < 400_000


def test_enhanced_render_falls_back_for_invalid_smiles(stub_loader, predicted_fp):
    assert render_enhanced_svg("not-a-smiles", predicted_fp, stub_loader) is None


def test_plain_render_stays_vector():
    out = render_plain_svg(RETRIEVED, 400)
    assert out.lstrip().startswith("<") and "svg" in out[:200].lower()
    assert len(out) < 100_000


def test_plain_render_returns_none_for_invalid_smiles():
    assert render_plain_svg("not-a-smiles") is None
