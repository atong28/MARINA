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
    _atom_weights, _cos_sim, _mfp_from_bitinfo, highlighting_available,
    render_enhanced_svg, render_plain_svg,
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


# ── Single-bit highlighting ───────────────────────────────────────────────────

def test_render_bit_svg_marks_the_requested_atoms():
    """The highlight view stays vector — no contour fill, so no rasterisation."""
    from app.renderer import render_bit_svg

    svg = render_bit_svg(RETRIEVED, atoms=[0, 1, 2], bonds=[0, 1], img_size=300)
    assert svg is not None
    assert svg.lstrip().startswith("<?xml") or "<svg" in svg
    assert not svg.startswith("data:image")


def test_render_bit_svg_tolerates_out_of_range_indices():
    """The client can hold a bit list from the previous molecule mid-swap."""
    from app.renderer import render_bit_svg

    assert render_bit_svg(RETRIEVED, atoms=[9999], bonds=[9999], img_size=300) is not None


def test_render_bit_svg_with_no_highlight_still_renders():
    from app.renderer import render_bit_svg

    assert render_bit_svg(RETRIEVED, atoms=[], bonds=[], img_size=300) is not None


def test_render_bit_svg_returns_none_for_bad_smiles():
    from app.renderer import render_bit_svg

    assert render_bit_svg("not-a-smiles", atoms=[0], bonds=[], img_size=300) is None


def test_render_bit_svg_differs_when_highlighting():
    """Guards against silently returning the plain depiction."""
    from app.renderer import render_bit_svg

    plain = render_bit_svg(RETRIEVED, atoms=[], bonds=[], img_size=300)
    marked = render_bit_svg(RETRIEVED, atoms=[0, 1, 2], bonds=[0, 1], img_size=300)
    assert plain != marked


# ── Fragment thumbnails ───────────────────────────────────────────────────────

@pytest.mark.parametrize("frag", [
    "CCCCC",              # plain chain
    "C=O",                # double bond
    "ccc",                # aromatic atoms clipped out of their ring
    "cc(C)oc(c)c",        # aromatic heterocycle fragment
    "CC(O)C(O)C(C)O",     # several dangling valences
])
def test_render_fragment_svg_draws_partial_structures(frag):
    """
    PathToSubmol fragments are partial: lowercase atoms are non-ring and fail a
    normal sanitize, so a plain MolFromSmiles path would return None for half
    the bit pool.
    """
    from app.renderer import render_fragment_svg

    svg = render_fragment_svg(frag)
    assert svg is not None and "<svg" in svg


def test_render_fragment_svg_falls_back_to_the_centre_atom():
    """Radius-0 bits carry no fragment SMILES at all — 52 of the 16,384."""
    from app.renderer import render_fragment_svg

    assert render_fragment_svg("", "O") is not None


def test_render_fragment_svg_does_not_invent_hydrogens():
    """
    Open valences are bonds to the rest of the molecule, not hydrogens. Left to
    RDKit a lone oxygen draws as "H2O" and an ether oxygen as "OH", asserting
    atoms the parent structure may not have.
    """
    from app.renderer import render_fragment_svg

    assert "H" not in _svg_text(render_fragment_svg("", "O"))
    assert "H" not in _svg_text(render_fragment_svg("CC(O)C(O)C(C)O"))


def _svg_text(svg: str) -> str:
    """Concatenate the glyph labels an SVG draws, ignoring markup."""
    import re
    return "".join(re.findall(r">([^<>]*)</text>", svg or ""))


def test_render_fragment_svg_returns_none_when_nothing_to_draw():
    from app.renderer import render_fragment_svg

    assert render_fragment_svg("", "") is None

# ── HIGHLIGHT_ENABLED switch ──────────────────────────────────────────────────

def test_highlighting_off_returns_none_rather_than_a_plain_copy(
    stub_loader, predicted_fp, monkeypatch,
):
    """
    Cards pair this with plain_svg, so returning a copy would double the payload
    and leave the client unable to tell the two depictions apart.
    """
    import app.config
    monkeypatch.setattr(app.config, "HIGHLIGHT_ENABLED", False)
    assert render_enhanced_svg(RETRIEVED, predicted_fp, stub_loader, img_size=300) is None


def test_highlighting_off_leaves_plain_rendering_alone(monkeypatch):
    import app.config
    monkeypatch.setattr(app.config, "HIGHLIGHT_ENABLED", False)
    assert render_plain_svg(RETRIEVED, 400) is not None


def test_highlighting_available_tracks_the_switch(monkeypatch):
    import app.config
    monkeypatch.setattr(app.config, "HIGHLIGHT_ENABLED", True)
    assert highlighting_available() is True
    monkeypatch.setattr(app.config, "HIGHLIGHT_ENABLED", False)
    assert highlighting_available() is False
