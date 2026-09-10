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


def _chain_fixture(marina_src):
    """
    Counting vocabulary over hexanoic acid (candidate) and butanoic acid (the
    prediction), which has fewer chain CH2 occurrences: the candidate's extra
    chain is surplus the prediction does not want, the carboxyl end is wanted by both.
    """
    from src.modules.data.fp_utils import (
        count_substructure_multiplicities, MULTIPLICITY_UNCAPPED,
    )
    long, short = "CCCCCC(=O)O", "CCCC(=O)O"

    cols: dict = {}
    for smi in (long, short):
        for feat in sorted(count_substructure_multiplicities(smi, RADIUS)):
            cols.setdefault(feat, len(cols))

    class _Loader:
        FEATURE_KIND = MULTIPLICITY_UNCAPPED
        max_radius = RADIUS
        bitinfo_to_fp_index_map = cols
        out_dim = len(cols)

    pred = torch.zeros(len(cols))
    for feat in count_substructure_multiplicities(short, RADIUS):
        pred[cols[feat]] = 1.0

    mol = _mol(long)
    carboxyl = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C" and
                any(n.GetSymbol() == "O" for n in a.GetNeighbors())][0]
    return long, mol, pred, _Loader(), carboxyl


def test_attribution_paints_surplus_chain_pink_and_shared_end_green(marina_src):
    """
    Direct attribution: the terminal methyl sits only in chain fragments the
    prediction lacks (long chains, and the ≥5×/≥6× carbon buckets), so it must
    be negative; the carboxyl carbon's fragments are predicted, so positive.
    """
    from app.renderer import _atom_weights_attribution

    long, mol, pred, loader, carboxyl = _chain_fixture(marina_src)
    weights = _atom_weights_attribution(mol, long, pred, loader)
    assert len(weights) == mol.GetNumAtoms()
    assert weights[0] < 0
    assert weights[carboxyl] > 0


def test_attribution_is_independent_of_atom_order(marina_src):
    """
    The ablation credited each occurrence to one centre atom, so the same
    molecule written two ways coloured differently. Attribution scores every
    copy of a fragment alike and spreads it over all its atoms, so reversing the
    SMILES must give the same weights on the same atoms.
    """
    from app.renderer import _atom_weights_attribution

    long, mol, pred, loader, _ = _chain_fixture(marina_src)
    fwd = _atom_weights_attribution(mol, long, pred, loader)
    rev_smiles = "OC(=O)CCCCC"
    rev = _atom_weights_attribution(_mol(rev_smiles), rev_smiles, pred, loader)
    # The two SMILES number the atoms differently; match them by canonical rank.
    from rdkit import Chem
    fwd_mol, rev_mol = _mol(long), _mol(rev_smiles)
    fwd_by_rank = dict(zip(Chem.CanonicalRankAtoms(fwd_mol, breakTies=True), fwd))
    rev_by_rank = dict(zip(Chem.CanonicalRankAtoms(rev_mol, breakTies=True), rev))
    assert fwd_by_rank.keys() == rev_by_rank.keys()
    for rank, w in rev_by_rank.items():
        assert abs(fwd_by_rank[rank] - w) < 1e-9, f"rank {rank}: {fwd_by_rank[rank]} vs {w}"


def test_attribution_uses_the_calibrator(marina_src):
    from app.renderer import _atom_weights_attribution

    long, mol, pred, loader, carboxyl = _chain_fixture(marina_src)
    raw = _atom_weights_attribution(mol, long, pred, loader)
    # A calibrator that halves every probability pulls everything toward "not predicted".
    halved = _atom_weights_attribution(mol, long, pred, loader, calibrator=lambda p: p * 0.5)
    assert all(h <= r + 1e-9 for h, r in zip(halved, raw))
    assert halved[carboxyl] < raw[carboxyl]


def test_multiplicity_ablation_drops_only_the_top_bucket(marina_src):
    """
    The retained ablation method on a counting vocabulary: removing one
    occurrence of a fragment lowers its count by one, so only the top cumulative
    bucket goes. Zeroing every bucket (the Morgan rule) made every atom look
    load-bearing and the map green-only.
    """
    from src.modules.data.fp_utils import MULTIPLICITY_UNCAPPED
    from app.renderer import _atom_weights_ablation, _multiplicity_drops

    long, mol, pred, loader, carboxyl = _chain_fixture(marina_src)
    weights = _atom_weights_ablation(mol, long, pred, loader)
    assert len(weights) == mol.GetNumAtoms()
    assert min(weights[:5]) < 0, "a surplus chain carbon should contradict the shorter prediction"
    assert weights[carboxyl] > 0

    # Direct check of the rule: an atom drops one bucket per occurrence it centres.
    from src.modules.data.fp_utils import get_feature_locations
    located = get_feature_locations(long, RADIUS, kind=MULTIPLICITY_UNCAPPED)
    atom_to_bits = {a: [f for f, _ in feats] for a, feats in located.items()}
    drops = _multiplicity_drops(atom_to_bits, MULTIPLICITY_UNCAPPED)
    for atom, removed in drops.items():
        per_frag: dict = {}
        for frag, k in removed:
            per_frag.setdefault(frag, []).append(k)
        centred = [f for f, k in atom_to_bits[atom] if k == 1]
        for frag, ks in per_frag.items():
            assert len(ks) == centred.count(frag)
            top = max(k for f, k in atom_to_bits[atom] if f == frag)
            assert sorted(ks) == list(range(top - len(ks) + 1, top + 1))


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
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    pixels = [(r, g, b) for r, g, b, a in _pixels(img) if a > 0]

    pink = sum(1 for r, g, b in pixels if r > g + 20 and b > g + 10)
    green = sum(1 for r, g, b in pixels if g > r + 20 and g > b + 20)
    assert pink > 0, "negative-weight highlights are missing"
    assert green > 0, "positive-weight highlights are missing"


def test_enhanced_render_payload_stays_small(stub_loader, predicted_fp):
    """
    The contour fill is one rect per grid cell in SVG (~1 MB/card); rasterising
    keeps it in the low hundreds of KB. Guard against a silent revert to vector output.
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


# ── Depiction geometry ────────────────────────────────────────────────────────

CHIRAL = "C[C@H](O)/C=C/c1ccccc1"


def test_geometry_covers_every_atom_and_bond():
    from app.renderer import depiction_geometry

    mol = _mol(RETRIEVED)
    geo = depiction_geometry(RETRIEVED, 400)
    assert geo["size"] == 400
    assert len(geo["atoms"]) == mol.GetNumAtoms()
    assert len(geo["bonds"]) == mol.GetNumBonds()
    for x, y in geo["atoms"]:
        assert 0 <= x <= 400 and 0 <= y <= 400


def test_geometry_matches_the_plain_depiction():
    """The overlay is drawn from these coordinates over the plain SVG, so the
    atom positions must be the ones that drawing actually used."""
    import re
    from app.renderer import depiction_geometry, render_plain_svg

    svg = render_plain_svg(RETRIEVED, 400)
    geo = depiction_geometry(RETRIEVED, 400)
    # Every bond line in the SVG starts or ends at a drawn atom position; check a
    # sample of atom coordinates appear (to the pixel) among the path endpoints.
    coords = {(round(float(x)), round(float(y)))
              for x, y in re.findall(r"([\d.]+),([\d.]+)", svg)}
    hits = sum((round(x), round(y)) in coords for x, y in geo["atoms"])
    assert hits >= len(geo["atoms"]) // 2


def test_geometry_returns_none_for_bad_smiles():
    from app.renderer import depiction_geometry

    assert depiction_geometry("not-a-smiles", 400) is None


def test_depictions_drop_stereochemistry():
    """MARINA is 2-D: no wedges or E/Z geometry may be drawn."""
    from app.renderer import render_plain_svg

    svg = render_plain_svg(CHIRAL, 400)
    assert "wedge" not in svg.lower()
    assert "<polygon" not in svg          # RDKit draws wedges as filled polygons


def test_plain_and_map_share_one_scale():
    """The whole point of the shared fit: the raster (map) drawer must place
    atoms exactly where the vector (plain) drawer does."""
    from rdkit.Chem.Draw import rdMolDraw2D
    from app.renderer import _fit, _prepare_mol

    mol = _prepare_mol(RETRIEVED)
    a = rdMolDraw2D.MolDraw2DSVG(400, 400)
    _fit(a, mol, 400)
    a.DrawMolecule(mol); a.FinishDrawing()
    b = rdMolDraw2D.MolDraw2DCairo(400, 400)
    _fit(b, mol, 400)
    b.DrawMolecule(mol); b.FinishDrawing()
    for i in range(mol.GetNumAtoms()):
        pa, pb = a.GetDrawCoords(i), b.GetDrawCoords(i)
        assert abs(pa.x - pb.x) < 0.05 and abs(pa.y - pb.y) < 0.05


def test_plain_drawing_has_no_background():
    """It is the top layer over the map wash, so it must not paint one."""
    from app.renderer import render_plain_svg

    svg = render_plain_svg(RETRIEVED, 400)
    assert "fill:#FFFFFF" not in svg and "fill:#ffffff" not in svg


def test_map_is_a_transparent_wash_without_the_molecule(stub_loader, predicted_fp):
    """The map is an underlay: transparent where nothing is weighted, and no
    black skeleton of its own (the client draws the plain SVG over it)."""
    from PIL import Image

    out = render_enhanced_svg(RETRIEVED, predicted_fp, stub_loader, img_size=300)
    img = Image.open(io.BytesIO(base64.b64decode(out.split(",", 1)[1])))
    assert img.mode == "RGBA"
    assert img.getpixel((1, 1))[3] == 0
    assert not any(r < 40 and g < 40 and b < 40 and a > 200 for r, g, b, a in _pixels(img))


def _pixels(img):
    return list(img.get_flattened_data()) if hasattr(img, "get_flattened_data") else list(img.getdata())


def test_map_is_none_when_the_molecule_cannot_be_weighted(predicted_fp):
    class _Empty:
        max_radius = RADIUS
        bitinfo_to_fp_index_map = {}
    assert render_enhanced_svg(RETRIEVED, predicted_fp, _Empty(), img_size=300) is None


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
