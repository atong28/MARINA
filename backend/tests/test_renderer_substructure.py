"""
app.renderer's leave-one-out atom ablation against the substructure vocabulary.

The sign of the weights is the whole point of the ablation, so the substructure path
has to reproduce it: a non-negative vector renders green-only and tells the user nothing.
"""
import pytest
import torch

from app.renderer import _atom_weights, render_enhanced_svg

pytestmark = pytest.mark.usefixtures("marina_src")

RETRIEVED = "CC(=O)Oc1ccccc1C(=O)O"          # aspirin
PREDICTED = "CCCCCCOc1ccccc1C(=O)NC"          # shares the aryl ether, differs elsewhere
RADIUS = 6


@pytest.fixture
def sub_loader(marina_src):
    """Feature map keyed on fragment SMILES, spanning both molecules."""
    from src.modules.data.fp_utils import get_substructure_smiles, SUBSTRUCTURE

    cols: dict = {}
    for smi in (RETRIEVED, PREDICTED):
        for frag in sorted(get_substructure_smiles(smi, RADIUS)):
            cols.setdefault(frag, len(cols))

    class _Loader:
        FEATURE_KIND = SUBSTRUCTURE
        max_radius = RADIUS
        bitinfo_to_fp_index_map = cols
        out_dim = len(cols)

    return _Loader()


@pytest.fixture
def sub_predicted_fp(sub_loader, marina_src):
    from src.modules.data.fp_utils import get_substructure_smiles
    fp = torch.zeros(sub_loader.out_dim)
    for frag in get_substructure_smiles(PREDICTED, RADIUS):
        col = sub_loader.bitinfo_to_fp_index_map.get(frag)
        if col is not None:
            fp[col] = 1.0
    return fp


def _mol(smiles):
    from rdkit import Chem
    return Chem.MolFromSmiles(smiles)


def test_weights_are_produced_per_atom(sub_loader, sub_predicted_fp):
    mol = _mol(RETRIEVED)
    weights = _atom_weights(mol, RETRIEVED, sub_predicted_fp, sub_loader)
    assert weights is not None
    assert len(weights) == mol.GetNumAtoms()


def test_weights_are_signed(sub_loader, sub_predicted_fp):
    """Atoms that pull the match apart must come out negative, or the view is useless."""
    weights = _atom_weights(_mol(RETRIEVED), RETRIEVED, sub_predicted_fp, sub_loader)
    assert any(w > 0 for w in weights), "no atom supports the match"
    assert any(w < 0 for w in weights), "no atom opposes the match"


def test_identical_molecule_gives_no_negative_weights(sub_loader):
    """Ablating any atom of a perfectly matching molecule can only hurt."""
    from src.modules.data.fp_utils import get_substructure_smiles
    fp = torch.zeros(sub_loader.out_dim)
    for frag in get_substructure_smiles(RETRIEVED, RADIUS):
        col = sub_loader.bitinfo_to_fp_index_map.get(frag)
        if col is not None:
            fp[col] = 1.0

    weights = _atom_weights(_mol(RETRIEVED), RETRIEVED, fp, sub_loader)
    assert all(w >= -1e-9 for w in weights)


def test_weights_are_none_without_a_feature_map(sub_predicted_fp):
    from src.modules.data.fp_utils import SUBSTRUCTURE

    class _Empty:
        FEATURE_KIND = SUBSTRUCTURE
        max_radius = RADIUS
        bitinfo_to_fp_index_map = {}
        out_dim = 0

    assert _atom_weights(_mol(RETRIEVED), RETRIEVED, sub_predicted_fp, _Empty()) is None


def test_enhanced_render_returns_a_png_data_uri(sub_loader, sub_predicted_fp):
    out = render_enhanced_svg(RETRIEVED, sub_predicted_fp, sub_loader)
    assert out and out.startswith("data:image")
