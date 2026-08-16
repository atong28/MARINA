"""
app.bit_explain against the substructure vocabulary.

A substructure model keys its feature map on canonical fragment SMILES rather than
Morgan BitInfo 4-tuples. Everything downstream of the loader has to cope with a feature
that is a plain string and carries no radius or centre-atom symbol of its own.
"""
import pytest

from app.bit_explain import GROUP_MATCH, GROUP_MISSING, explain_bits

pytestmark = pytest.mark.usefixtures("marina_src")

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
OTHER = "CCCCCCOc1ccccc1C(=O)NC"
CHIRAL_ALA = "C[C@H](N)C(=O)O"
RADIUS = 6


@pytest.fixture
def sub_session(marina_src):
    """Session stub whose feature map is keyed on fragment SMILES."""
    from src.modules.data.fp_utils import get_substructure_smiles, SUBSTRUCTURE

    cols: dict = {}
    for smi in (ASPIRIN, OTHER, CHIRAL_ALA):
        for frag in sorted(get_substructure_smiles(smi, RADIUS)):
            cols.setdefault(frag, len(cols))

    class _Loader:
        FEATURE_KIND = SUBSTRUCTURE
        max_radius = RADIUS
        bitinfo_to_fp_index_map = cols
        fp_index_to_bitinfo_map = {v: k for k, v in cols.items()}
        out_dim = len(cols)

    class _Session:
        fp_loader = _Loader()

    return _Session()


def _cols_of(sub_session, smiles):
    from src.modules.data.fp_utils import get_substructure_smiles
    m = sub_session.fp_loader.bitinfo_to_fp_index_map
    return {m[f] for f in get_substructure_smiles(smiles, RADIUS) if f in m}


def test_feature_map_keys_are_strings(sub_session):
    keys = list(sub_session.fp_loader.bitinfo_to_fp_index_map)
    assert keys and all(isinstance(k, str) for k in keys)


def test_present_bits_carry_atom_locations(sub_session):
    n = sub_session.fp_loader.out_dim
    fp = [0.0] * n
    for c in _cols_of(sub_session, ASPIRIN):
        fp[c] = 0.99

    out = explain_bits(sub_session, ASPIRIN, fp, limit=500)
    located = [b for b in out["bits"] if b["present"]]
    assert located, "aspirin's own substructures should be found in aspirin"
    for b in located:
        assert b["atoms"], f"bit {b['index']} present but has no atoms"


def test_row_reports_fragment_and_radius(sub_session):
    """The feature IS the fragment SMILES; radius comes from where it was found."""
    n = sub_session.fp_loader.out_dim
    fp = [0.0] * n
    for c in _cols_of(sub_session, ASPIRIN):
        fp[c] = 0.99

    out = explain_bits(sub_session, ASPIRIN, fp, limit=500)
    present = [b for b in out["bits"] if b["present"]]
    for b in present:
        assert b["fragment_smiles"], "substructure row must name its fragment"
        assert b["atom_symbol"] == "", "substructure features have no centre symbol"
        assert b["radius"] >= 0, "a located feature must report the radius it was found at"


def test_absent_bits_have_no_location(sub_session):
    n = sub_session.fp_loader.out_dim
    only_other = _cols_of(sub_session, OTHER) - _cols_of(sub_session, ASPIRIN)
    assert only_other, "fixture molecules must differ"

    fp = [0.0] * n
    for c in only_other:
        fp[c] = 0.99

    out = explain_bits(sub_session, ASPIRIN, fp, limit=500)
    for b in out["bits"]:
        if not b["present"]:
            assert b["atoms"] == [] and b["bonds"] == []


def test_groups_split_on_presence(sub_session):
    """A confident bit aspirin has is a match; one it lacks is a miss."""
    n = sub_session.fp_loader.out_dim
    mine = _cols_of(sub_session, ASPIRIN)
    theirs = _cols_of(sub_session, OTHER) - mine
    assert mine and theirs

    fp = [0.0] * n
    for c in mine | theirs:
        fp[c] = 0.99

    out = explain_bits(sub_session, ASPIRIN, fp, limit=1000)
    by_col = {b["index"]: b for b in out["bits"]}
    assert all(by_col[c]["group"] == GROUP_MATCH for c in mine if c in by_col)
    assert all(by_col[c]["group"] == GROUP_MISSING for c in theirs if c in by_col)


def test_stereo_input_still_matches_the_vocabulary(sub_session):
    """
    The vocabulary is built from stereo-stripped SMILES. A candidate drawn with
    stereochemistry must still light up its bits, or the panel silently shows nothing.
    """
    n = sub_session.fp_loader.out_dim
    fp = [0.99] * n
    out = explain_bits(sub_session, CHIRAL_ALA, fp, limit=1000)
    assert any(b["present"] for b in out["bits"]), \
        "no substructure matched a stereo-bearing candidate"


def test_atom_indices_are_valid_for_the_drawn_molecule(sub_session):
    """Locations must index the molecule as given, not a canonicalised renumbering."""
    from rdkit import Chem
    n_atoms = Chem.MolFromSmiles(CHIRAL_ALA).GetNumAtoms()
    fp = [0.99] * sub_session.fp_loader.out_dim
    out = explain_bits(sub_session, CHIRAL_ALA, fp, limit=1000)
    for b in out["bits"]:
        for a in b["atoms"]:
            assert 0 <= a < n_atoms, f"atom index {a} out of range for {CHIRAL_ALA}"
