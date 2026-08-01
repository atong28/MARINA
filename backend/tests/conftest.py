"""
Shared fixtures.

Environment is configured before `app.*` is imported, since app.config reads
os.environ at module load. Tests that need the MARINA source tree (torch, rdkit,
the model classes) are skipped rather than failed when it is not reachable, so
the fast tests still run in a bare checkout.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
MARINA_ROOT = BACKEND_ROOT.parent

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Must be set before app.config is imported anywhere.
os.environ.setdefault("MARINA_ROOT", str(MARINA_ROOT))
os.environ.setdefault("DATA_DIR", "/tmp/marina-tests-nodata")
os.environ.setdefault("DATASET_ROOT", "/tmp/marina-tests-nodata")
os.environ.setdefault("STATS_PATH", "")          # in-memory counters
os.environ.setdefault("RATE_LIMIT_PREDICT", "")  # off unless a test asks for it
os.environ.setdefault("RATE_LIMIT_SMILES", "")


def _marina_importable() -> bool:
    return (MARINA_ROOT / "src" / "modules").is_dir()


requires_marina = pytest.mark.skipif(
    not _marina_importable(),
    reason="MARINA src/ tree not reachable from the backend checkout",
)


@pytest.fixture(scope="session")
def marina_src():
    """Import path bootstrap for tests that touch src.modules.*."""
    if not _marina_importable():
        pytest.skip("MARINA src/ tree not reachable")
    if str(MARINA_ROOT) not in sys.path:
        sys.path.insert(0, str(MARINA_ROOT))
    return MARINA_ROOT


@pytest.fixture
def client():
    """TestClient without the lifespan, so no model or worker pool is needed."""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


# ── Synthetic model fixture ───────────────────────────────────────────────────

FIXTURE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCCCCCOc1ccccc1C(=O)NC",
    "c1ccccc1",
    "CCO",
    "CC(C)CC1NC(=O)C(Cc2ccccc2)NC(=O)C(CO)NC(=O)C1",
]
FIXTURE_RADIUS = 6


def build_model_dir(root: Path, model_type: str, *, normalize: bool = True,
                    out_dim_override: int | None = None,
                    extra_params: dict | None = None) -> Path:
    """
    Write a complete, loadable model directory with randomly-initialised weights.

    Real checkpoints are not needed to exercise loading, collation, retrieval and
    rendering — only the file layout and tensor shapes have to be right.
    """
    import torch
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    from src.modules.data.fp_utils import get_bitinfos

    root.mkdir(parents=True, exist_ok=True)
    (root / "RankingEntropy").mkdir(exist_ok=True)

    cols: dict = {}
    for smi in FIXTURE_SMILES:
        for bit in sorted(get_bitinfos(smi, FIXTURE_RADIUS)[1]):
            cols.setdefault(bit, len(cols))
    n_features = len(cols)

    with open(root / "RankingEntropy" / "bitinfo_to_idx.pkl", "wb") as fh:
        pickle.dump(cols, fh)

    rows = torch.zeros(len(FIXTURE_SMILES), n_features)
    for i, smi in enumerate(FIXTURE_SMILES):
        for bit in get_bitinfos(smi, FIXTURE_RADIUS)[1]:
            rows[i, cols[bit]] = 1.0
    if normalize:
        # build_rankingset_csr L2-normalises rows so cosine == dot product.
        rows = torch.nn.functional.normalize(rows, dim=1, p=2.0)
    torch.save(rows.to_sparse_csr(), root / "RankingEntropy" / "rankingset.pt")

    with open(root / "retrieval.pkl", "wb") as fh:
        pickle.dump([{"smiles": s} for s in FIXTURE_SMILES], fh)

    (root / "metadata.json").write_text(json.dumps({
        str(i): {
            "canonical_2d_smiles": smi,
            "mw": Descriptors.MolWt(Chem.MolFromSmiles(smi)),
        }
        for i, smi in enumerate(FIXTURE_SMILES)
    }))

    # Deliberately small so the fixture builds in well under a second. The
    # coordinate dims must still sum to dim_model per encoder, and the two model
    # types shape them differently: SPECTRE packs every modality into one
    # 3-column stream, MARINA keeps a separate encoder per modality.
    params: dict = {"out_dim": out_dim_override or n_features,
                    "dim_model": 64, "heads": 4, "layers": 2, "ff_dim": 128}
    if model_type == "spectre":
        params.update(nmr_dim_coords=[30, 30, 4],
                      ms_dim_coords=[32, 32, 0],
                      mw_dim_coords=[64, 0, 0])
    else:
        params.update(nmr_dim_coords=[30, 30, 4],
                      c_nmr_dim_coords=[64],
                      h_nmr_dim_coords=[64],
                      ms_dim_coords=[32, 32],
                      mw_dim_coords=[64])
    if extra_params:
        params.update(extra_params)
    (root / "params.json").write_text(json.dumps(params))

    if model_type == "spectre":
        from src.modules.spectre.args import SPECTREArgs as Args
        from src.modules.spectre.model import SPECTRE as Model
    else:
        from src.modules.marina.args import MARINAArgs as Args
        from src.modules.marina.model import MARINA as Model

    model = Model(Args(**{k: v for k, v in params.items()
                          if k in Args.__dataclass_fields__}), fp_loader=None)
    torch.save({"state_dict": model.state_dict()}, root / "best.ckpt")
    return root


@pytest.fixture(scope="session")
def spectre_model_dir(marina_src, tmp_path_factory) -> Path:
    """A loadable synthetic SPECTRE model directory, built once per session."""
    root = tmp_path_factory.mktemp("spectre_model")
    return build_model_dir(root / "spectre_synth", "spectre")


@pytest.fixture
def spectre_session(spectre_model_dir):
    from app.session import ModelSession
    return ModelSession.from_model_root(str(spectre_model_dir), model_type="spectre")
