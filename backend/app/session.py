"""
ModelSession: loads a MARINA or SPECTRE model and its associated resources
(fingerprint loader, rankingset, metadata, MW index) from a per-model directory.

Each model directory must contain:
    best.ckpt       – PyTorch Lightning checkpoint
    params.json     – JSON dict of model hyperparameters
    retrieval.pkl   – List[Dict] or Dict[int, Dict] with SMILES entries
    metadata.json   – Dict[str, Dict] keyed by str(index)

Optional (auto-built if absent):
    RankingEntropy/rankingset.pt
    RankingEntropy/bitinfo_to_idx.pkl

Optional (feature is simply off when absent):
    npclassifier.json – NPClassifier annotations, keyed by str(index) like metadata.json
"""
from __future__ import annotations

import bisect
import glob
import json
import logging
import os
import re
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Distinct MW-filter ranges to keep materialised at once. Each entry is a full
# copy of the rankingset, and the key is user-supplied, so this must stay small.
MAX_FILTERED_CACHE = 4

# Per-model cache of monoisotopic masses, one entry per rankingset row (null
# where the structure could not be parsed). Written on first use; precomputable
# with scripts/website/build_mw_index.py.
MW_INDEX_FILENAME = "mw_index.json"
# Per-row element-count index (row i → {element: count} or null), aligned to the
# rankingset like mw_index.json. Built by scripts/website/build_formula_index.py.
FORMULA_INDEX_FILENAME = "formula_index.json"

# Fallback Morgan radius, used only when a model dir carries no count file to infer
# from. 10 is the project-wide default (src.modules.data.fp_loader.DEFAULT_FP_RADIUS,
# same FP_RADIUS env).
FP_RADIUS_DEFAULT = int(os.environ.get("FP_RADIUS", "10"))


def _infer_fp_radius(model_root: str) -> int:
    """
    Radius the stored fingerprints were built at, read from the count filename
    (count_<kind>_under_radius_<R>.pkl) in the model dir. This MUST match the radius
    used to build rankingset.pt: query FPs are extracted at this radius, so a stale
    value builds them at the wrong radius and the same molecule fails to self-retrieve
    at 1.0. A radius-6 constant serving a radius-10 model is exactly that bug.
    """
    radii = [
        int(m.group(1))
        for p in glob.glob(os.path.join(model_root, "count_*_under_radius_*.pkl"))
        if (m := re.search(r"under_radius_(\d+)\.pkl$", os.path.basename(p)))
    ]
    return max(radii) if radii else FP_RADIUS_DEFAULT


class MWDataUnavailable(RuntimeError):
    """Raised when a MW filter is requested but no masses could be derived."""


class FormulaDataUnavailable(RuntimeError):
    """Raised when a formula/atom-count filter is requested but no counts are available."""


# ── Lazy imports from MARINA src ─────────────────────────────────────────────

def _import_marina():
    """Import MARINA model class (deferred until after sys.path is set up)."""
    from src.modules.marina.model import MARINA
    return MARINA


def _import_spectre():
    from src.modules.spectre.model import SPECTRE
    return SPECTRE


def _import_marina_args():
    from src.modules.marina.args import MARINAArgs
    return MARINAArgs


def _import_spectre_args():
    from src.modules.spectre.args import SPECTREArgs
    return SPECTREArgs


def _import_fp_loader(fp_type: str = "RankingEntropy"):
    """
    Resolve the loader class for a model's fp_type.

    A substructure model keys its feature map on fragment SMILES rather than Morgan
    BitInfo tuples, so serving it with the Morgan loader would look up every column
    against the wrong key type and silently return an all-zero fingerprint.
    """
    from src.modules.data.fp_loader import FP_LOADERS
    loader_class = FP_LOADERS.get(fp_type)
    if loader_class is None:
        raise RuntimeError(
            f"Unknown fp_type {fp_type!r}; known types: {sorted(FP_LOADERS)}"
        )
    return loader_class


# ── Helper: load metadata ────────────────────────────────────────────────────

def _load_json(path: str) -> Any:
    with open(path, "r") as fh:
        return json.load(fh)


# Compute workers never read the annotations (see disable_annotations), but they do
# build a full ModelSession, so without this they would each hold a private copy.
_annotations_enabled = True


def disable_annotations() -> None:
    """
    Skip NPClassifier annotations in this process.

    Result cards are built in the API process; a compute worker only returns
    scores, indices and the predicted fingerprint. The table costs ~270 MB per
    process, so a worker loading it pays for something it will never read.
    """
    global _annotations_enabled
    _annotations_enabled = False


def _load_npclassifier(path: str) -> Optional[Dict[str, Any]]:
    """
    NPClassifier annotations, or None when the model directory has none.

    The file interns its label strings, so it is validated here rather than at every
    lookup: a payload whose `labels` tables do not cover the ids in `entries` would
    otherwise surface as a wrong class name on a result card, which is worse than no
    annotation at all.
    """
    if not _annotations_enabled:
        return None
    if not os.path.exists(path):
        logger.info("No npclassifier.json at %s; class annotations disabled.", path)
        return None
    try:
        payload = _load_json(path)
        tiers = payload["tiers"]
        labels = {t: payload["labels"][t] for t in tiers}
        entries = payload["entries"]
    except Exception as exc:
        logger.warning("Ignoring unreadable %s: %s", path, exc)
        return None
    logger.info(
        "Loaded NPClassifier annotations for %d molecules (%s) from %s",
        len(entries), ", ".join(f"{len(labels[t])} {t}" for t in tiers), path,
    )
    return {"tiers": tiers, "labels": labels, "entries": entries}


def _load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load a checkpoint, preferring the sandboxed unpickler.

    weights_only=True blocks arbitrary code execution during unpickling, but
    rejects Lightning checkpoints that embed non-tensor objects, so fall back
    with a warning rather than refusing to start.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.warning(
            "Loading %s with weights_only=True failed (%s); falling back to a full "
            "unpickle. Only load checkpoints you trust.", path, exc,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def _mw_from_entry(entry: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Monoisotopic mass for one retrieval entry.

    Computed from the structure, because the metadata generator does not write
    an `mw` field at all — reading only that field left the MW index empty and
    the retrieval filter silently matching everything. The field is still
    honoured as a fallback for model directories that do carry one.

    Monoisotopic (not average) mass, to agree with the "Exact mass" shown on
    result cards: a user filtering to the mass they read off a card must get
    that card back.
    """
    if not entry:
        return None

    smiles = entry.get("canonical_3d_smiles")
    if not smiles or smiles == "N/A":
        smiles = entry.get("canonical_2d_smiles") or entry.get("smiles")
    if smiles:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return float(Descriptors.ExactMolWt(mol))
        except Exception:
            pass

    mw = entry.get("mw")
    if isinstance(mw, (int, float)):
        try:
            return float(mw)
        except Exception:
            pass
    return None


# ── RankingSet wrapper ────────────────────────────────────────────────────────

class RankingSet:
    """
    Thin wrapper around src.modules.core.ranker.RankingSet that exposes
    retrieve_with_scores (scores + indices) and handles sparse CSR row filtering.
    """

    def __init__(self, store: torch.Tensor, metric: str = "cosine") -> None:
        from src.modules.core.ranker import RankingSet as _RS
        self._rs = _RS(store=store, metric=metric)

    @property
    def data(self) -> torch.Tensor:
        return self._rs.data

    def retrieve_with_scores(
        self, query: torch.Tensor, n: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (scores, indices) tensors of shape (n,) for a single query."""
        if query.dim() == 1:
            query = query.unsqueeze(0)
        sims = self._rs._sims(query)          # (N, 1)
        k_eff = min(n, sims.size(0))
        sims_topk, idxs_topk = torch.topk(sims, k=k_eff, dim=0)
        return sims_topk.squeeze(1), idxs_topk.squeeze(1)


def _filter_csr_rows(store: torch.Tensor, kept: List[int]) -> torch.Tensor:
    """Return a new CSR tensor containing only the rows in `kept`."""
    if not kept:
        return torch.sparse_csr_tensor(
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.float32),
            size=(0, store.shape[1]),
        )
    if store.layout != torch.sparse_csr:
        return store[kept]

    crow = store.crow_indices()
    col  = store.col_indices()
    val  = store.values()

    # Vectorised gather: the equivalent Python loop reads two tensor scalars per
    # kept row, which costs seconds on a database-sized rankingset.
    rows     = torch.as_tensor(kept, dtype=torch.int64)
    starts   = crow[rows].to(torch.int64)
    ends     = crow[rows + 1].to(torch.int64)
    lengths  = ends - starts
    new_crow = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lengths, 0)])
    nnz      = int(new_crow[-1])

    if nnz == 0:
        return torch.sparse_csr_tensor(
            new_crow,
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.float32),
            size=(len(kept), store.shape[1]),
        )

    # Expand each kept row's [start, end) span into a flat gather index.
    offsets = torch.repeat_interleave(starts, lengths)
    within  = torch.arange(nnz, dtype=torch.int64) - torch.repeat_interleave(new_crow[:-1], lengths)
    gather  = offsets + within

    return torch.sparse_csr_tensor(
        new_crow,
        col[gather].to(torch.int64),
        val[gather].to(torch.float32),
        size=(len(kept), store.shape[1]),
    )


def _warn_if_not_normalized(store: torch.Tensor, fp_type: str, model_root: str) -> None:
    """
    Check that rankingset rows are unit length.

    RankingSet's "cosine" metric normalises only the query, so the stored rows
    must already be L2-normalised for the dot product to be a cosine (see
    build_rankingset_csr, which does this at build time). A rankingset carried
    over from another codebase may hold raw 0/1 rows instead — the scores then
    exceed 1, get clamped, and every result card reads 1.000.
    """
    try:
        n_rows = store.shape[0]
        if n_rows == 0:
            return
        sample = sorted({0, n_rows // 2, n_rows - 1})

        if store.layout == torch.sparse_csr:
            # CSR has no strides, so rows cannot be gathered by fancy indexing;
            # read each row's values straight out of the value array instead.
            crow, vals = store.crow_indices(), store.values()
            norms = torch.stack([
                vals[int(crow[r]):int(crow[r + 1])].float().pow(2).sum().sqrt()
                for r in sample
            ])
        else:
            norms = store[sample].float().pow(2).sum(dim=1).sqrt()

        norms = norms[norms > 0]
        if norms.numel() and not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
            logger.warning(
                "Rankingset %s/%s in %s does not have unit-norm rows (sampled norms: %s). "
                "Cosine scores will be inflated and clamped to 1.0. Rebuild it with "
                "build_rankingset_csr, or L2-normalise the rows before serving.",
                fp_type, "rankingset.pt", model_root,
                [round(float(v), 3) for v in norms[:3]],
            )
    except Exception as exc:
        logger.debug("Rankingset normalisation check skipped: %s", exc)


# ── ModelSession ──────────────────────────────────────────────────────────────

@dataclass
class ModelSession:
    """Encapsulates a loaded model with all associated retrieval resources."""

    model_type: str                    # "marina" or "spectre"
    model: torch.nn.Module
    fp_loader: Any                     # EntropyFPLoader
    fp_type: str
    model_root: str
    device: torch.device
    _metadata: Dict[str, Any]
    _npclassifier: Optional[Dict[str, Any]] = None

    # Lazily built retrieval index
    _rankingset_store:   Optional[torch.Tensor]          = field(default=None, repr=False)
    _rankingset_wrapper: Optional[RankingSet]            = field(default=None, repr=False)
    _filtered_cache:     "OrderedDict[Tuple, Tuple[RankingSet, List[int]]]" = field(
        default_factory=OrderedDict, repr=False)
    _mw_sorted:          Optional[List[Tuple[float,int]]] = field(default=None, repr=False)
    _mw_by_idx:          Optional[Dict[int, float]]      = field(default=None, repr=False)
    _mw_values:          Optional[List[float]]           = field(default=None, repr=False)
    _no_mw:              Optional[List[int]]             = field(default=None, repr=False)
    _all_indices:        Optional[List[int]]             = field(default=None, repr=False)
    # Per-row element counts for the atom-count filter. [] once loaded but absent
    # (distinct from None = not yet loaded); each present row is {element: count} or None.
    _formula_index:      Optional[List[Optional[Dict[str, int]]]] = field(default=None, repr=False)
    _num_rows:           Optional[int]                   = field(default=None, repr=False)
    _lock:               threading.RLock                 = field(default_factory=threading.RLock, repr=False)

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_model_root(
        cls,
        model_root: str,
        model_type: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> "ModelSession":
        from app.marina_import import ensure_marina_importable
        ensure_marina_importable()

        if device is None:
            from app.config import DEVICE
            device = torch.device(DEVICE)

        model_type = (model_type or "marina").lower()

        params_path    = os.path.join(model_root, "params.json")
        ckpt_path      = os.path.join(model_root, "best.ckpt")
        metadata_path  = os.path.join(model_root, "metadata.json")
        retrieval_path = os.path.join(model_root, "retrieval.pkl")

        params = _load_json(params_path)

        if model_type == "spectre":
            Args  = _import_spectre_args()
            Model = _import_spectre()
        else:
            Args  = _import_marina_args()
            Model = _import_marina()

        # params.json written against a different implementation (notably the
        # published SPECTRE code, whose architecture args differ from this one)
        # must not load silently: pydantic dataclasses DISCARD unknown keys, so
        # the model would be built from this codebase's defaults instead of the
        # settings the checkpoint was trained with.
        known = set(getattr(Args, "__dataclass_fields__", {}))
        unknown = sorted(set(params) - known) if known else []
        if unknown:
            raise RuntimeError(
                f"params.json at {params_path!r} contains {len(unknown)} key(s) the "
                f"{model_type} implementation in this codebase does not define: {unknown}. "
                "These would be ignored, silently building the model with default "
                "settings instead of the ones it was trained with. Translate the file "
                "to this implementation's arguments before serving it."
            )

        try:
            args = Args(**params)
        except Exception as exc:
            raise RuntimeError(
                f"params.json at {params_path!r} is not valid for the {model_type} "
                f"model: {exc}"
            ) from exc

        fp_type = getattr(args, "fp_type", "RankingEntropy")

        FPLoaderClass = _import_fp_loader(fp_type)
        fp_loader = FPLoaderClass(dataset_root=model_root, retrieval_path=retrieval_path)
        fp_loader.setup(args.out_dim, _infer_fp_radius(model_root), fp_type=fp_type,
                        retrieval_path=retrieval_path)

        # The model's output layer is sized from args.out_dim while retrieval
        # matmuls against the feature map's width. If they disagree the scores
        # are computed against the wrong columns, or the matmul fails outright.
        if fp_loader.out_dim != args.out_dim:
            raise RuntimeError(
                f"Fingerprint size mismatch for model at {model_root!r}: params.json declares "
                f"out_dim={args.out_dim} but the feature map in {fp_type}/bitinfo_to_idx.pkl "
                f"has {fp_loader.out_dim} entries. Rebuild the feature map or correct params.json."
            )

        model = Model(args, fp_loader)

        logger.debug("Loading checkpoint from %s → %s", ckpt_path, device)
        ckpt = _load_checkpoint(ckpt_path)
        state_dict = ckpt.get("state_dict", ckpt)

        # strict=False is kept because Lightning checkpoints carry buffers the
        # inference model does not declare, but the result must be inspected:
        # silently loading a mismatched checkpoint leaves the weights randomly
        # initialised and the model serves plausible-looking garbage.
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing    = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if missing:
            raise RuntimeError(
                f"Checkpoint {ckpt_path!r} is missing {len(missing)} parameter(s) required by "
                f"the {model_type} model — the weights would be randomly initialised. "
                f"First few: {missing[:5]}"
            )
        if unexpected:
            logger.warning(
                "Checkpoint %s has %d unexpected key(s), ignored. First few: %s",
                ckpt_path, len(unexpected), unexpected[:5],
            )

        model.to(device)
        model.eval()

        metadata = _load_json(metadata_path)
        npclassifier = _load_npclassifier(os.path.join(model_root, "npclassifier.json"))

        return cls(
            model_type=model_type,
            model=model,
            fp_loader=fp_loader,
            fp_type=getattr(args, "fp_type", "RankingEntropy"),
            model_root=model_root,
            device=device,
            _metadata=metadata,
            _npclassifier=npclassifier,
        )

    # ── Metadata helpers ─────────────────────────────────────────────────────

    def get_entry(self, idx: int) -> Optional[Dict[str, Any]]:
        return self._metadata.get(str(idx))

    def get_npclassifier(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        NPClassifier annotation for a retrieval index, de-interned into label strings.

        None when annotations are not loaded or this molecule has no record. A molecule
        NPClassifier declined to classify has a record with empty tiers, which is a
        different fact and is reported as such.
        """
        if not self._npclassifier:
            return None
        row = self._npclassifier["entries"].get(str(idx))
        if row is None:
            return None
        tiers = self._npclassifier["tiers"]
        labels = self._npclassifier["labels"]
        out: Dict[str, Any] = {}
        for tier, ids in zip(tiers, row):
            table = labels[tier]
            out[tier] = [table[i] for i in ids if 0 <= i < len(table)]
        out["isglycoside"] = bool(row[len(tiers)])
        return out

    def get_smiles(self, idx: int) -> Optional[str]:
        entry = self.get_entry(idx)
        if not entry:
            return None
        smi = entry.get("canonical_3d_smiles")
        if smi and smi != "N/A":
            return smi
        return entry.get("canonical_2d_smiles")

    def get_exact_mass(self, idx: int) -> Optional[float]:
        return _mw_from_entry(self.get_entry(idx))

    # ── MW index & filtering ─────────────────────────────────────────────────

    def _load_mw_cache(self, n: int) -> Optional[List[Optional[float]]]:
        """Reads the precomputed mass sidecar, or None if it is absent or stale."""
        path = os.path.join(self.model_root, MW_INDEX_FILENAME)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as fh:
                masses = json.load(fh)
            if not isinstance(masses, list) or len(masses) != n:
                logger.warning(
                    "%s holds %s entries but the rankingset has %d rows — rebuilding.",
                    path, len(masses) if isinstance(masses, list) else "?", n,
                )
                return None
            logger.info("Loaded %d masses from %s", n, path)
            return masses
        except Exception as exc:
            logger.warning("Could not read %s: %s — rebuilding.", path, exc)
            return None

    def _save_mw_cache(self, masses: List[Optional[float]]) -> None:
        """Best-effort persist. A read-only model mount is a normal deployment."""
        path = os.path.join(self.model_root, MW_INDEX_FILENAME)
        try:
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".")
            with os.fdopen(fd, "w") as fh:
                json.dump(masses, fh)
            os.replace(tmp, path)
            logger.info("Wrote mass index to %s", path)
        except Exception as exc:
            logger.warning(
                "Could not cache the mass index to %s: %s — it will be recomputed "
                "on the next start.", path, exc,
            )

    def _ensure_store(self) -> None:
        """Loads the rankingset. On the path of every query, so it stays cheap."""
        with self._lock:
            if self._rankingset_store is not None:
                return
            store = self.fp_loader.load_rankingset(self.fp_type)
            _warn_if_not_normalized(store, self.fp_type, self.model_root)
            self._rankingset_store = store
            self._num_rows = store.shape[0]

    def _ensure_mw_index(self) -> None:
        """
        Builds the mass index, separately from loading the store: deriving
        masses is minutes of RDKit against a cold cache, and only MW-filtered
        queries need them. Doing both in one step would put that cost in front
        of every first query, filtered or not.
        """
        self._ensure_store()
        with self._lock:
            if self._mw_sorted is not None:
                return
            n = self._num_rows
            assert n is not None

            masses = self._load_mw_cache(n)
            if masses is None:
                # RDKit parses roughly 2k structures/s, so a full database is
                # minutes. Precompute with scripts/website/build_mw_index.py to
                # keep this off the first filtered request.
                logger.warning(
                    "No %s in %s — deriving %d masses from structures. This takes "
                    "a few minutes and is cached for subsequent starts.",
                    MW_INDEX_FILENAME, self.model_root, n,
                )
                masses = [_mw_from_entry(self._metadata.get(str(i))) for i in range(n)]
                self._save_mw_cache(masses)

            mw_by_idx: Dict[int, float] = {
                i: float(m) for i, m in enumerate(masses) if isinstance(m, (int, float))
            }
            if not mw_by_idx:
                logger.error(
                    "No molecular masses could be derived for %s — MW filtering "
                    "will be rejected rather than silently ignored.", self.model_root,
                )
            self._mw_by_idx = mw_by_idx
            self._mw_sorted = sorted((mw, idx) for idx, mw in mw_by_idx.items())
            # Precomputed once: indices_in_mw_range runs per request and would
            # otherwise rebuild both of these over the whole database each time.
            self._mw_values = [t[0] for t in self._mw_sorted]
            self._no_mw = sorted(i for i in range(n) if i not in mw_by_idx)

    def indices_in_mw_range(
        self,
        mw_min: Optional[float],
        mw_max: Optional[float],
    ) -> List[int]:
        """Return all retrieval indices whose MW is within [mw_min, mw_max]."""
        self._ensure_store()
        assert self._num_rows is not None

        if mw_min is None and mw_max is None:
            if self._all_indices is None:
                self._all_indices = list(range(self._num_rows))
            return self._all_indices

        self._ensure_mw_index()
        assert self._mw_sorted is not None and self._mw_values is not None
        assert self._no_mw is not None
        if not self._mw_sorted:
            # Returning everything here is how the filter came to be silently
            # ignored: the caller asked to narrow the search and got the whole
            # database back, with nothing to say so.
            raise MWDataUnavailable(
                f"No molecular masses are available for the model at {self.model_root!r}, "
                "so the MW filter cannot be applied."
            )

        lo = bisect.bisect_left(self._mw_values, mw_min)  if mw_min is not None else 0
        hi = bisect.bisect_right(self._mw_values, mw_max) if mw_max is not None else len(self._mw_values)
        in_range = [self._mw_sorted[i][1] for i in range(lo, hi)]
        # Entries with no recorded MW are kept rather than filtered out, so a
        # sparse metadata field cannot silently hide candidates.
        return sorted(in_range + self._no_mw)

    # ── Atom-count (molecular-formula) filter ────────────────────────────────

    def _ensure_formula_index(self) -> None:
        """Load formula_index.json (row → {element: count} or null), aligned to rows."""
        self._ensure_store()
        with self._lock:
            if self._formula_index is not None:
                return
            n = self._num_rows
            path = os.path.join(self.model_root, FORMULA_INDEX_FILENAME)
            data = None
            if os.path.isfile(path):
                try:
                    with open(path) as fh:
                        data = json.load(fh)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Failed to read %s: %s", path, exc)
            if not isinstance(data, list) or len(data) != n:
                if data is not None:
                    logger.error(
                        "%s has %s rows but the rankingset has %s — atom-count filtering disabled.",
                        FORMULA_INDEX_FILENAME, len(data) if isinstance(data, list) else "?", n)
                self._formula_index = []          # sentinel: unavailable
                return
            self._formula_index = data

    def indices_matching_formula(
        self,
        constraints: Dict[str, Tuple[Optional[int], Optional[int]]],
    ) -> List[int]:
        """
        Global indices whose per-element atom counts satisfy every constraint
        {element: (min, max)} (either bound may be None). Rows whose formula could
        not be parsed (null) are kept rather than dropped, mirroring the MW filter,
        so a sparse index cannot silently hide candidates.
        """
        self._ensure_store()
        assert self._num_rows is not None
        n = self._num_rows
        if not constraints:
            if self._all_indices is None:
                self._all_indices = list(range(n))
            return self._all_indices

        self._ensure_formula_index()
        if not self._formula_index:
            raise FormulaDataUnavailable(
                f"No atom-count data is available for the model at {self.model_root!r}, "
                "so the molecular-formula filter cannot be applied.")

        items = list(constraints.items())
        kept: List[int] = []
        for i in range(n):
            counts = self._formula_index[i]
            if counts is None:
                kept.append(i)                     # unparseable formula → keep
                continue
            ok = True
            for el, (lo, hi) in items:
                c = counts.get(el, 0)
                if (lo is not None and c < lo) or (hi is not None and c > hi):
                    ok = False
                    break
            if ok:
                kept.append(i)
        return kept

    # ── RankingSet access ────────────────────────────────────────────────────

    def get_rankingset(self) -> RankingSet:
        """Lazily load the full retrieval rankingset."""
        self._ensure_store()
        if self._rankingset_wrapper is None:
            assert self._rankingset_store is not None
            self._rankingset_wrapper = RankingSet(store=self._rankingset_store, metric="cosine")
        return self._rankingset_wrapper

    def get_filtered_rankingset(
        self,
        mw_min: Optional[float],
        mw_max: Optional[float],
        formula_filter: Optional[Dict[str, Tuple[Optional[int], Optional[int]]]] = None,
    ) -> Tuple[RankingSet, List[int]]:
        """
        Return a (RankingSet, kept_indices) pair filtered by MW range and/or an
        atom-count (molecular-formula) constraint. kept_indices maps local row
        positions back to global indices.

        Cached per (mw_min, mw_max, formula_key), bounded by MAX_FILTERED_CACHE.
        The bound matters: the key comes straight from user input and each entry
        holds a full copy of the rankingset, so an unbounded cache lets anyone
        exhaust memory just by varying the filter.
        """
        has_mw = mw_min is not None or mw_max is not None
        has_formula = bool(formula_filter)
        if not has_mw and not has_formula:
            return self.get_rankingset(), self.indices_in_mw_range(None, None)

        formula_key = (
            tuple(sorted((el, lo, hi) for el, (lo, hi) in formula_filter.items()))
            if has_formula else None
        )
        key = (mw_min, mw_max, formula_key)
        with self._lock:
            cached = self._filtered_cache.get(key)
            if cached is not None:
                self._filtered_cache.move_to_end(key)
                return cached

            if has_mw and has_formula:
                kept = sorted(set(self.indices_in_mw_range(mw_min, mw_max))
                              & set(self.indices_matching_formula(formula_filter)))
            elif has_mw:
                kept = self.indices_in_mw_range(mw_min, mw_max)
            else:
                kept = self.indices_matching_formula(formula_filter)

            base  = self.get_rankingset().data
            store = _filter_csr_rows(base, kept)
            rs    = RankingSet(store=store, metric="cosine")
            self._filtered_cache[key] = (rs, kept)
            while len(self._filtered_cache) > MAX_FILTERED_CACHE:
                evicted, _ = self._filtered_cache.popitem(last=False)
                logger.debug("Evicted filter cache entry %s", (evicted,))
            return rs, kept

    # ── Fingerprint helpers ──────────────────────────────────────────────────

    def fp_indices_for_smiles(self, smiles: str) -> Optional[List[int]]:
        """Return sorted list of active fingerprint bit indices for a SMILES string."""
        from src.modules.data.fp_utils import extract_features, MORGAN
        try:
            present = extract_features(
                smiles, self.fp_loader.max_radius,
                kind=getattr(self.fp_loader, "FEATURE_KIND", MORGAN))
            indices = sorted(
                col
                for b, col in (
                    (b, self.fp_loader.bitinfo_to_fp_index_map.get(b))
                    for b in present
                )
                if col is not None
            )
            return indices
        except Exception as exc:
            logger.warning("fp_indices_for_smiles failed for %r: %s", smiles, exc)
            return None

    def retrieved_fp_indices(self, global_idx: int) -> List[int]:
        """Return the non-zero column indices for a retrieval entry's fingerprint."""
        store = self.get_rankingset().data
        if store.layout == torch.sparse_csr:
            crow = store.crow_indices()
            col  = store.col_indices()
            s, e = int(crow[global_idx]), int(crow[global_idx + 1])
            return col[s:e].cpu().tolist() if e > s else []
        row = store[global_idx].cpu()
        nz  = torch.nonzero(row > 0.5, as_tuple=False).squeeze(-1)
        return nz.tolist() if nz.numel() else []
