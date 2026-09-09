# fp_loader.py
from __future__ import annotations

import os
import time
import pickle
from typing import Optional, Dict

import numpy as np
import torch

from ..core.const import DATASET_ROOT
from ..log import get_logger
from .arrow_store import open_fragidx_store

from .fp_utils import (
    Feature,                       # BitInfo 4-tuple, or fragment SMILES for SUBSTRUCTURE
    MORGAN,
    SUBSTRUCTURE,
    MULTIPLICITY,
    MULTIPLICITY_UNCAPPED,
    UNIQUE_MULTIPLICITY,
    compute_entropy,
    load_smiles_index,
    count_fragments_over_retrieval,
    write_counts,
    build_rankingset_csr,
    build_fragidx_parquets,
    extract_features,
    canonicalize_smiles,
)

DEFAULT_FP_RADIUS = int(os.environ.get("FP_RADIUS", "10"))


class FPLoader:
    def build_mfp(self, idx: int) -> torch.Tensor:
        raise NotImplementedError()

    def load_rankingset(self, fp_type: str) -> torch.Tensor:
        raise NotImplementedError()


class EntropyFPLoader(FPLoader):
    """
    Select top-K features by entropy (from retrieval set), then:
      • build MFPs for training molecules using pre-mapped FragIdx (Arrow int32 lists)
      • build a CSR rankingset over the retrieval molecules (optional)

    Data layout (required):
        DATASET_ROOT/arrow/<split>/<FRAGIDX_FILENAME>  # per-idx sorted int32 col indices

    Subclasses select a feature vocabulary via FEATURE_KIND. Every on-disk artifact is
    named from that kind, because column indices and counts are only meaningful against
    the feature map that produced them.
    """

    FEATURE_KIND: str = MORGAN
    FRAGIDX_FILENAME: str = "FragIdx.parquet"
    COUNTS_PREFIX: str = "count_hashes_under_radius"

    def __init__(
        self,
        dataset_root: str = DATASET_ROOT,
        retrieval_path: Optional[str] = None,
    ) -> None:
        # Config
        self.dataset_root = dataset_root
        self.retrieval_path = retrieval_path

        # Selection state
        self.max_radius: Optional[int] = None
        self.out_dim: Optional[int] = None
        self.bitinfo_to_fp_index_map: Dict[Feature, int] = {}
        self.fp_index_to_bitinfo_map: Dict[int, Feature] = {}

        # Index for idx→split routing (loaded lazily)
        self._idx_to_split: Optional[Dict[int, str]] = None
        self._index_loaded: bool = False

        # Split-scoped FragIdx stores (lazy-open on first use)
        # store both split keys and FragIdx keys
        self._frag_stores: Dict[str, Optional[ArrowFragIdxStore]] = {}

    # ---------- internal helpers: index & Arrow FragIdx ----------

    def _ensure_index(self):
        if self._index_loaded:
            return
        index_path = os.path.join(self.dataset_root, "index.pkl")
        with open(index_path, "rb") as f:
            index = pickle.load(f)
        # Build compact idx→split map
        self._idx_to_split = {int(idx): entry.get("split", "train")
                              for idx, entry in index.items()}
        self._index_loaded = True

    def _split_for_idx(self, idx: int) -> str:
        self._ensure_index()
        return self._idx_to_split[int(idx)]  # KeyError → fail fast if missing

    def _ensure_fragidx_store_for_split(self, split: str) -> ArrowFragIdxStore:
        key = f"{split}__{self.FRAGIDX_FILENAME}"
        store = self._frag_stores.get(key)
        if store is None:
            arrow_path = os.path.join(
                self.dataset_root, "arrow", split, self.FRAGIDX_FILENAME)
            if not os.path.isfile(arrow_path):
                raise FileNotFoundError(f"Missing FragIdx shard: {arrow_path}")
            store = open_fragidx_store(arrow_path)
            self._frag_stores[key] = store
        return store

    def _load_fragment_indices(self, idx: int) -> np.ndarray:
        """
        Returns np.int32 array of sorted unique FP column indices for this idx.
        """
        split = self._split_for_idx(idx)
        store = self._ensure_fragidx_store_for_split(split)
        return store.get_indices(idx)

    def _counts_path(self, radius: int) -> str:
        return os.path.join(self.dataset_root, f"{self.COUNTS_PREFIX}_{radius}.pkl")

    def _prepare_counts(self, radius: int, num_procs: int = 0):
        if self.retrieval_path is None:
            raise ValueError("EntropyFPLoader: retrieval_path not set.")
        counts_path = self._counts_path(radius)
        if os.path.exists(counts_path):
            with open(counts_path, "rb") as f:
                return pickle.load(f)
        counter = count_fragments_over_retrieval(
            self.retrieval_path, radius, num_procs=num_procs,
            feature_kind=self.FEATURE_KIND)
        write_counts(counter, counts_path)
        return counter

    def _filter_by_radius(self, counts: Dict[Feature, int]) -> list:
        """
        Drop features above max_radius. A counts file built once at a given radius is
        reusable for any smaller max_radius because radius is part of the key.
        """
        return [(k, c) for k, c in counts.items() if k[3] <= self.max_radius]

    def setup(self, out_dim, max_radius, fp_type: str = "RankingEntropy",
              retrieval_path: Optional[str] = None, num_procs: int = 0):
        logger = get_logger(__file__)

        if retrieval_path is not None:
            self.retrieval_path = retrieval_path

        # Fast path: precomputed mapping already on disk
        mapping_path = os.path.join(self.dataset_root, fp_type, "bitinfo_to_idx.pkl")
        if os.path.exists(mapping_path):
            with open(mapping_path, "rb") as f:
                self.bitinfo_to_fp_index_map = pickle.load(f)
            self.fp_index_to_bitinfo_map = {v: k for k, v in self.bitinfo_to_fp_index_map.items()}
            self.max_radius = int(max_radius)
            self.out_dim = len(self.bitinfo_to_fp_index_map)
            logger.info(f"Loaded precomputed feature map ({self.out_dim} features) from {mapping_path}")
            return

        # Full build: counts → entropy selection → fragidx → rankingset
        logger.info("Setting up EntropyFPLoader (full build)...")
        start = time.time()

        self.max_radius = int(max_radius)
        hashed_bits_count = self._prepare_counts(self.max_radius, num_procs)
        if not hashed_bits_count:
            raise RuntimeError("Failed to load or compute retrieval counts.")

        filtered = self._filter_by_radius(hashed_bits_count)
        if not filtered:
            raise RuntimeError(
                "No features <= max_radius found in retrieval counts.")

        bitinfos, counts = zip(*filtered)
        counts = np.asarray(counts)
        logger.debug(f"Found {len(bitinfos)} features with radius <= {self.max_radius}.")

        retrieval_size = len(load_smiles_index(self.retrieval_path))
        ent = compute_entropy(counts, total_dataset_size=retrieval_size)

        k = len(filtered) if (out_dim == "inf" or out_dim == float("inf")) else int(out_dim)
        topk_idx = np.argpartition(-ent, kth=min(k, len(ent)-1))[:k]
        topk_sorted = sorted(topk_idx, key=lambda i: (-ent[i], bitinfos[i]))

        self.out_dim = len(topk_sorted)
        self.bitinfo_to_fp_index_map = {
            bitinfos[i]: j for j, i in enumerate(topk_sorted)}
        self.fp_index_to_bitinfo_map = {
            v: k for k, v in self.bitinfo_to_fp_index_map.items()}

        elapsed = time.time() - start
        logger.info(f"Done! Selected {self.out_dim} features in {elapsed:.2f}s.")

        index_path = os.path.join(self.dataset_root, "index.pkl")
        logger.info("Building FragIdx parquets for all splits...")
        build_fragidx_parquets(
            index_path=index_path,
            out_dir=self.dataset_root,
            bitinfo_to_col=self.bitinfo_to_fp_index_map,
            radius=self.max_radius,
            num_procs=num_procs,
            feature_kind=self.FEATURE_KIND,
            filename=self.FRAGIDX_FILENAME,
        )
        logger.info("Building rankingset...")
        self._build_rankingset(fp_type, num_procs=num_procs)

    def build_mfp(self, idx: int) -> torch.Tensor:
        if self.out_dim is None:
            raise RuntimeError("Call setup() first.")
        cols = self._load_fragment_indices(idx)
        mfp = np.zeros(self.out_dim, dtype=np.float32)
        if cols.size > 0:
            mfp[cols] = 1.0
        return torch.from_numpy(mfp)

    def build_mfp_for_smiles(self, smiles: str, ignore_atoms=None) -> torch.Tensor:
        if self.out_dim is None or self.max_radius is None:
            raise RuntimeError("Call setup() first.")
        mfp = np.zeros(self.out_dim, dtype=np.float32)
        present = extract_features(
            smiles, radius=self.max_radius, kind=self.FEATURE_KIND,
            ignore_atoms=ignore_atoms or [])
        for bitinfo in present.keys():
            col = self.bitinfo_to_fp_index_map.get(bitinfo)
            if col is not None and 0 <= col < self.out_dim:
                mfp[col] = 1.0
        return torch.from_numpy(mfp)

    def build_mfp_from_bitinfo(self, atom_to_bitinfos: Dict[int, list], ignore_atoms=None) -> torch.Tensor:
        if self.out_dim is None:
            raise RuntimeError("Call setup() first.")
        mfp = np.zeros(self.out_dim, dtype=np.float32)
        ignore = set(ignore_atoms or [])
        for atom_idx, bitinfos in atom_to_bitinfos.items():
            if atom_idx in ignore:
                continue
            for bitinfo in bitinfos:
                col = self.bitinfo_to_fp_index_map.get(bitinfo)
                if col is not None and 0 <= col < self.out_dim:
                    mfp[col] = 1.0
        return torch.from_numpy(mfp)

    def _build_rankingset(self, fp_type: str, num_procs: int = 0) -> torch.Tensor:
        if self.max_radius is None or not self.bitinfo_to_fp_index_map:
            raise RuntimeError("Call setup() first.")
        csr = build_rankingset_csr(
            retrieval_path=self.retrieval_path,
            bitinfo_to_col=self.bitinfo_to_fp_index_map,
            radius=self.max_radius,
            num_procs=num_procs,
            feature_kind=self.FEATURE_KIND,
        )
        out_dir = os.path.join(self.dataset_root, fp_type)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(csr, os.path.join(out_dir, "rankingset.pt"))
        with open(os.path.join(out_dir, "bitinfo_to_idx.pkl"), "wb") as f:
            pickle.dump(self.bitinfo_to_fp_index_map, f)
        return csr

    def load_rankingset(self, fp_type: str) -> torch.Tensor:
        rankingset_path = os.path.join(self.dataset_root, fp_type, "rankingset.pt")
        if not os.path.exists(rankingset_path):
            get_logger(__file__).info(
                f"Rankingset not found at {rankingset_path}, building...")
            self._build_rankingset(fp_type)
        return torch.load(rankingset_path, weights_only=True)


class SubstructureEntropyFPLoader(EntropyFPLoader):
    """
    Entropy-selected fingerprint over canonical fragment SMILES instead of Morgan hashes.

    Same selection and storage machinery as the parent; only the feature vocabulary
    changes. Features that describe the same substructure collapse to one bit, which
    recovers the ~54% of the top-16384 Morgan selection spent on duplicate chemistry.
    """

    FEATURE_KIND = SUBSTRUCTURE
    FRAGIDX_FILENAME = "FragIdxSubstructure.parquet"
    COUNTS_PREFIX = "count_substructures_under_radius"

    def _filter_by_radius(self, counts: Dict[Feature, int]) -> list:
        """
        Fragment SMILES carry no radius, so the counts file cannot be sub-filtered the way
        the Morgan one can. _counts_path is already radius-scoped, so the file for radius R
        holds exactly the features at radius <= R.
        """
        return list(counts.items())

    def build_mfp_from_bitinfo(self, atom_to_bitinfos, ignore_atoms=None):
        raise NotImplementedError(
            "build_mfp_from_bitinfo takes Morgan BitInfo keys; the substructure loader "
            "has fragment-SMILES keys. Use build_mfp_for_smiles instead."
        )


class MultiplicityEntropyFPLoader(SubstructureEntropyFPLoader):
    """
    Entropy-selected fingerprint over (fragment SMILES, multiplicity bucket) pairs.

    Extends the substructure vocabulary with how many times each fragment occurs in the
    molecule, as cumulative buckets >=1, >=2, >=3, >=4, >=5. The feature stays binary -- a
    fragment occurring n times lights every bucket up to min(n, 5) -- so the model, the loss
    and the CSR rankingset are untouched. Only the candidate pool entering entropy selection
    grows.

    Cumulative rather than exact buckets, because retrieval is cosine over these columns:
    exact buckets would give two molecules differing only in a fragment's multiplicity no
    shared column for it at all, making a near-miss on count as expensive as wrong chemistry.

    Since (frag, >=1) is exactly the parent's feature, this vocabulary is a strict superset
    of the substructure one competing for the same output width. Selection spends the budget
    across both chemistry and multiplicity, so the selected columns are NOT all distinct
    fragments the way the parent's are -- that trades away the parent's "0% duplicate
    chemistry" property in exchange for representing repetition at all.

    Inherits _filter_by_radius from the substructure loader: these keys carry no radius
    either, so the radius-scoped counts file already holds exactly the right features.
    """

    FEATURE_KIND = MULTIPLICITY
    FRAGIDX_FILENAME = "FragIdxMultiplicity.parquet"
    COUNTS_PREFIX = "count_multiplicity_under_radius"


class MultiplicityUncappedEntropyFPLoader(MultiplicityEntropyFPLoader):
    """Multiplicity vocabulary with no bucket ceiling (see MULTIPLICITY_CAP).

    Identical to the parent except a fragment occurring n times lights all n cumulative
    buckets rather than saturating at 5. This lets radius-0 atom environments count carbons,
    hydrogens etc. outright -- an implicit molecular-formula signal the capped variant throws
    away. Artifacts are namespaced separately so both vocabularies coexist on disk.
    """

    FEATURE_KIND = MULTIPLICITY_UNCAPPED
    FRAGIDX_FILENAME = "FragIdxMultiplicityUncapped.parquet"
    COUNTS_PREFIX = "count_multiplicity_uncapped_under_radius"


class UniqueMultiplicityEntropyFPLoader(MultiplicityUncappedEntropyFPLoader):
    """Uncapped multiplicity, but occurrences are deduplicated by atom set.

    Identical to the uncapped variant except the per-molecule count of a fragment is the
    number of DISTINCT (canonical fragment, atom-set) occurrences, not the number of
    atom-centred environments. This removes the per-atom over-counting of symmetric groups
    (an isopropyl the uncapped variant counts 3x is one here; an amide counted twice from the
    O@radius-2 and C@radius-1 centres is one) and, via re-canonicalising each fragment, the
    split of one substructure across two spellings (CC(N)=O vs CC(=O)N). Artifacts are
    namespaced separately so it coexists on disk with the other vocabularies.
    """

    FEATURE_KIND = UNIQUE_MULTIPLICITY
    FRAGIDX_FILENAME = "FragIdxUniqueMultiplicity.parquet"
    COUNTS_PREFIX = "count_unique_multiplicity_under_radius"


FP_LOADERS = {
    "RankingEntropy": EntropyFPLoader,
    "RankingEntropySubstructure": SubstructureEntropyFPLoader,
    "RankingEntropyMultiplicity": MultiplicityEntropyFPLoader,
    "RankingEntropyMultiplicityUncapped": MultiplicityUncappedEntropyFPLoader,
    "RankingEntropyUniqueMultiplicity": UniqueMultiplicityEntropyFPLoader,
}


def make_fp_loader(fp_type: str, entropy_out_dim=16384, max_radius=DEFAULT_FP_RADIUS, retrieval_path: Optional[str] = None):
    loader_class = FP_LOADERS.get(fp_type)
    if loader_class is None:
        raise NotImplementedError(f"FP type {fp_type} not implemented")
    fp_loader = loader_class(retrieval_path=retrieval_path)
    fp_loader.setup(entropy_out_dim, max_radius, fp_type=fp_type,
                    retrieval_path=retrieval_path)
    return fp_loader

