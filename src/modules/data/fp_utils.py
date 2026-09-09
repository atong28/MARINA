# fp_utils.py
from __future__ import annotations

import os
import json
import math
import pickle
import multiprocessing as mp
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from tqdm import tqdm

from .smiles import canonicalize_smiles as _canonicalize_fixed_point

# ---------------------------
# Types
# ---------------------------
BitInfo = Tuple[int, str, str, int]  # (bit_id, atom_symbol, fragment_smiles, radius)
Substructure = str                   # canonical fragment SMILES
Multiplicity = Tuple[str, int]       # (canonical fragment SMILES, multiplicity bucket)
Feature = Any                        # BitInfo, Substructure or Multiplicity, per feature_kind

MORGAN = "morgan"
SUBSTRUCTURE = "substructure"
MULTIPLICITY = "multiplicity"
MULTIPLICITY_UNCAPPED = "multiplicity_uncapped"

# Buckets for the multiplicity vocabulary: a fragment occurring n times lands in
# min(n, MULTIPLICITY_CAP), so the top bucket means "n or more". 5 is where the measured
# distribution flattens -- over a 2,000-molecule MARINA1 sample, counts 1-5 cover 92.2% of
# (molecule, fragment) pairs, and the cap only bounds how far the candidate pool is
# enumerated. Selection prunes buckets that are too rare to be worth a bit anyway.
MULTIPLICITY_CAP = 5


def _multiplicity_cap(kind: str) -> int:
    """Bucket ceiling for a multiplicity vocabulary. The capped variant saturates at
    MULTIPLICITY_CAP; the uncapped variant applies no ceiling, so a fragment occurring n
    times lights all n cumulative buckets -- radius-0 atom environments then count carbons,
    hydrogens etc. outright, letting the vocabulary encode molecular-formula constraints."""
    if kind == MULTIPLICITY:
        return MULTIPLICITY_CAP
    if kind == MULTIPLICITY_UNCAPPED:
        return 1 << 30  # effectively unbounded; keeps min(n, cap) integer
    raise ValueError(f"Not a multiplicity kind: {kind}")

G_RADIUS = None
G_MAPPING = None  # for CSR bitinfo_to_col
G_KIND = MORGAN

def canonicalize_smiles(smiles: str, keep_stereo: bool = False):
    """Fixed-point canonical SMILES for fingerprint-identity keys. Delegates to the single
    source of truth in smiles.py (largest fragment + fixed point) so FP keys and the serving
    path agree with the index/retrieval/benchmark canonicalization; raises on invalid to
    preserve this module's contract with its CSR/enumeration workers (which cannot take None)."""
    if smiles is None or smiles == '':
        raise ValueError("Invalid empty SMILES")
    out = _canonicalize_fixed_point(smiles, keep_stereo=keep_stereo, largest_fragment=True)
    if out is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return out

def _init_count(radius: int, kind: str = MORGAN):
    global G_RADIUS, G_KIND
    G_RADIUS = radius
    G_KIND = kind

def _worker_count_one(smi: str):
    # uses G_RADIUS / G_KIND set by _init_count
    return extract_features(smi, G_RADIUS, kind=G_KIND)

def _init_csr(radius: int, mapping: Dict[Feature, int], kind: str = MORGAN):
    global G_RADIUS, G_MAPPING, G_KIND
    G_RADIUS = radius
    G_MAPPING = mapping
    G_KIND = kind

def _worker_row_nonzeros(args):
    # args: (row_idx, smi)
    row_idx, smi = args
    present = extract_features(smi, G_RADIUS, kind=G_KIND)
    cols = []
    get = G_MAPPING.get
    for b in present.keys():
        col = get(b)
        if col is not None:
            cols.append(col)
    cols = sorted(set(cols))
    return row_idx, cols

# ---------------------------
# IO helpers
# ---------------------------
def _load_as_dict(path: str) -> Any:
    with open(path, "rb") as f:
        if path.endswith(".pkl") or path.endswith(".pickle"):
            return pickle.load(f)
    with open(path, "r") as f:
        return json.load(f)


def load_smiles_index(path: str) -> Dict[int, str]:
    """
    Load an idx->smiles mapping from .json or .pkl.

    Accepts one of:
      - List[Dict[...]] with a smiles-like field
      - Dict[int, Dict[...]] with a smiles-like field
      - Dict[str|int, str] direct mapping to smiles

    Recognized smiles fields (first found wins): 'smiles', 'canonical_2d_smiles'
    """
    data = _load_as_dict(path)

    def _extract_smiles(rec: Any) -> Optional[str]:
        if isinstance(rec, str):
            return rec
        if isinstance(rec, dict):
            for key in ("smiles", "canonical_2d_smiles"):
                v = rec.get(key)
                if isinstance(v, str) and v:
                    return v
        return None

    smiles_map: Dict[int, str] = {}
    if isinstance(data, list):
        for idx, rec in enumerate(data):
            s = _extract_smiles(rec)
            if s:
                smiles_map[idx] = s
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str):
                s = v
            else:
                s = _extract_smiles(v)
            if s:
                try:
                    idx = int(k)
                except Exception:
                    continue
                smiles_map[idx] = s
    else:
        raise ValueError(f"Unsupported index structure in {path}")
    if not smiles_map:
        raise ValueError(f"No smiles found in {path}")
    return smiles_map


# ---------------------------
# Morgan / fragments
# ---------------------------
def _mk_rdkit(radius: int):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    return gen, ao


def get_bitinfos(
    smiles: str, radius: int, ignore_atoms: Optional[Iterable[int]] = None
) -> Tuple[Optional[Dict[int, List[BitInfo]]], Optional[Set[BitInfo]]]:
    """
    Extract Morgan bit environments for each atom index (for a specific radius).
    """
    ignore_atoms = tuple(ignore_atoms or ())
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    gen, ao = _mk_rdkit(radius)
    _ = gen.GetSparseFingerprint(mol, additionalOutput=ao)
    info = ao.GetBitInfoMap()

    atom_to_bit_infos: Dict[int, List[BitInfo]] = defaultdict(list)
    all_bit_infos: Set[BitInfo] = set()

    for bit_id, atom_envs in info.items():
        for atom_idx, curr_radius in atom_envs:
            if atom_idx in ignore_atoms:
                continue
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, curr_radius, atom_idx)
            submol = Chem.PathToSubmol(mol, env)
            frag_smiles = Chem.MolToSmiles(submol, canonical=True)
            atom_sym = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            bit_info: BitInfo = (bit_id, atom_sym, frag_smiles, curr_radius)
            atom_to_bit_infos[atom_idx].append(bit_info)
            all_bit_infos.add(bit_info)

    return atom_to_bit_infos, all_bit_infos


def count_circular_substructures(
    smiles: str, radius: int, ignore_atoms: Optional[Iterable[int]] = None
) -> Dict[BitInfo, int]:
    """
    Return presence map BitInfo -> 1 for a SMILES at given radius.
    """
    smiles = canonicalize_smiles(smiles)
    bit_info_counter: Dict[BitInfo, int] = defaultdict(int)
    atom_to_bit_infos, all_bit_infos = get_bitinfos(smiles, radius, ignore_atoms or ())
    if atom_to_bit_infos is None:
        return bit_info_counter
    for bit_info in all_bit_infos:
        bit_info_counter[bit_info] = 1
    return bit_info_counter


# ---------------------------
# Substructures
# ---------------------------
def _substructure_occurrences(mol, radius: int, ignore_atoms: Iterable[int] = ()):
    """
    Yield (atom_idx, fragment_smiles, env_radius) for every atom environment.

    isomericSmiles=False so a stereo-bearing input still produces fragment strings that
    match the vocabulary, which is built from SMILES already stripped of stereo by
    canonicalize_smiles. Without it, the backend would silently miss every bit on any
    candidate drawn with stereochemistry.
    """
    ignore = set(ignore_atoms or ())
    for atom_idx in range(mol.GetNumAtoms()):
        if atom_idx in ignore:
            continue
        for r in range(radius + 1):
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
            if r > 0 and len(env) == 0:
                break  # atom has no environment this large; neither will larger radii
            atoms = {atom_idx}
            for bond_idx in env:
                bond = mol.GetBondWithIdx(bond_idx)
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())
            frag = Chem.MolFragmentToSmiles(
                mol, atomsToUse=sorted(atoms), bondsToUse=list(env),
                canonical=True, isomericSmiles=False)
            yield atom_idx, frag, r


def get_feature_locations(
    smiles: str, radius: int, kind: str = MORGAN,
    ignore_atoms: Optional[Iterable[int]] = None,
) -> Optional[Dict[int, List[Tuple[Feature, int]]]]:
    """
    Map atom index -> [(feature key, environment radius), ...] for either vocabulary.

    The backend needs both halves: the feature key to look up a fingerprint column, and
    the radius to reconstruct that bit's atom/bond footprint for highlighting.

    The SMILES is used as given, NOT canonicalized -- atom indices have to line up with
    the molecule the renderer draws, and canonicalizing would renumber them.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if kind == MORGAN:
        atom_to_bits, _ = get_bitinfos(smiles, radius, ignore_atoms or ())
        if not atom_to_bits:
            return None
        return {a: [(b, b[3]) for b in bits] for a, bits in atom_to_bits.items()}

    if kind == SUBSTRUCTURE:
        out: Dict[int, List[Tuple[Feature, int]]] = defaultdict(list)
        for atom_idx, frag, r in _substructure_occurrences(mol, radius, ignore_atoms or ()):
            out[atom_idx].append((frag, r))
        return dict(out) or None

    if kind in (MULTIPLICITY, MULTIPLICITY_UNCAPPED):
        # The bucket is a property of the whole molecule, not of one occurrence, so the
        # occurrences have to be counted before any atom can be assigned its feature keys.
        # Buckets are cumulative, so an atom in a fragment occurring 3 times belongs to the
        # >=1, >=2 and >=3 columns alike -- the backend highlights the same atoms for each.
        cap = _multiplicity_cap(kind)
        occurrences = list(_substructure_occurrences(mol, radius, ignore_atoms or ()))
        counts = Counter(frag for _, frag, _ in occurrences)
        out = defaultdict(list)
        for atom_idx, frag, r in occurrences:
            for k in range(1, min(counts[frag], cap) + 1):
                out[atom_idx].append(((frag, k), r))
        return dict(out) or None

    raise ValueError(f"Unknown feature kind: {kind}")


def get_substructure_smiles(
    smiles: str, radius: int, ignore_atoms: Optional[Iterable[int]] = None
) -> Set[Substructure]:
    """
    Enumerate the atom environments of radius 0..radius as canonical fragment SMILES.

    This is the substructure analogue of get_bitinfos. Two environments that describe the
    same fragment collapse to one feature here, whereas Morgan hashing keeps them apart by
    bit_id: in the top-16384 entropy selection, 54% of the bits are duplicate chemistry
    (isopropyl alone occupies 124 of them).

    Fragments are written with MolFragmentToSmiles rather than PathToSubmol so that the
    center atom survives at radius 0 -- PathToSubmol on an empty bond environment returns an
    empty molecule, which would collapse every radius-0 feature into "".  Writing from the
    parent also keeps its aromaticity perception, so 'ccc' (benzene) stays distinct from
    'CCC' (cyclohexane).
    """
    mol = Chem.MolFromSmiles(canonicalize_smiles(smiles))
    if mol is None:
        return set()

    frags: Set[Substructure] = {
        frag for _, frag, _ in
        _substructure_occurrences(mol, radius, ignore_atoms or ())
    }

    return frags


def count_substructures(
    smiles: str, radius: int, ignore_atoms: Optional[Iterable[int]] = None
) -> Dict[Substructure, int]:
    """Return presence map Substructure -> 1 for a SMILES at given radius."""
    return {frag: 1 for frag in get_substructure_smiles(smiles, radius, ignore_atoms)}


def count_substructure_multiplicities(
    smiles: str, radius: int, ignore_atoms: Optional[Iterable[int]] = None,
    cap: int = MULTIPLICITY_CAP,
) -> Dict[Multiplicity, int]:
    """
    Presence map keyed on (fragment SMILES, multiplicity bucket).

    A fragment occurring n times in one molecule lights cumulative buckets 1..min(n, cap) --
    so "appears twice" is a different feature from "appears once", and the top bucket
    saturates at `cap` (5 for the capped variant, effectively unbounded for the uncapped one).

    The values are 1, not n. This is deliberately still a *presence* map, just over a
    larger vocabulary, which is what lets entropy selection, the FragIdx parquets, the CSR
    rankingset and the binary BCE/cosine loss all work unchanged: only the feature identity
    differs, exactly as it does for the substructure vocabulary.

    Entropy selection then decides how many buckets each fragment earns. A motif that is
    commonly doubled wins a bit for its doubled form; a rare one wins none. So the selected
    top-K are no longer K *distinct* fragments -- a fragment may hold several columns that
    denote different multiplicities of the same chemistry.

    The buckets are cumulative (thermometer), not exact: a fragment appearing 3 times lights
    (frag, 1), (frag, 2) AND (frag, 3). This is what keeps cosine similarity well behaved.
    Under exact buckets, a molecule with 3 copies and one with 4 would share no column at all
    for that fragment, so a near-miss on multiplicity would cost exactly as much as wrong
    chemistry; cumulatively they share 3 of 4 and the penalty degrades smoothly.

    A consequence worth naming: (frag, 1) is precisely the binary substructure feature, so
    this vocabulary is a strict superset of the substructure one competing for the same
    budget. It also means (frag, k+1) deterministically implies (frag, k), which will make
    the fingerprint look more redundant on any joint-entropy measure -- by construction, not
    as a defect. Judge this vocabulary on retrieval, not on bits-of-joint-entropy.
    """
    mol = Chem.MolFromSmiles(canonicalize_smiles(smiles))
    if mol is None:
        return {}
    occurrences = Counter(
        frag for _, frag, _ in _substructure_occurrences(mol, radius, ignore_atoms or ())
    )
    return {
        (frag, k): 1
        for frag, n in occurrences.items()
        for k in range(1, min(n, cap) + 1)
    }


def count_fragment_occurrences(
    smiles: str, radius: int, ignore_atoms: Optional[Iterable[int]] = None,
) -> Counter:
    """Return Counter[fragment SMILES -> occurrence count] for a SMILES.

    Unlike count_substructure_multiplicities (which returns a *presence* map over
    (frag, bucket) keys), this returns the raw multiplicity: how many radius-0..radius
    atom environments produce each fragment. It is the un-bucketed intermediate the
    thermometer vocabulary is built from -- 4_fp_rankingset.py needs the raw per-molecule
    counts to histogram them across the corpus and enumerate the uncapped rung ladder.

    Fragments come from the same _substructure_occurrences generator as every other
    fragment vocabulary here (MolFragmentToSmiles, isomericSmiles=False, over a
    stereo-stripped canonical SMILES), so the keys match those the multiplicity loader
    selects and stores."""
    mol = Chem.MolFromSmiles(canonicalize_smiles(smiles))
    if mol is None:
        return Counter()
    return Counter(
        frag for _, frag, _ in _substructure_occurrences(mol, radius, ignore_atoms or ())
    )


def extract_features(
    smiles: str, radius: int, kind: str = MORGAN,
    ignore_atoms: Optional[Iterable[int]] = None,
) -> Dict[Feature, int]:
    """Presence map for a feature vocabulary. Keys are BitInfo, fragment SMILES, or
    (fragment SMILES, multiplicity bucket)."""
    if kind == MORGAN:
        return count_circular_substructures(smiles, radius, ignore_atoms)
    if kind == SUBSTRUCTURE:
        return count_substructures(smiles, radius, ignore_atoms)
    if kind in (MULTIPLICITY, MULTIPLICITY_UNCAPPED):
        return count_substructure_multiplicities(
            smiles, radius, ignore_atoms, cap=_multiplicity_cap(kind))
    raise ValueError(f"Unknown feature kind: {kind}")


def merge_counts(counts_list: Iterable[Dict[BitInfo, int]]) -> Counter:
    total_count = Counter()
    for count in counts_list:
        total_count.update(count)
    return total_count


# ---------------------------
# Entropy
# ---------------------------
def compute_entropy(counts: np.ndarray, total_dataset_size: int) -> np.ndarray:
    """
    Standard binary entropy (positive):
      H(p) = - [ p log2 p + (1-p) log2 (1-p) ]
    """
    p = np.clip(counts / float(total_dataset_size), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def select_topk_by_entropy(entropies: np.ndarray, candidates, k: int) -> List[int]:
    """Select the top-`k` feature indices by entropy (descending), breaking ties by
    the candidate value ascending, returning the indices in that order.

    This is the dedup/selection core: EntropyFPLoader.setup and the vocab-build script
    (scripts/marina_db/build/4_fp_rankingset.py) both need argpartition(-ent, kth=min(k, len-1))[:k]
    then sorted(key=lambda i: (-ent[i], candidates[i])). `candidates` is any sequence
    indexable by the selected indices whose elements sort as the tiebreak (a 4-tuple
    BitInfo, a fragment SMILES, or a (frag, k) tuple). The caller computes `k` itself
    (the "inf" -> len(candidates) logic lives at the call site)."""
    topk_idx = np.argpartition(-entropies, kth=min(k, len(entropies) - 1))[:k]
    return sorted(topk_idx, key=lambda i: (-entropies[i], candidates[i]))


# ---------------------------
# Retrieval counting
# ---------------------------
def count_fragments_over_retrieval(
    retrieval_path: str,
    radius: int,
    num_procs: int = 0,
    feature_kind: str = MORGAN,
) -> Counter:
    """
    Return Counter[Feature] over the retrieval set (presence in #molecules).

    Note this must be a true recount per feature_kind -- substructure counts cannot be
    derived by summing the Morgan counts, since a molecule reaching one fragment via two
    different bit_ids would be counted twice.
    """
    smiles_map = load_smiles_index(retrieval_path)
    smiles_list = list(smiles_map.values())
    procs = (mp.cpu_count() if not num_procs else max(1, int(num_procs)))

    total = Counter()
    if procs == 1:
        # serial fallback (useful for debugging)
        for smi in tqdm(smiles_list, total=len(smiles_list), desc="Counting retrieval fragments"):
            total.update(extract_features(smi, radius, kind=feature_kind))
    else:
        with mp.Pool(processes=procs, initializer=_init_count, initargs=(radius, feature_kind)) as pool:
            for c in tqdm(
                pool.imap_unordered(_worker_count_one, smiles_list, chunksize=64),
                total=len(smiles_list),
                desc="Counting retrieval fragments",
            ):
                total.update(c)
    return total


def write_counts(counter: Counter, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(counter, f)


# ---------------------------
# Rankingset (CSR)
# ---------------------------
def build_rankingset_csr(
    retrieval_path: str,
    bitinfo_to_col: Dict[Feature, int],
    radius: int,
    num_procs: int = 0,
    feature_kind: str = MORGAN,
) -> torch.Tensor:
    """
    Build a torch.sparse_csr_tensor with shape (num_retrieval, num_features)
    where values are 0/1 presence for each retrieval SMILES.

    Rows are L2-normalized to unit length (so cosine similarity = dot product).
    """
    smiles_map = load_smiles_index(retrieval_path)
    ordered = sorted(smiles_map.items(), key=lambda kv: kv[0])  # consistent row order
    num_rows = len(ordered)
    num_cols = max(bitinfo_to_col.values()) + 1 if bitinfo_to_col else 0

    procs = (mp.cpu_count() if not num_procs else max(1, int(num_procs)))
    rows_cols: List[Tuple[int, List[int]]] = []

    if procs == 1:
        _init_csr(radius, bitinfo_to_col, feature_kind)  # workers read these as globals
        for res in tqdm(
            ((i, smi) for i, (_, smi) in enumerate(ordered)),
            total=num_rows,
            desc="Building CSR rows",
        ):
            rows_cols.append(_worker_row_nonzeros(res))
    else:
        with mp.Pool(
            processes=procs,
            initializer=_init_csr,
            initargs=(radius, bitinfo_to_col, feature_kind),
        ) as pool:
            for res in tqdm(
                pool.imap_unordered(
                    _worker_row_nonzeros,
                    [(i, smi) for i, (_, smi) in enumerate(ordered)],
                    chunksize=64,
                ),
                total=num_rows,
                desc="Building CSR rows",
            ):
                rows_cols.append(res)

    # Reassemble in row order
    rows_cols.sort(key=lambda x: x[0])

    crow_indices = [0]
    col_indices: List[int] = []
    values: List[float] = []
    nnz_so_far = 0

    for _, cols in rows_cols:
        if not cols:
            crow_indices.append(nnz_so_far)
            continue
        nnz = len(cols)
        col_indices.extend(cols)
        inv_len = 1.0 / math.sqrt(nnz)
        values.extend([inv_len] * nnz)
        nnz_so_far += nnz
        crow_indices.append(nnz_so_far)

    crow = torch.tensor(crow_indices, dtype=torch.int64)
    cols = torch.tensor(col_indices, dtype=torch.int64)
    vals = torch.tensor(values, dtype=torch.float32)
    return torch.sparse_csr_tensor(crow, cols, vals, size=(num_rows, num_cols))


# ---------------------------
# FragIdx parquet builder
# ---------------------------
def build_fragidx_parquets(
    index_path: str,
    out_dir: str,
    bitinfo_to_col: Dict[Feature, int],
    radius: int,
    num_procs: int = 0,
    feature_kind: str = MORGAN,
    filename: str = "FragIdx.parquet",
) -> None:
    """
    Build DATASET_ROOT/arrow/<split>/<filename> for every split found in index.pkl.

    Schema: (idx: int64, cols: list<int32>) where cols are sorted feature column indices
    mapped through bitinfo_to_col.

    filename is parameterized because the column indices are only meaningful against the
    feature map that produced them -- a substructure fingerprint writing to the default
    FragIdx.parquet would silently corrupt the Morgan one.
    """
    data = _load_as_dict(index_path)

    split_items: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for k, v in data.items():
        idx = int(k)
        if isinstance(v, dict):
            smi = v.get("smiles") or v.get("canonical_2d_smiles")
            split = v.get("split", "train")
        elif isinstance(v, str):
            smi = v
            split = "train"
        else:
            continue
        if smi:
            split_items[split].append((idx, smi))

    procs = mp.cpu_count() if not num_procs else max(1, int(num_procs))

    for split, items in split_items.items():
        results: List[Tuple[int, List[int]]] = []
        if procs == 1:
            _init_csr(radius, bitinfo_to_col, feature_kind)  # workers read these as globals
            for item in tqdm(items, desc=f"Building FragIdx [{split}]"):
                results.append(_worker_row_nonzeros(item))
        else:
            with mp.Pool(
                processes=procs,
                initializer=_init_csr,
                initargs=(radius, bitinfo_to_col, feature_kind),
            ) as pool:
                for res in tqdm(
                    pool.imap_unordered(_worker_row_nonzeros, items, chunksize=64),
                    total=len(items),
                    desc=f"Building FragIdx [{split}]",
                ):
                    results.append(res)

        out_path = os.path.join(out_dir, "arrow", split, filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        table = pa.table({
            "idx": pa.array([r[0] for r in results], type=pa.int64()),
            "cols": pa.array([r[1] for r in results], type=pa.list_(pa.int32())),
        })
        pq.write_table(table, out_path)

