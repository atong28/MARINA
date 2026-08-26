"""
Single source of truth for every path the MARINA-DB build touches.

No stage script should hardcode an absolute path. Roots are resolved once here,
reusing the deployment-aware resolution in `src/modules/core/const.py` (which already
handles the website / yuzu / nautilus / env-var setups and lets DATASET_ROOT and
BENCHMARK_ROOT be overridden from the environment).

Override any of these from the environment before running the build, e.g.:
    export DATASET_ROOT=/workspace
    export BENCHMARK_ROOT=/root/gurusmart/Benchmark
    export MARINA_DATA_ROOT=/home/user/atong          # parent of Datasets/, Benchmark/
    export COCONUT_RELEASE=coconut_csv-08-2026         # pin the COCONUT dump
"""
import os
from pathlib import Path

from src.modules.core.const import CODE_ROOT, DATASET_ROOT, BENCHMARK_ROOT

# Repo root (…/MARINA). CODE_ROOT is set on managed deployments; fall back to walking up.
REPO_ROOT = Path(CODE_ROOT) if CODE_ROOT else Path(__file__).resolve().parents[2]

# Parent directory that holds the sibling data trees (Datasets/, Benchmark/, Snapshots/).
MARINA_DATA_ROOT = Path(os.environ.get("MARINA_DATA_ROOT", REPO_ROOT.parent))

# ---- build working directories (under the repo, as the legacy build_dataset.sh assumed) ----
DATA_RAW = REPO_ROOT / "data" / "raw"          # downloaded dumps + unzipped source archives
DATA_CLEANED = REPO_ROOT / "data" / "cleaned"  # intermediate jsonl / index / retrieval
DATA_DATASET = Path(DATASET_ROOT) if DATASET_ROOT else REPO_ROOT / "data" / "dataset"

# ---- source dumps ----
NPMRD_CSV = DATA_RAW / "npmrd.csv"
COCONUT_CSV = DATA_RAW / "coconut.csv"
LOTUS_TXT = DATA_RAW / "lotus.txt"
COCONUT_RELEASE = os.environ.get("COCONUT_RELEASE", "coconut_csv-08-2026")
# SPECTRE retrieval bank (structure candidates), extracted from SPECTRE inference metadata
# (~/SPECTRECheckpoints/inference_metadata_name_updated.pkl, 526,316 SMILES). Folded into
# the retrieval set (3_build_retrieval.py) so MARINA-DB retrieval is a strict superset of
# SPECTRE's candidate pool.
SPECTRE_RETRIEVAL = DATA_RAW / "spectre_retrieval.pkl"

# ---- intermediates ----
SMILES_DICT = DATA_CLEANED / "smiles_dict.json"
INDEX_PKL = DATA_CLEANED / "index.pkl"
RETRIEVAL_PKL = DATA_CLEANED / "retrieval.pkl"
METADATA_JSON = DATA_CLEANED / "metadata.json"

# ---- benchmarks ----
# The Journal (benchmark-journal.pkl) is the sole benchmark going forward; it is a FROZEN
# curated input (hand-extracted shifts). 10_build_journal.py regenerates only the *prepared*
# view (leakage flags + retrieval_idx) against the current index/splits.
BENCH_ROOT = Path(BENCHMARK_ROOT) if BENCHMARK_ROOT else MARINA_DATA_ROOT / "Benchmark"
BENCH_JOURNAL = BENCH_ROOT / "benchmark-journal.pkl"                 # frozen curated input
BENCH_JOURNAL_PREPARED = BENCH_ROOT / "benchmark-journal-prepared.pkl"
BENCH_FILTERED = BENCH_ROOT / "filtered"        # hand-extracted per-NPID CSVs (from curation/)

# ---- SPECTRE split source (published partition, for alignment) ----
# Derived from the SPECTRE corpus index (data/raw/index.pkl 'split' field) by
# 5_spectre_splits.py; consumed by 7_splits.py and 10_build_journal.py.
SPECTRE_SPLITS_PKL = DATA_CLEANED / "spectre_splits.pkl"

# ---- source priority per modality (single source of truth) ----
# Order stage 6 (6_build_index.py, index has_* flags) and stage 8
# (8_assemble_arrow.py, materialized spectra) resolve each modality's source in.
# Both stages MUST agree so the index flags and materialized spectra never desync.
SOURCE_PRIORITY = {
    "hsqc": ["spectre", "mnova"],
    "c_nmr": ["mnova", "spectre"],
    "h_nmr": ["mnova", "spectre"],
    "mass_spec": ["spectre", "ms"],
}

# ---- fingerprint / dataset build knobs (decisions D6-D9) ----
FP_TYPE = os.environ.get("FP_TYPE", "RankingEntropyMultiplicityUncapped")  # D8 (deployed)
# Fingerprints built by stage 4 (fp_rankingset, vocab + rankingset) and stage 9
# (fp_fragidx, training columns). All four entropy-selected loaders, so the fp-quality
# analysis can compare on the same retrieval set: the deployed uncapped multiplicity (D8),
# the presence-only "sherlock" (RankingEntropy), the capped-k5 multiplicity, and the
# substructure FP. All at FP_RADIUS / FP_OUT_DIM. Override via FP_TYPES="a,b,...".
FP_TYPES = [t for t in os.environ.get(
    "FP_TYPES",
    "RankingEntropyMultiplicityUncapped,RankingEntropy,"
    "RankingEntropyMultiplicity,RankingEntropySubstructure",
).split(",") if t]
FP_OUT_DIM = int(os.environ.get("FP_OUT_DIM", "16384"))
FP_RADIUS = int(os.environ.get("FP_RADIUS", "10"))  # D10: r10 wins both intrinsic axes (wiki/experiments/fp-quality.md)
MW_MAX_EXACT = 1000.0     # D-filters: exact (monoisotopic) mass ceiling
MIN_HEAVY_ATOMS = 3
SPLIT_WEIGHTS = (0.90, 0.05, 0.05)   # D9: balanced against GLOBAL totals
PEAK_COLLAPSE_TOL = 1e-4             # D-per-peak: exact-duplicate 1H/13C merge (ppm)
SEED = 0
