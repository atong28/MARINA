#!/usr/bin/env bash
#
# MARINA-DB — end-to-end build driver. One source of truth for the run order.
#
# Usage:
#   scripts/marina_db/run_build.sh                 # full build, all stages
#   scripts/marina_db/run_build.sh --from 5        # resume from a stage id
#   scripts/marina_db/run_build.sh --only 4        # run a single stage
#   scripts/marina_db/run_build.sh --from 1 --to 4 # structure track only (no MS/MS needed)
#   SKIP_DOWNLOAD=1 scripts/marina_db/run_build.sh # reuse existing data/raw
#
# Two tracks: stages 1-4 are the STRUCTURE track (retrieval + fingerprint rankingsets),
# which need no spectral data and are NOT blocked on the positive MS/MS re-prediction.
# Stages 5-12 are the SPECTRAL/training track (index, splits, arrow, fragidx, journal,
# collapse, verify); MS/MS is optional there until the dataset is finalized.
#
# Paths and knobs come from scripts/marina_db/config.py (override via env:
# DATASET_ROOT, BENCHMARK_ROOT, MARINA_DATA_ROOT, COCONUT_RELEASE, FP_TYPE, FP_TYPES...).
# Decisions encoded here: D6 tie-aware rank (eval), D7 full-467 journal exclusion +
# D9 global-90/5/5 balance (stage 7), per-peak collapse dataset-wide (stage 11),
# D8 uncapped-multiplicity fingerprint (stage 4). See README.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts/marina_db:${PYTHONPATH:-}"
MDB="scripts/marina_db"
PY="${MARINA_PY:-pixi run python}"

# Ordered stage table: "id:command". ids are stable handles for --from/--only/--to.
STAGES=(
  "1:bash $MDB/build/1_download.sh"
  "2:$PY $MDB/build/2_smiles_merge.py"
  "3:$PY $MDB/build/3_build_retrieval.py"
  "4:$PY $MDB/build/4_fp_rankingset.py"
  "5:$PY $MDB/build/5_spectre_splits.py"
  "6:$PY $MDB/build/6_build_index.py"
  "7:$PY $MDB/build/7_splits.py"
  "8:$PY $MDB/build/8_assemble_arrow.py"
  "9:$PY $MDB/build/9_fp_fragidx.py"
  "10:$PY $MDB/build/10_build_journal.py"
  "11:$PY $MDB/build/11_collapse_peaks.py --target all"
  "12:$PY $MDB/build/12_verify.py"
)

FROM=""; ONLY=""; TO=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --to)   TO="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

started=0
for entry in "${STAGES[@]}"; do
  id="${entry%%:*}"; cmd="${entry#*:}"
  if [[ -n "$ONLY" ]]; then [[ "$id" == "$ONLY" ]] || continue; fi
  if [[ -n "$FROM" && $started -eq 0 ]]; then
    [[ "$id" == "$FROM" ]] && started=1 || continue
  fi
  if [[ "$id" == "1" && "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
    echo "== [skip] stage 1 download (SKIP_DOWNLOAD=1)"; continue
  fi
  echo "== [stage $id] $cmd"
  eval "$cmd"
  if [[ -n "$TO" && "$id" == "$TO" ]]; then break; fi
done
echo "== MARINA-DB build: done (stages $( [[ -n "$ONLY" ]] && echo "$ONLY" || echo "${FROM:-1}..${TO:-12}" ))"
