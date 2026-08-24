#!/usr/bin/env bash
#
# MARINA-DB — end-to-end build driver. One source of truth for the run order.
#
# Usage:
#   scripts/marina_db/run_build.sh                 # full build, all stages
#   scripts/marina_db/run_build.sh --from 30       # resume from a stage id
#   scripts/marina_db/run_build.sh --only 60       # run a single stage
#   SKIP_DOWNLOAD=1 scripts/marina_db/run_build.sh # reuse existing data/raw
#
# Paths and knobs come from scripts/marina_db/config.py (override via env:
# DATASET_ROOT, BENCHMARK_ROOT, MARINA_DATA_ROOT, COCONUT_RELEASE, FP_TYPE...).
# Decisions encoded here: D6 tie-aware rank (eval), D7 full-467 journal exclusion +
# D9 global-90/5/5 balance (stage 30), per-peak collapse dataset-wide (stage 50),
# D8 uncapped-multiplicity fingerprint (stage 60). See README.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts/marina_db:${PYTHONPATH:-}"
MDB="scripts/marina_db"
PY="${MARINA_PY:-pixi run python}"

# Ordered stage table: "id:command". ids are stable handles for --from/--only.
STAGES=(
  "00:bash $MDB/build/00_download.sh"
  "10:$PY $MDB/build/10_smiles_merge.py"
  "20:$PY $MDB/build/20_build_index.py"
  "30:$PY $MDB/build/30_splits.py"
  "40:$PY $MDB/build/40_assemble_arrow.py"
  "45a:$PY $MDB/benchmarks/build_annotated.py"
  "45b:$PY $MDB/benchmarks/build_journal.py"
  "45c:$PY $MDB/benchmarks/build_simulated.py"
  "50:$PY $MDB/build/50_collapse_peaks.py --target all"
  "60:$PY $MDB/build/60_fp_vocab.py"
  "70:$PY $MDB/build/70_retrieval_augment.py"
  "80:$PY $MDB/build/80_verify.py"
)

FROM=""; ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
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
  if [[ "$id" == "00" && "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
    echo "== [skip] stage 00 download (SKIP_DOWNLOAD=1)"; continue
  fi
  echo "== [stage $id] $cmd"
  eval "$cmd"
done
echo "== MARINA-DB build: done (stages $( [[ -n "$ONLY" ]] && echo "$ONLY" || echo "${FROM:-00}..80" ))"
