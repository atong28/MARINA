#!/bin/bash
# Paper eval driver for the MARINA-DB (CH-NMR-NP-first, ex-"OPEN") rewrite. Runs on one GPU box
# (grapefruit) against a staged work dir W:
#   W/data/MARINA-DB/                      the CH-NMR-NP-first dataset (arrow/{val,test}, index, retrieval,
#                                          RankingEntropy*/, nmr_sources.parquet)
#   W/bench/{full,clean_test,clean_val,simnmr}/benchmark-journal.pkl
#        full       = journal 466 (232 val / 234 test), real NMR + simulated MS/MS
#        clean_test = SPECTRE-clean test 205 (Table 3), clean_val = SPECTRE-clean val 207
#        simnmr     = the same 466 with Mnova-simulated NMR (sim-exp gap table)
#   W/ckpt/<experiment>/<ts>/{epoch_*.ckpt,params.json}
# Outputs land in W/bench/<bench>/benchmarks/ and W/out/.
#
# Usage:  W=<work dir> bash scripts/benchmark/paper_eval.sh <mode> <experiment> [<experiment> ...]
#   mode = flagship : per experiment: full journal (--deltas) + sim test & val (--deltas, per-item records)
#                     + clean_test + clean_val + simnmr journals + single-atom bits; then, given exactly
#                     3 experiments, the 3-seed consensus on the full journal (val + test)
#          journal  : full journal only (--deltas), e.g. the PRIVATE flagship against this retrieval set
# Every step is resumable / idempotent per output file, so rerunning after an interruption is safe.
set -uo pipefail
: "${W:?set W=<work dir>}"
MODE=$1; shift
EXPS=("$@")
cd "$(dirname "$0")/../.."
export DATASET_ROOT=$W/data/MARINA-DB PYTHONPATH=$PWD
mkdir -p "$W/out" "$W/logs"
LOG="$W/logs/paper_eval_$(date +%Y%m%d-%H%M%S).log"
echo "[paper-eval] code=$(git rev-parse --short HEAD) mode=$MODE exps=${EXPS[*]} log=$LOG" | tee -a "$LOG"
FILT='it/s\]|s/it\]|UserWarning|warnings.warn'

run() {  # run <label> <command...>; logs, never aborts the whole driver
    echo "[paper-eval] ===== $(date) $1" | tee -a "$LOG"
    shift
    "$@" 2>&1 | grep --line-buffered -vE "$FILT" | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" = 0 ] || echo "[paper-eval] FAILED (continuing)" | tee -a "$LOG"
}

journal() {  # journal <bench> <exp>
    local out="$W/bench/$1/benchmarks/$2_benchmark_journal_results.pkl"
    [ -s "$out" ] && { echo "[paper-eval] have $out" | tee -a "$LOG"; return; }
    BENCHMARK_ROOT=$W/bench/$1 run "$2 journal $1" \
        pixi run python scripts/benchmark/eval_journal_benchmark.py --deltas \
        --results_root "$W/ckpt" --experiments "$2"
}

for e in "${EXPS[@]}"; do
    journal full "$e"
    [ "$MODE" = flagship ] || continue
    # sim test + val: resumable per combo inside the driver
    BENCHMARK_ROOT=$W/bench/full run "$e sim test+val" \
        pixi run python scripts/benchmark/eval_journal_benchmark.py --no-journal --sim --deltas \
        --sim_splits test val --results_root "$W/ckpt" --experiments "$e"
    journal clean_test "$e"
    journal clean_val "$e"
    journal simnmr "$e"
    sa="$W/out/single_atom_bits_$e.json"
    if [ ! -s "$sa" ]; then
        ck=$(ls "$W"/ckpt/"$e"/*/*.ckpt)
        BENCHMARK_ROOT=$W/bench/full run "$e single-atom bits" \
            pixi run python scripts/benchmark/single_atom_bit_eval.py --ckpt "$ck" --splits val test --out "$sa"
    fi
done

if [ "$MODE" = flagship ] && [ "${#EXPS[@]}" = 3 ]; then
    co="$W/out/consensus_${EXPS[0]%-s0}_full.json"
    [ -s "$co" ] || BENCHMARK_ROOT=$W/bench/full run "consensus ${EXPS[*]}" \
        pixi run python -m scripts.benchmark.eval_consensus_journal --deltas \
        --results_root "$W/ckpt" --experiments "${EXPS[@]}" --out "$co"
fi
echo "[paper-eval] ALL DONE $(date)" | tee -a "$LOG"
