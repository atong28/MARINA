#!/bin/bash
# Paper-table evaluation runner (main-paper tables; specs in paper/specs/*.json, README in paper/README.md).
# Retrieval is ranked by cosine (the paper metric, RANK_METRIC=cosine, default) or binary Tanimoto (RANK_METRIC=jaccard,
# kept for comparison); annotation is always ECFP4 cosine >= 0.8.
#
# Work dir W (one GPU box; build it with the "stage" commands in paper/README.md):
#   W/data/MARINA-DB/            CH-NMR-NP-first dataset (disk name MARINA-DB-OPEN): index, retrieval, arrow/{val,test},
#                                RankingEntropyUniqueMultiplicity/
#   W/data/MARINA-DB-PRIVATE/    original dataset (disk name MARINA-DB): index, retrieval, arrow/test, the five
#                                RankingEntropy*/ dirs (fp sweep + training regimes)
#   W/data/SPECTRE-clean-{test,val}/  SPECTRE retrieval + RankingEntropy/ bank with the clean-subset structures appended
#   W/bench/{full,clean_test,clean_val}/benchmark-journal.pkl
#        full = MARINA-Bench 466 (232 val / 234 test; experimental NMR + simulated MS/MS = Benchmark/benchmark-sim.pkl)
#        clean_test = SPECTRE-clean test 205, clean_val = SPECTRE-clean val 207
#   W/ckpt/<experiment>/<ts>/{epoch_*.ckpt,params.json}   (exactly one ckpt per experiment)
#   W/ckpt/spectre-deployed/{best.ckpt,params.json}
# Outputs: W/results-<metric>/<bench>/benchmarks/<name>_benchmark_journal_results.pkl and <name>_sim_results.json.
#
# Usage:  W=<work dir> bash paper/run_eval.sh <group> [experiment ...]
#   flagship <exp...>  Tables results_main + spectre_comparison (MARINA side): full, clean_test, clean_val journals
#   spectre            Table spectre_comparison (SPECTRE side): clean_test + clean_val
#   fpsweep  <exp...>  Table fp_comparison: full journal on MARINA-DB-PRIVATE
#   regime   <exp...>  Table results_training_regime: full journal + simulated MARINA-DB-PRIVATE test (4 NMR combos)
#   collect            copy W/results-<metric>/*/benchmarks/* into paper/results/raw-<metric>/<bench>/ (then commit)
# Each step is skipped when its output exists, so reruns after an interruption are safe.
# Results of the two metrics never mix: W/results-cosine <-> paper/results/raw-cosine,
# W/results-jaccard <-> paper/results/raw-jaccard.
set -uo pipefail
: "${W:?set W=<work dir>}"
GROUP=$1; shift
EXPS=("$@")
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH=$REPO
mkdir -p "$W/logs"
LOG="$W/logs/run_eval_${GROUP}_$(date +%Y%m%d-%H%M%S).log"
FILT='it/s\]|s/it\]|UserWarning|warnings.warn'
RANK_METRIC=${RANK_METRIC:-cosine}
case "$RANK_METRIC" in cosine|jaccard) RES=results-$RANK_METRIC; RAW=raw-$RANK_METRIC ;;
    *) echo "RANK_METRIC must be jaccard or cosine" >&2; exit 2 ;; esac
EVAL="pixi run python scripts/benchmark/eval_journal_benchmark.py --rank_metric $RANK_METRIC --deltas"
echo "[run-eval] code=$(git rev-parse --short HEAD) group=$GROUP rank_metric=$RANK_METRIC exps=${EXPS[*]} log=$LOG" | tee -a "$LOG"

run() {  # run <label> <command...>; logs, never aborts the whole runner
    echo "[run-eval] ===== $(date) $1" | tee -a "$LOG"
    shift
    "$@" 2>&1 | grep --line-buffered -vE "$FILT" | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" = 0 ] || echo "[run-eval] FAILED (continuing)" | tee -a "$LOG"
}

bench_root() {  # bench_root <bench> -> W/$RES/<bench> (journal pkl linked in, outputs beside it)
    local r="$W/$RES/$1"
    mkdir -p "$r/benchmarks"
    ln -sfn "$W/bench/$1/benchmark-journal.pkl" "$r/benchmark-journal.pkl"
    echo "$r"
}

journal() {  # journal <bench> <data dir> <name> <eval args...>
    local bench=$1 data=$2 name=$3; shift 3
    local root; root=$(bench_root "$bench")
    local out="$root/benchmarks/${name}_benchmark_journal_results.pkl"
    [ -s "$out" ] && { echo "[run-eval] have $out" | tee -a "$LOG"; return; }
    DATASET_ROOT=$W/data/$data BENCHMARK_ROOT=$root run "$name journal $bench" $EVAL "$@"
}

case "$GROUP" in
flagship)
    for e in "${EXPS[@]}"; do
        for b in full clean_test clean_val; do
            journal "$b" MARINA-DB "$e" --results_root "$W/ckpt" --experiments "$e"
        done
    done ;;
spectre)
    # the released SPECTRE params.json has no input_types; its 4-row type embedding is HSQC / 13C / 1H / MW
    S=(--project_name SPECTRE --fp_type RankingEntropy --legacy_spectre --input_types hsqc c_nmr h_nmr mw
       --ckpt "$W/ckpt/spectre-deployed/best.ckpt" --params "$W/ckpt/spectre-deployed/params.json"
       --name spectre-deployed)
    journal clean_test SPECTRE-clean-test spectre-deployed "${S[@]}"
    journal clean_val SPECTRE-clean-val spectre-deployed "${S[@]}" ;;
fpsweep)
    for e in "${EXPS[@]}"; do
        journal full MARINA-DB-PRIVATE "$e" --results_root "$W/ckpt" --experiments "$e"
    done ;;
regime)
    for e in "${EXPS[@]}"; do
        journal full MARINA-DB-PRIVATE "$e" --results_root "$W/ckpt" --experiments "$e"
        root=$(bench_root full)
        if [ -s "$root/benchmarks/${e}_sim_results.json" ] && \
           python3 -c "import json,sys; m=json.load(open(sys.argv[1]))['metrics']; sys.exit(0 if all(f'test/mean_tani/{c}' in m for c in ['hsqc_c_nmr_h_nmr','hsqc','c_nmr','h_nmr']) else 1)" \
               "$root/benchmarks/${e}_sim_results.json"; then
            echo "[run-eval] have sim $e" | tee -a "$LOG"
        else
            DATASET_ROOT=$W/data/MARINA-DB-PRIVATE BENCHMARK_ROOT=$root run "$e sim test" \
                $EVAL --no-journal --sim --sim_splits test --sim_only hsqc_c_nmr_h_nmr hsqc c_nmr h_nmr \
                --results_root "$W/ckpt" --experiments "$e"
        fi
    done ;;
collect)
    for r in "$W/$RES"/*/; do
        b=$(basename "$r")
        mkdir -p "paper/results/$RAW/$b"
        cp -v "$r"benchmarks/*_benchmark_journal_results.pkl "$r"benchmarks/*_sim_results.json "paper/results/$RAW/$b/" 2>/dev/null \
            | tee -a "$LOG"
    done ;;
*)
    echo "unknown group $GROUP" >&2; exit 2 ;;
esac
echo "[run-eval] ALL DONE $(date)" | tee -a "$LOG"
