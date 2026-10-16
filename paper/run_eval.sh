#!/bin/bash
# Paper-table evaluation runner (main-paper tables; specs in paper/specs/*.json, README in paper/README.md).
# Retrieval is ranked by cosine (the paper metric, RANK_METRIC=cosine, default) or binary Tanimoto (RANK_METRIC=jaccard,
# kept for comparison); annotation is always ECFP4 cosine >= 0.8.
#
# Work dir W (one GPU box; build it with the "stage" commands in paper/README.md):
#   W/data/MARINA-DB/            release MARINA-DB (disk Datasets/MARINA-DB, zip 49442996...): index, retrieval, arrow/{val,test},
#                                RankingEntropyUniqueMultiplicity/ + RankingEntropy/ (release vocab)
#   W/data/MARINA-DB-OPEN-OLD/   the CH-NMR-NP-first build the 2026-10-02 tables came from (old vocab; old checkpoints only)
#   W/data/MARINA-DB-PRIVATE/    original dataset (disk Datasets/MARINA-DB-PRIVATE): index, retrieval, arrow/test, the five
#                                RankingEntropy*/ dirs (fp sweep + training regimes)
#   W/data/SPECTRE-clean-{test,val}/  SPECTRE retrieval + RankingEntropy/ bank with the clean-subset structures appended
#   W/bench/{full,clean_test,clean_val,simnmr}/benchmark-journal.pkl
#        full = MARINA-Bench (experimental NMR + simulated MS/MS = Benchmark/benchmark-sim.pkl; 456 = 227 val / 229 test
#               since 2026-10-15, Bench-466 = 232 val / 234 test before, archived in Benchmark/archive/pre-nist23-drop-2026-10-14/)
#        clean_test = SPECTRE-clean test 205, clean_val = SPECTRE-clean val 206 (207 before the NIST23 drop)
#        simnmr = the same benchmark compounds with Mnova-simulated NMR (Benchmark/benchmark-sim-nmr-466.pkl; holds 456, legacy filename)
#   W/ckpt/<experiment>/<ts>/{epoch_*.ckpt,params.json}   (exactly one ckpt per experiment)
#   W/ckpt/spectre-deployed/{best.ckpt,params.json}
# Outputs: W/results-<metric><RES_TAG>/<bench>/benchmarks/<name>_benchmark_journal_results.pkl and <name>_sim_results.json.
# Env: FLAG_DATA (data dir of flagship/simgap/simfull/singleatom, default MARINA-DB = the release), FPSWEEP_DATA (fpsweep,
#      default MARINA-DB-PRIVATE; MARINA-DB for the release Sherlock arms), RES_TAG (suffix of W/results-<metric> and of
#      paper/results/raw-<metric>; default empty = the paper package: release MARINA-DB + MARINA-Bench 456). Reproducing the
#      2026-10-02 package: RES_TAG=-466 FLAG_DATA=MARINA-DB-OPEN-OLD with the Bench-466 journals in W/bench, so its outputs
#      never mix with the paper's and skip-if-exists cannot reuse a file from the other package.
#
# Usage:  W=<work dir> bash paper/run_eval.sh <group> [experiment ...]
#   flagship <exp...>  Tables results_main + spectre_comparison (MARINA side): full, clean_test, clean_val journals
#   spectre            Table spectre_comparison (SPECTRE side): clean_test + clean_val
#   fpsweep  <exp...>  Table fp_comparison: full journal on $FPSWEEP_DATA (default MARINA-DB-PRIVATE)
#   regime   <exp...>  Table results_training_regime: full journal + simulated MARINA-DB-PRIVATE test (4 NMR combos)
#   simgap   <exp...>  Table sim_exp_gap (appendix): simulated MARINA-DB test (4 NMR combos)
#                      + the benchmark compounds with Mnova-simulated NMR (bench simnmr)
#   simfull  <exp...>  Tables S1/S1b (appendix): simulated MARINA-DB test over all 24 {spectra,+F,+MW} combos
#                      (resumes a simgap run: combos already scored are kept)
#   singleatom <exp...> single_atom_formula tables (appendix): single-atom multiplicity bits, journal val+test
#   collect            copy W/results-<metric><RES_TAG>/*/benchmarks/* into paper/results/raw-<metric><RES_TAG>/<bench>/ (then commit)
# Each step is skipped when its output exists, so reruns after an interruption are safe.
# Results of the two metrics never mix: W/results-cosine<RES_TAG> <-> paper/results/raw-cosine<RES_TAG>,
# W/results-jaccard<RES_TAG> <-> paper/results/raw-jaccard<RES_TAG>.
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
RES_TAG=${RES_TAG:-}
FLAG_DATA=${FLAG_DATA:-MARINA-DB}
FPSWEEP_DATA=${FPSWEEP_DATA:-MARINA-DB-PRIVATE}
case "$RANK_METRIC" in cosine|jaccard) RES=results-$RANK_METRIC$RES_TAG; RAW=raw-$RANK_METRIC$RES_TAG ;;
    *) echo "RANK_METRIC must be jaccard or cosine" >&2; exit 2 ;; esac
EVAL="pixi run python scripts/benchmark/eval_journal_benchmark.py --rank_metric $RANK_METRIC --deltas"
echo "[run-eval] code=$(git rev-parse --short HEAD) group=$GROUP rank_metric=$RANK_METRIC res=$RES flag_data=$FLAG_DATA fpsweep_data=$FPSWEEP_DATA exps=${EXPS[*]} log=$LOG" | tee -a "$LOG"

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
            journal "$b" "$FLAG_DATA" "$e" --results_root "$W/ckpt" --experiments "$e"
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
        journal full "$FPSWEEP_DATA" "$e" --results_root "$W/ckpt" --experiments "$e"
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
simgap)
    for e in "${EXPS[@]}"; do
        journal simnmr "$FLAG_DATA" "$e" --results_root "$W/ckpt" --experiments "$e"
        root=$(bench_root full)
        if [ -s "$root/benchmarks/${e}_sim_results.json" ]; then
            echo "[run-eval] have sim $e" | tee -a "$LOG"
        else
            DATASET_ROOT=$W/data/$FLAG_DATA BENCHMARK_ROOT=$root run "$e sim test" \
                $EVAL --no-journal --sim --sim_splits test --sim_only hsqc_c_nmr_h_nmr hsqc c_nmr h_nmr \
                --results_root "$W/ckpt" --experiments "$e"
        fi
    done ;;
simfull)
    ALL="hsqc_c_nmr_h_nmr_mass_spec_mass_spec_neg hsqc_c_nmr_h_nmr hsqc c_nmr h_nmr hsqc_c_nmr hsqc_h_nmr c_nmr_h_nmr"
    for e in "${EXPS[@]}"; do
        root=$(bench_root full); out="$root/benchmarks/${e}_sim_results.json"
        if [ -s "$out" ] && python3 -c "import json,sys; m=json.load(open(sys.argv[1]))['metrics']; sys.exit(0 if all(f'test/mean_ann_1/{c}{s}' in m for c in sys.argv[2].split() for s in ('','_formula','_mw')) else 1)" "$out" "$ALL"; then
            echo "[run-eval] have full sim $e" | tee -a "$LOG"
        else
            DATASET_ROOT=$W/data/$FLAG_DATA BENCHMARK_ROOT=$root run "$e sim test (24 combos)" \
                $EVAL --no-journal --sim --sim_splits test --results_root "$W/ckpt" --experiments "$e"
        fi
    done ;;
singleatom)
    mkdir -p "$W/$RES/singleatom"
    for e in "${EXPS[@]}"; do
        out="$W/$RES/singleatom/${e}.json"
        [ -s "$out" ] && { echo "[run-eval] have $out" | tee -a "$LOG"; continue; }
        ck=$(ls "$W"/ckpt/"$e"/*/epoch_*.ckpt)
        DATASET_ROOT=$W/data/$FLAG_DATA BENCHMARK_ROOT=$W/bench/full run "$e single-atom bits" \
            pixi run python scripts/benchmark/single_atom_bit_eval.py --ckpt "$ck" --splits val test --out "$out"
    done ;;
collect)
    if [ -d "$W/$RES/singleatom" ]; then
        mkdir -p "paper/results/$RAW/singleatom" && cp -v "$W/$RES"/singleatom/*.json "paper/results/$RAW/singleatom/" | tee -a "$LOG"
    fi
    for r in "$W/$RES"/*/; do
        b=$(basename "$r"); [ "$b" = singleatom ] && continue
        mkdir -p "paper/results/$RAW/$b"
        cp -v "$r"benchmarks/*_benchmark_journal_results.pkl "$r"benchmarks/*_sim_results.json "paper/results/$RAW/$b/" 2>/dev/null \
            | tee -a "$LOG"
    done ;;
*)
    echo "unknown group $GROUP" >&2; exit 2 ;;
esac
echo "[run-eval] ALL DONE $(date)" | tee -a "$LOG"
