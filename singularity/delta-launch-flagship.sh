#!/bin/bash
# Fallback for MARINA-DB flagship arms that will not schedule on DeltaAI: submit them to the A100 Delta.
# Run ON a Delta login node (e.g. `ssh ncsa ARMS="sherlock:2" bash -s < singularity/delta-launch-flagship.sh`)
# after scancel-ing the arm's pending DeltaAI job (a pending chunk has not queued its successor yet).
#
# Same code tree as DeltaAI (code/MARINA-flagship on the shared /projects, already at the pinned COMMIT; this
# script refuses to run if HEAD differs; CODE=<dir> overrides it for arms pinned to another commit) and the same
# dataset zip. Arm `hsqcshifts` = the uniqmult flagship with a shifts-only HSQC encoder (multiplicity-edit sign
# column dropped: --nmr_dim_coords 392 392 --nmr_is_sign_encoding False False; needs MARINA >= the commit adding it). Recipe: 2xA100, bs256 x 2 ranks x accum1
# = global 512, bf16-mixed, 7 inputs, bernoulli, 750 ep / patience 30; amd64 image marina.sif.
# MAX_CHUNKS=3: ~2x the GH200 epoch time, so a run can need a second 48h chunk.
set -euo pipefail
: "${COMMIT:?set COMMIT=<pinned MARINA sha>}"
: "${ARMS:?set ARMS, e.g. ARMS=\"uniqmult:1 sherlock:2\"}"
PROJECT_ROOT=/projects/bibx/atong1
CODE=${CODE:-$PROJECT_ROOT/code/MARINA-flagship}
TS=$(date +%Y%m%d-%H%M)
[ "$(git -C "$CODE" rev-parse HEAD)" = "$(git -C "$CODE" rev-parse "$COMMIT^{commit}")" ] \
    || { echo "[launch] $CODE is not at $COMMIT" >&2; exit 1; }

for arm in $ARMS; do
    fp=${arm%%:*}; s=${arm##*:}
    EXTRA=""
    case $fp in
        uniqmult) FP_TYPE=RankingEntropyUniqueMultiplicity ;;
        sherlock) FP_TYPE=RankingEntropy ;;
        hsqcshifts) FP_TYPE=RankingEntropyUniqueMultiplicity; EXTRA="--nmr_dim_coords 392 392 --nmr_is_sign_encoding False False" ;;
        *) echo "[launch] unknown fp '$fp'" >&2; exit 1 ;;
    esac
    jid=$(EXPERIMENT_NAME=marina-db-flagship-$fp-s$s DATASET=MARINA-DB MAX_CHUNKS=3 \
        GPUS_PER_NODE=2 CPUS_PER_TASK=32 MEM=64G CODE_DIR="$CODE" CHAIN_SCRIPT="$CODE/singularity/delta-train.sbatch" \
        TRAIN_ARGS="--fp_type $FP_TYPE --seed $s --batch_size 256 --accumulate_grad_batches_num 1 --precision bf16-mixed --input_types hsqc c_nmr h_nmr mass_spec mass_spec_neg mw formula $EXTRA" \
        sbatch --parsable --gpus-per-node=2 --cpus-per-task=32 --mem=64G --job-name=mdbflag-$fp-s$s-$TS \
        "$CODE/singularity/delta-train.sbatch")
    echo "[launch] $fp s$s -> Delta job $jid (mdbflag-$fp-s$s-$TS)"
done
