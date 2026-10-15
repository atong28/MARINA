#!/bin/bash
# Launch the MARINA-DB flagship arms (2026-10-14 MARINA-DB: NIST23 MS/MS blacklist, current ICEBERG MS/MS,
# vocab recomputed) on NCSA DeltaAI: unique multiplicity + Sherlock (RankingEntropy), seeds 0-2.
# Run ON the login node (e.g. `ssh ncsa-ai COMMIT=<sha> bash -s < singularity/deltaai-launch-flagship.sh`).
#
# Code: git clone of MARINA at the pinned COMMIT in code/MARINA-flagship (on /projects, so Delta can run
# the same tree). Dataset: datasets/MARINA-DB.zip (flat; carries every FP family's vocab, rankingset,
# FragIdx and packed/, so fp-artifacts/MARINA-DB must NOT exist -- the sbatch would overlay it).
# Recipe: 1xGH200, bs512 accum1 (global 512), bf16, 7 inputs, bernoulli, 750 ep / patience 30.
# ARMS="uniqmult:0 sherlock:2" submits a subset. MAX_CHUNKS=2: chunk 2 only if chunk 1 hits the 48h wall.
set -euo pipefail
: "${COMMIT:?set COMMIT=<pinned MARINA sha>}"
PROJECT_ROOT=/projects/bibx/atong1
CODE=$PROJECT_ROOT/code/MARINA-flagship
DATASET=MARINA-DB
ARMS="${ARMS:-uniqmult:0 uniqmult:1 uniqmult:2 sherlock:0 sherlock:1 sherlock:2}"
TS=$(date +%Y%m%d-%H%M)

if [ -e "$PROJECT_ROOT/fp-artifacts/$DATASET" ]; then
    echo "[launch] refusing: fp-artifacts/$DATASET exists and would be overlaid onto the dataset" >&2; exit 1
fi
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
[ -d "$CODE/.git" ] || git clone --quiet git@github.com:atong28/MARINA.git "$CODE"
git -C "$CODE" fetch --quiet origin
git -C "$CODE" checkout --quiet --detach "$COMMIT"
echo "[launch] code=$(git -C "$CODE" rev-parse --short HEAD) dataset=$(ls -la $PROJECT_ROOT/datasets/$DATASET.zip)"

# import smoke test in the arm64 image before spending GPU hours
apptainer exec --cleanenv --pwd /code -B "$CODE":/code --env DATASET_ROOT=/tmp \
    $PROJECT_ROOT/images/marina-arm64.sif python -c "import src.main, src.modules.train" \
    && echo "[launch] import smoke OK"

for arm in $ARMS; do
    fp=${arm%%:*}; s=${arm##*:}
    case $fp in
        uniqmult) FP_TYPE=RankingEntropyUniqueMultiplicity ;;
        sherlock) FP_TYPE=RankingEntropy ;;
        *) echo "[launch] unknown fp '$fp'" >&2; exit 1 ;;
    esac
    jid=$(EXPERIMENT_NAME=marina-db-flagship-$fp-s$s DATASET=$DATASET FP_TYPE=$FP_TYPE SEED=$s MAX_CHUNKS=2 \
        CODE_DIR="$CODE" CHAIN_SCRIPT="$CODE/singularity/deltaai-train.sbatch" \
        sbatch --parsable --job-name=mdbflag-$fp-s$s-$TS "$CODE/singularity/deltaai-train.sbatch")
    echo "[launch] $fp s$s -> job $jid (mdbflag-$fp-s$s-$TS)"
done
