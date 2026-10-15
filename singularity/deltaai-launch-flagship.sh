#!/bin/bash
# Launch the 3-seed MARINA-DB-OPEN (CH-NMR-NP-first NMR) arms on NCSA DeltaAI.
# Run ON the login node (e.g. `ssh ncsa-ai COMMIT=<sha> bash -s < singularity/deltaai-launch-open.sh`).
#
# Code: git clone of MARINA at the pinned COMMIT in code/MARINA-open-chnmr (separate from the
# Mnova-only OPEN arms' code/MARINA-open @ f673c3c). Dataset: datasets/MARINA-DB-OPEN.zip (flat,
# carries the uniqmult vocab/rankingset + FragIdx, so no fp-artifacts overlay).
# Recipe = the Mnova-only OPEN arms: 1xGH200, bs512 accum1, bf16, 7 inputs, defaults otherwise.
# MAX_CHUNKS=2: chunk 2 only resumes if chunk 1 hits the 48h wall; else it sees COMPLETE and exits.
set -euo pipefail
: "${COMMIT:?set COMMIT=<pinned MARINA sha>}"
PROJECT_ROOT=/projects/bibx/atong1
CODE=$PROJECT_ROOT/code/MARINA-open-chnmr
DATASET=MARINA-DB-OPEN
TS=$(date +%Y%m%d-%H%M)

export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
[ -d "$CODE/.git" ] || git clone --quiet git@github.com:atong28/MARINA.git "$CODE"
git -C "$CODE" fetch --quiet origin
git -C "$CODE" checkout --quiet --detach "$COMMIT"
echo "[launch] code=$(git -C "$CODE" rev-parse --short HEAD) dataset=$(ls -la $PROJECT_ROOT/datasets/$DATASET.zip)"

# import smoke test in the arm64 image before spending GPU hours
apptainer exec --cleanenv --pwd /code -B "$CODE":/code --env DATASET_ROOT=/tmp \
    $PROJECT_ROOT/images/marina-arm64.sif python -c "import src.main, src.modules.train" \
    && echo "[launch] import smoke OK"

for s in 0 1 2; do
    jid=$(EXPERIMENT_NAME=marina-db-open-chnmr-uniqmult-formula-s$s DATASET=$DATASET \
        FP_TYPE=RankingEntropyUniqueMultiplicity SEED=$s MAX_CHUNKS=2 \
        CODE_DIR="$CODE" CHAIN_SCRIPT="$CODE/singularity/deltaai-train.sbatch" \
        sbatch --parsable --job-name=mdbopen-chnmr-s$s-$TS "$CODE/singularity/deltaai-train.sbatch")
    echo "[launch] s$s -> job $jid (mdbopen-chnmr-s$s-$TS)"
done
