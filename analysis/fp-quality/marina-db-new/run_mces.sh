#!/bin/bash
# Exp 1 RASCAL MCES over the NEW MARINA-DB (CH-NMR-NP-first, N=531,927) retrieval set.
# Original recipe: n-pool 5000, n-pairs 100000, seed 0, timeout 60; 20 sequential shards for durability.
# No --fp here: FP Tanimoto columns are added afterwards with exp1_add_fps.py.
set -u
cd /home/atong/Workspace
OUT=MARINA/analysis/fp-quality/marina-db-new/results/mces_shards
mkdir -p $OUT
W=${W:-12}
for s in $(seq 0 19); do
  f=$OUT/shard$(printf %02d $s).parquet
  [ -f $f ] && continue
  nice -n 5 pixi run python MARINA/analysis/fp-quality/scripts/exp1_mces.py \
    --retrieval Datasets/MARINA-DB-OPEN/retrieval.pkl --n-pool 5000 --n-pairs 100000 \
    --workers $W --seed 0 --timeout 60 --num-shards 20 --shard-id $s --out $f
done
echo ALL_SHARDS_DONE
