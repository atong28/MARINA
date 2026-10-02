#!/bin/bash
# Second MCES runner started once the reference builds freed their cores: takes shards from the
# END (19 downward) with W workers and stops at the first shard the main runner (run_mces.sh,
# ascending) has already started or finished. Same arguments -> same pair slices.
set -u
cd /home/atong/Workspace
OUT=MARINA/analysis/fp-quality/marina-db-new/results/mces_shards
LOG=MARINA/analysis/fp-quality/marina-db-new/logs/mces.log
W=${W:-3}
for s in $(seq 19 -1 0); do
  f=$OUT/shard$(printf %02d $s).parquet
  if [ -f $f ] || grep -q "^shard $s/20" $LOG; then break; fi
  echo "tail runner: shard $s"
  nice -n 5 pixi run python MARINA/analysis/fp-quality/scripts/exp1_mces.py \
    --retrieval Datasets/MARINA-DB-OPEN/retrieval.pkl --n-pool 5000 --n-pairs 100000 \
    --workers $W --seed 0 --timeout 60 --num-shards 20 --shard-id $s --out $f.tmp.parquet \
    && mv $f.tmp.parquet $f
done
echo TAIL_DONE
