#!/bin/bash
# Build the reference FP rankingsets over the NEW MARINA-DB retrieval (N=531,927) with the repo's
# builders (same flags as the published rows), then per FP: Exp 2 (+ prefix check) and FP-Tanimoto
# on the Exp 1 pair index. Large rankingsets (Atom Pair, MAP4: 2.5-6 GB each) are deleted after use
# because the workstation disk is ~98% full; they are deterministic and rebuildable with this script.
set -eu
cd /home/atong/Workspace
H=MARINA/analysis/fp-quality
N=$H/marina-db-new
RET=Datasets/MARINA-DB-OPEN/retrieval.pkl
RS=$N/rankingsets
W=${W:-3}
PAIRS=$N/results/pairs_fp_tanimoto.parquet
export PYTHONPATH=/tmp/fpq_vendor   # biosynfoni==1.0.0 (only PyPI release), unpacked wheel, no deps
PY="nice -n 5 pixi run python"

one() {  # name keep(0/1) builder args...
  local name=$1 keep=$2; shift 2
  if [ ! -f $N/results/exp2_$name.json ]; then
    [ -f $RS/$name/rankingset.pt ] || $PY "$@" --retrieval $RET --out_dir $RS --workers $W --name $name
    $PY $N/scripts/exp2_with_prefix.py --retrieval $RET --mass-cache $N/results/masses.npy \
        --prefix 531087 --fp $name=$RS/$name/rankingset.pt --out $N/results/exp2_$name.json
  fi
  $PY $H/scripts/exp1_add_fps.py --pairs $PAIRS --out $PAIRS --fp $name=$RS/$name/rankingset.pt
  if [ $keep = 0 ]; then rm -f $RS/$name/rankingset.pt; echo "deleted $RS/$name/rankingset.pt"; fi
}

[ -f $PAIRS ] || cp $N/results/pairs_index.parquet $PAIRS
# MARINA entropy FPs (already built over the new set in the dataset dir) -> Tanimoto columns
D=Datasets/MARINA-DB-OPEN
$PY $H/scripts/exp1_add_fps.py --pairs $PAIRS --out $PAIRS \
  --fp uniqmult=$D/RankingEntropyUniqueMultiplicity/rankingset.pt \
  --fp uncapped=$D/RankingEntropyMultiplicityUncapped/rankingset.pt \
  --fp cap5=$D/RankingEntropyMultiplicity/rankingset.pt \
  --fp sherlock=$D/RankingEntropy/rankingset.pt \
  --fp substructure=$D/RankingEntropySubstructure/rankingset.pt

B=$H/scripts
one ECFP4_2048       1 $B/build_ecfp4.py --fp-type morgan --radius 2  --nbits 2048
one ECFP4_16384      1 $B/build_ecfp4.py --fp-type morgan --radius 2  --nbits 16384
one Morgan_r10_16384 1 $B/build_ecfp4.py --fp-type morgan --radius 10 --nbits 16384
one FCFP9_2048       1 $B/build_ecfp4.py --fp-type fcfp   --radius 9  --nbits 2048
one FCFP9_16384      1 $B/build_ecfp4.py --fp-type fcfp   --radius 9  --nbits 16384
one Biosynfoni       1 $B/build_biosynfoni.py
one AtomPair_2048    0 $B/build_ecfp4.py --fp-type atompair --nbits 2048
one AtomPair_16384   0 $B/build_ecfp4.py --fp-type atompair --nbits 16384
one MAP4_2048        0 $B/build_map4.py --radius 2 --nbits 2048
one MAP4_16384       0 $B/build_map4.py --radius 2 --nbits 16384
echo ALL_REFS_DONE
