# mnova-atom-mapping

Does the Mnova prediction archive (`Snapshots/RawData/mnova_predictions.zip`) carry an
atom map back to the SMILES, and does it carry the J-couplings needed to derive HMBC/COSY
"ceiling" peak lists? Verified 2026-09-15 on shards 000000 / 000200 / 000400 (3,000 records).
Write-up: `wiki/models/marina2.0-hmbc-cosy.md`.

```bash
mkdir -p /tmp/mnova
for i in 000000 000200 000400; do
  unzip -p ~/Workspace/Snapshots/RawData/mnova_predictions.zip predictions_$i.jsonl > /tmp/mnova/p$i.jsonl
done
cd ~/Workspace/MARINA
pixi run python3 analysis/mnova-atom-mapping/scripts/verify_atom_mapping.py        > analysis/mnova-atom-mapping/results/verify_atom_mapping.txt
pixi run python3 analysis/mnova-atom-mapping/scripts/j_coupling_ceiling_stats.py   > analysis/mnova-atom-mapping/results/j_coupling_ceiling_stats.txt
```

`MNOVA_SHARDS` (glob) overrides the shard location. Environment: MARINA pixi (RDKit 2025.09.6).
