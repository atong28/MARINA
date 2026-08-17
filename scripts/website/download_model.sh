#!/bin/bash

# downloads the website models and writes the models.json the backend serves.
# usage: bash scripts/website/download_model.sh
#
# Fill in the gdown file ids below before running. Each id points at a self-contained
# zip whose entries sit at the model-directory root (params.json, best.ckpt,
# calibration.json, the RankingEntropy* vocabulary, ...), so it unpacks straight into
# checkpoints/<root>/. Leave an id empty to skip that model.

# marina_multiplicity is the served default. Its zip carries its own calibration.json
# and count_multiplicity_under_radius_6.pkl -- the Morgan bundle's are not valid for it.
MULTIPLICITY_ID=1FGQqtw38B2kD2lLccaJ4btiHzIFqlm0d

# marina_best (Morgan) is kept selectable. Re-packed in place (same file id) to add
# calibration.json and npclassifier.json, so the old single-file zip is superseded.
MORGAN_ID=18OPIPeDXfdU30zYNnjca244MjSpedWUn

mkdir -p checkpoints

fetch() {
    local root=$1 id=$2
    if [ -z "$id" ]; then
        echo "skipping $root: no gdown id set"
        return
    fi
    mkdir -p "checkpoints/$root"
    gdown "$id" --output "checkpoints/$root.zip"
    unzip -o "checkpoints/$root.zip" -d "checkpoints/$root"
    rm "checkpoints/$root.zip"
}

fetch marina_multiplicity "$MULTIPLICITY_ID"
fetch marina_best "$MORGAN_ID"

cat << EOF > checkpoints/models.json
{
    "models": [
        {
            "id": "marina_multiplicity",
            "root": "marina_multiplicity",
            "type": "marina",
            "default": true,
            "display_name": "MARINA (multiplicity FP)"
        },
        {
            "id": "marina_best",
            "root": "marina_best",
            "type": "marina",
            "default": false,
            "display_name": "MARINA (Morgan FP)"
        }
    ]
}
EOF
