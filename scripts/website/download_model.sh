#!/bin/bash

# downloads the best model and sets up usage for website.
# usage: bash scripts/website/download_model.sh

MODEL=marina_best

mkdir -p checkpoints
mkdir -p checkpoints/$MODEL

gdown 18OPIPeDXfdU30zYNnjca244MjSpedWUn --output checkpoints/$MODEL/$MODEL.zip
unzip checkpoints/$MODEL/$MODEL.zip -d checkpoints/$MODEL
rm checkpoints/$MODEL/$MODEL.zip

cat << EOF > checkpoints/models.json
{
    "models": [
        {
            "id": "$MODEL",
            "root": "$MODEL",
            "type": "marina",
            "default": true,
            "display_name": "MARINA"
        }
    ]
}
EOF
