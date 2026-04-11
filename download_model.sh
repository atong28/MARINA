MODEL=marina_best

mkdir -p checkpoints
mkdir -p checkpoints/$MODEL

gdown 1e-lhm2I2GnMLY2d9rDB3Tt7vP8RnjnJ3 --output checkpoints/$MODEL/$MODEL.zip
unzip checkpoints/$MODEL/$MODEL.zip -d checkpoints/$MODEL
rm checkpoints/$MODEL/$MODEL.zip

cat << 'EOF' > checkpoints/models.json
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