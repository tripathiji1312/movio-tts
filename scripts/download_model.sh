#!/bin/bash
# Download fine-tuned model via direct HTTP (reliable for large files)
set -e

API_KEY="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"
DEST="models/indicf5_tanglish_v2"
mkdir -p "$DEST"

echo "Downloading config.json..."
curl -L -H "Authorization: Bearer $API_KEY" \
  "https://api.wandb.ai/artifactsV2/gcp-us/tripathiji1312-vit/QXJ0aWZhY3Q6MzQ2MjMyNDk2MQ==/793ff4c2ddc450aae4b66bfbd6ca8ccb" \
  -o "$DEST/config.json"

echo "Downloading vocab.txt..."
curl -L -H "Authorization: Bearer $API_KEY" \
  "https://api.wandb.ai/artifactsV2/gcp-us/tripathiji1312-vit/QXJ0aWZhY3Q6MzQ2Njk5MzY0NQ==/107f727df4d70d256c51ba9bdbd25b8e" \
  -o "$DEST/vocab.txt"

echo "Downloading model.pt (1.3GB — this will take a few minutes)..."
curl -L --progress-bar -H "Authorization: Bearer $API_KEY" \
  "https://api.wandb.ai/artifactsV2/gcp-us/tripathiji1312-vit/QXJ0aWZhY3Q6MzQ2Njk5MzY0NQ==/04bdde51c86972a048f3a378726b157a" \
  -o "$DEST/model.pt"

echo ""
echo "Done. Files:"
ls -lh "$DEST/"
