#!/usr/bin/env bash
# Upload demo audio and data assets to a GitHub release for the web demo
# CI pipeline.
#
# Usage:
#   ./scripts/upload-demo-audio.sh [TAG]
#
# If TAG is omitted, defaults to "web-assets-v1".
# If the release already exists, the assets are replaced.
#
# Prerequisites:
#   - gh CLI authenticated
#   - 15 MP3 files in demo_audio/
#   - Exported data in web/static/data/ (run export_web_data.py first)
#   - If trained with unfreeze_text_layers > 0, ONNX model in web/static/models/

set -euo pipefail

TAG="${1:-web-assets-v1}"
AUDIO_DIR="demo_audio"
DATA_DIR="web/static/data"
MODELS_DIR="web/static/models"
AUDIO_TARBALL="demo-audio.tar.gz"
DATA_TARBALL="demo-data.tar.gz"
MODELS_TARBALL="demo-models.tar.gz"

if [ ! -d "$AUDIO_DIR" ]; then
    echo "Error: $AUDIO_DIR/ directory not found." >&2
    echo "See manual_download.md for download instructions." >&2
    exit 1
fi

count=$(find "$AUDIO_DIR" -maxdepth 1 -name '*.mp3' | wc -l)
if [ "$count" -eq 0 ]; then
    echo "Error: no MP3 files found in $AUDIO_DIR/" >&2
    exit 1
fi
echo "Found $count MP3 file(s) in $AUDIO_DIR/"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: $DATA_DIR/ directory not found." >&2
    echo "Run export_web_data.py first." >&2
    exit 1
fi

for f in text_projection.json audio_embeddings.json tracks.json; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "Error: $DATA_DIR/$f not found. Run export_web_data.py first." >&2
        exit 1
    fi
done
echo "Data files present in $DATA_DIR/"

echo "Packaging $AUDIO_TARBALL..."
tar -czf "$AUDIO_TARBALL" -C "$AUDIO_DIR" .

echo "Packaging $DATA_TARBALL..."
tar -czf "$DATA_TARBALL" -C "$DATA_DIR" .

ASSETS=("$AUDIO_TARBALL" "$DATA_TARBALL")

# Package fine-tuned text encoder if exported
if [ -d "$MODELS_DIR" ] && [ -f "$MODELS_DIR/model.onnx" ]; then
    echo "Packaging $MODELS_TARBALL (fine-tuned text encoder)..."
    tar -czf "$MODELS_TARBALL" -C "$MODELS_DIR" .
    ASSETS+=("$MODELS_TARBALL")
else
    echo "No custom text encoder found in $MODELS_DIR/, skipping."
fi

echo "Uploading to release $TAG..."
if gh release view "$TAG" > /dev/null 2>&1; then
    gh release upload "$TAG" "${ASSETS[@]}" --clobber
else
    gh release create "$TAG" "${ASSETS[@]}" \
        --title "Web demo assets" \
        --notes "Audio and model data for the web demo. See manual_download.md for sources."
fi

rm -f "${ASSETS[@]}"
echo "Done. Release: $TAG"
