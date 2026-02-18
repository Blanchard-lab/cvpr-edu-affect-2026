#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

DETECTIONS_DIR="${DETECTIONS_DIR:-data/interim/detections}"
OUTPUT_DIR="${OUTPUT_DIR:-data/interim/crops}"

PAD_RATIO="${PAD_RATIO:-0.25}"      # 0.25 = add 25% padding around bbox
OUT_SIZE="${OUT_SIZE:-224}"         # output crop size (square)
IMAGE_EXT="${IMAGE_EXT:-jpg}"

"$PYTHON_BIN" -m src.preprocess.crop_faces \
  --detections_dir "$DETECTIONS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --pad_ratio "$PAD_RATIO" \
  --out_size "$OUT_SIZE" \
  --image_ext "$IMAGE_EXT"