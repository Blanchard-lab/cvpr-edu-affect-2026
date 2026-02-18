#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

INPUT_DIR="${INPUT_DIR:-data/raw/videos}"
OUTPUT_DIR="${OUTPUT_DIR:-data/interim/frames}"

FPS="${FPS:-1}"
IMAGE_EXT="${IMAGE_EXT:-jpg}"

"$PYTHON_BIN" -m src.preprocess.extract_frames \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --fps "$FPS" \
  --image_ext "$IMAGE_EXT"
