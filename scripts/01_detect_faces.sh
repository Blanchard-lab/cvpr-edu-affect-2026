#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

FRAMES_DIR="${FRAMES_DIR:-data/interim/frames}"
OUTPUT_DIR="${OUTPUT_DIR:-data/interim/detections}"

BACKEND="${BACKEND:-retinaface}"      # retinaface | insightface
DEVICE="${DEVICE:-cpu}"              # cpu | cuda
DETECT_EVERY_N="${DETECT_EVERY_N:-1}" # 1 = every frame, 2 = every other frame
MAX_FACES="${MAX_FACES:-3}"          # keep top-k faces per frame by score

"$PYTHON_BIN" -m src.preprocess.detect_faces \
  --frames_dir "$FRAMES_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --backend "$BACKEND" \
  --device "$DEVICE" \
  --detect_every_n "$DETECT_EVERY_N" \
  --max_faces "$MAX_FACES"
