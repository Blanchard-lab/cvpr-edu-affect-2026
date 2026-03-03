#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

INDEX_CSV="${INDEX_CSV:-data/processed/index_sampled_k10_m5.csv}"
OUT_CSV="${OUT_CSV:-data/processed/preds_cage_k10_m5.csv}"

WEIGHTS_PATH="${WEIGHTS_PATH:-models/AffectNet8_Maxvit_Combined/model.pt}"
DEVICE="${DEVICE:-cuda}"
K_FRAMES="${K_FRAMES:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"

"$PYTHON_BIN" -m src.inference.run_cage \
  --index_csv "$INDEX_CSV" \
  --out_csv "$OUT_CSV" \
  --weights_path "$WEIGHTS_PATH" \
  --device "$DEVICE" \
  --k_frames "$K_FRAMES" \
  --batch_size "$BATCH_SIZE"