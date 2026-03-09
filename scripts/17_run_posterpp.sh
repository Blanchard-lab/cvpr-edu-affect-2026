#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

INDEX_CSV="${INDEX_CSV:-data/processed/index_sampled_k10_m5.csv}"
OUT_CSV="${OUT_CSV:-data/processed/preds_posterpp_k10_m5.csv}"

POSTER_ROOT="${POSTER_ROOT:-./models/POSTER_V2}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./models/POSTER_V2/checkpoint/raf-db-model_best.pth}"
DEVICE="${DEVICE:-cuda}"
K_FRAMES="${K_FRAMES:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LABELS="${LABELS:-Surprise,Fear,Disgust,Happiness,Sadness,Anger,Neutral}"

"$PYTHON_BIN" -m src.inference.run_posterpp \
  --index_csv "$INDEX_CSV" \
  --out_csv "$OUT_CSV" \
  --poster_root "$POSTER_ROOT" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --device "$DEVICE" \
  --k_frames "$K_FRAMES" \
  --batch_size "$BATCH_SIZE" \
  --labels "$LABELS"