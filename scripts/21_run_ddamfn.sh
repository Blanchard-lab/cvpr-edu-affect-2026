#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

INDEX_CSV="${INDEX_CSV:-data/processed/index_sampled_k10_m5.csv}"
OUT_CSV="${OUT_CSV:-data/processed/preds_ddamfn_k10_m5.csv}"

DDAMFN_ROOT="${DDAMFN_ROOT:-./models/DDAMFN}"
MODEL_PATH="${MODEL_PATH:-./models/DDAMFN/DDAMFN++/checkpoints_ver2.0/rafdb_epoch20_acc0.9204_bacc0.8617.pth}"
DEVICE="${DEVICE:-cuda}"
K_FRAMES="${K_FRAMES:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_HEAD="${NUM_HEAD:-2}"
NUM_CLASS="${NUM_CLASS:-7}"
LABELS="${LABELS:-Neutral,Happy,Sad,Surprise,Fear,Disgust,Angry}"

"$PYTHON_BIN" -m src.inference.run_ddamfn \
  --index_csv "$INDEX_CSV" \
  --out_csv "$OUT_CSV" \
  --ddamfn_root "$DDAMFN_ROOT" \
  --model_path "$MODEL_PATH" \
  --device "$DEVICE" \
  --k_frames "$K_FRAMES" \
  --batch_size "$BATCH_SIZE" \
  --num_head "$NUM_HEAD" \
  --num_class "$NUM_CLASS" \
  --labels "$LABELS"