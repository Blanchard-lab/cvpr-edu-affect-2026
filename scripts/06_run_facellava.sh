#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

INDEX_CSV="${INDEX_CSV:-data/processed/index_sampled_k10_m5.csv}"
OUT_CSV="${OUT_CSV:-data/processed/preds_facellava_k10_m5.csv}"

FACELLAVA_ROOT="${FACELLAVA_ROOT:-models/face-llava}"
MODEL_PATH="${MODEL_PATH:-checkpoints/FaceLLaVA}"
DEVICE="${DEVICE:-2}"
K_FRAMES="${K_FRAMES:-10}"

CUDA_VISIBLE_DEVICES=$DEVICE "$PYTHON_BIN" -m src.inference.run_facellava \
  --index_csv "$INDEX_CSV" \
  --out_csv "$OUT_CSV" \
  --facellava_root "$FACELLAVA_ROOT" \
  --model_path "$MODEL_PATH" \
  --k_frames "$K_FRAMES"
