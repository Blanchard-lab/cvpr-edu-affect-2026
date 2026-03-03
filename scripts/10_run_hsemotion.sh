#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

INDEX_CSV="${INDEX_CSV:-data/processed/index_sampled_k10_m5.csv}"
OUT_CSV="${OUT_CSV:-data/processed/preds_hsemotion_k10_m5.csv}"

ONNX_PATH="${ONNX_PATH:-models/hsemotion-onnx/hsemotion_1280.onnx}"
K_FRAMES="${K_FRAMES:-10}"

"$PYTHON_BIN" -m src.inference.run_hsemotion \
  --index_csv "$INDEX_CSV" \
  --out_csv "$OUT_CSV" \
  --onnx_path "$ONNX_PATH" \
  --k_frames "$K_FRAMES"