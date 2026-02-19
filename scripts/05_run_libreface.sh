#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

INDEX_CSV="${INDEX_CSV:-data/processed/index_sampled_k10_m5.csv}"
OUT_CSV="${OUT_CSV:-data/processed/preds_libreface_k10_m5.csv}"

K_FRAMES="${K_FRAMES:-10}"

"$PYTHON_BIN" -m src.inference.run_libreface \
  --index_csv "$INDEX_CSV" \
  --out_csv "$OUT_CSV" \
  --k_frames "$K_FRAMES"