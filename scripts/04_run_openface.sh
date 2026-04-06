#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

INDEX_CSV="${INDEX_CSV:-data/processed/index_sampled_k10_m3.csv}"
OUT_CSV="${OUT_CSV:-data/processed/preds_openface_k10_m3.csv}"

"$PYTHON_BIN" -m src.inference.run_openface \
  --index_csv "$INDEX_CSV" \
  --out_csv "$OUT_CSV"