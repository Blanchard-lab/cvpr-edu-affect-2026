#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

INDEX_CSV="${INDEX_CSV:-data/processed/affectnet_sample_200.csv}"
OUT_CSV="${OUT_CSV:-data/processed/preds_libreface_affectnet_200.csv}"

"$PYTHON_BIN" -m src.inference.run_libreface_affectnet \
  --index_csv "$INDEX_CSV" \
  --out_csv "$OUT_CSV"