#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

INDEX_CSV="${INDEX_CSV:-data/processed/index_sampled_k10_m5.csv}"
OUT_ROOT="${OUT_ROOT:-sampled_data}"

"$PYTHON_BIN" -m src.dataset.package_sampled_data \
  --index_csv "$INDEX_CSV" \
  --out_root "$OUT_ROOT"