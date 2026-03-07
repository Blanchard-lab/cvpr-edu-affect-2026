#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

OPENFACE_CSV="${OPENFACE_CSV:-data/processed/preds_openface_affectnet_200.csv}"
LIBREFACE_CSV="${LIBREFACE_CSV:-data/processed/preds_libreface_affectnet_200.csv}"
OUT_DIR="${OUT_DIR:-results/eval_affectnet}"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" -m src.evaluation.eval_affectnet_agreement \
  --openface_csv "$OPENFACE_CSV" \
  --libreface_csv "$LIBREFACE_CSV" \
  --out_dir "$OUT_DIR"