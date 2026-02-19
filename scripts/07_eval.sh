#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

SPLITS_JSON="${SPLITS_JSON:-data/processed/splits_groups.json}"
OUT_DIR="${OUT_DIR:-results/eval}"

OPENFACE_PREDS="${OPENFACE_PREDS:-data/processed/preds_openface_k10_m5.csv}"
LIBREFACE_PREDS="${LIBREFACE_PREDS:-data/processed/preds_libreface_k10_m5.csv}"

EVAL_SPLIT="${EVAL_SPLIT:-all}"   # all | test | val | train


mkdir -p "$OUT_DIR"

"$PYTHON_BIN" -m src.evaluation.eval_preds \
  --splits_json "$SPLITS_JSON" \
  --out_dir "$OUT_DIR" \
  --openface_csv "$OPENFACE_PREDS" \
  --libreface_csv "$LIBREFACE_PREDS"\
  --eval_split "$EVAL_SPLIT"

