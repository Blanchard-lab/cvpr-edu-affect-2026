#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

SPLITS_JSON="${SPLITS_JSON:-data/processed/splits_groups.json}"
OUT_DIR="${OUT_DIR:-results/eval_openface_posterpp}"

OPENFACE_CSV="${OPENFACE_CSV:-data/processed/preds_openface_k10_m5.csv}"
POSTERPP_CSV="${POSTERPP_CSV:-data/processed/preds_ddamfn_k10_m5.csv}"

EVAL_SPLIT="${EVAL_SPLIT:-all}"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" -m src.evaluation.eval_pairwise_preds \
  --splits_json "$SPLITS_JSON" \
  --out_dir "$OUT_DIR" \
  --model_a_csv "$OPENFACE_CSV" \
  --model_b_csv "$POSTERPP_CSV" \
  --model_a_name "openface" \
  --model_b_name "ddamfn" \
  --eval_split "$EVAL_SPLIT"