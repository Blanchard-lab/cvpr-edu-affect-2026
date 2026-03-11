#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

EVAL_DIR="${EVAL_DIR:-results/eval_openface_posterpp}"
FIG_DIR="${FIG_DIR:-results/figures_openface_posterpp}"
SAVE_PDF="${SAVE_PDF:-1}"

mkdir -p "$FIG_DIR"

PYTHONNOUSERSITE=1 "$PYTHON_BIN" -m src.evaluation.plot_pairwise_matrices \
  --eval_dir "$EVAL_DIR" \
  --fig_dir "$FIG_DIR" \
  --model_a_name "openface" \
  --model_b_name "ddamfn" \
  --save_pdf "$SAVE_PDF"