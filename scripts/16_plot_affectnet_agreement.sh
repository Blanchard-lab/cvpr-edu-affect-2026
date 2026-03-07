#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

EVAL_DIR="${EVAL_DIR:-results/eval_affectnet}"
FIG_DIR="${FIG_DIR:-results/figures_affectnet}"

SAVE_PDF="${SAVE_PDF:-1}"

mkdir -p "$FIG_DIR"

PYTHONNOUSERSITE=1 "$PYTHON_BIN" -m src.evaluation.plot_affectnet_agreement \
  --eval_dir "$EVAL_DIR" \
  --fig_dir "$FIG_DIR" \
  --save_pdf "$SAVE_PDF"