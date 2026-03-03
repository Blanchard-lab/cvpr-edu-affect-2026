#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

CAGE_CSV="${CAGE_CSV:-data/processed/preds_cage_k10_m5.csv}"
HSEMOTION_CSV="${HSEMOTION_CSV:-data/processed/preds_hsemotion_k10_m5.csv}"

FIG_DIR="${FIG_DIR:-results/figures_va}"
TAB_DIR="${TAB_DIR:-results/tables_va}"

SAVE_PDF="${SAVE_PDF:-0}"

mkdir -p "$FIG_DIR" "$TAB_DIR"

"$PYTHON_BIN" -m src.evaluation.plot_va \
  --cage_csv "$CAGE_CSV" \
  --hsemotion_csv "$HSEMOTION_CSV" \
  --fig_dir "$FIG_DIR" \
  --tab_dir "$TAB_DIR" \
  --save_pdf "$SAVE_PDF"