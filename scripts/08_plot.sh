    #!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

EVAL_DIR="${EVAL_DIR:-results/eval}"
FIG_DIR="${FIG_DIR:-results/figures}"
TAB_DIR="${TAB_DIR:-results/tables}"

SAVE_PDF="${SAVE_PDF:-1}"

mkdir -p "$FIG_DIR" "$TAB_DIR"

PYTHONNOUSERSITE=1 PYTHONPATH="" "$PYTHON_BIN" -m src.evaluation.plot_results \
  --eval_dir "$EVAL_DIR" \
  --fig_dir "$FIG_DIR" \
  --tab_dir "$TAB_DIR" \
  --save_pdf "$SAVE_PDF"

