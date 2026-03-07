#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

AFFECTNET_ROOT="${AFFECTNET_ROOT:-data/affectnet_subset}"
LABELS_CSV="${LABELS_CSV:-data/affectnet_subset/labels.csv}"
OUT_CSV="${OUT_CSV:-data/processed/affectnet_sample_200.csv}"

N_SAMPLES="${N_SAMPLES:-3000}"
SEED="${SEED:-42}"

"$PYTHON_BIN" -m src.dataset.sample_affectnet_subset \
  --affectnet_root "$AFFECTNET_ROOT" \
  --labels_csv "$LABELS_CSV" \
  --out_csv "$OUT_CSV" \
  --n_samples "$N_SAMPLES" \
  --seed "$SEED"