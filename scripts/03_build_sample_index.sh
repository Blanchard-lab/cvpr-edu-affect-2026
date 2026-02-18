#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

CROPS_DIR="${CROPS_DIR:-data/interim/crops}"
REPORTS_CSV="${REPORTS_CSV:-data/raw/metadata/reports-agg.csv}"
OUT_DIR="${OUT_DIR:-data/processed}"

K_FRAMES="${K_FRAMES:-10}"
MARGIN_S="${MARGIN_S:-5}"
SEED="${SEED:-42}"

TEST_GROUPS="${TEST_GROUPS:-2}"
VAL_GROUPS="${VAL_GROUPS:-1}"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" -m src.dataset.build_sample_index \
  --crops_dir "$CROPS_DIR" \
  --reports_csv "$REPORTS_CSV" \
  --out_dir "$OUT_DIR" \
  --k_frames "$K_FRAMES" \
  --margin_s "$MARGIN_S" \
  --seed "$SEED" \
  --test_groups "$TEST_GROUPS" \
  --val_groups "$VAL_GROUPS"
