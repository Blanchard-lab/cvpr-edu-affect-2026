#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

INDEX_CSV="${INDEX_CSV:-data/processed/affectnet_sample_200.csv}"
OUT_CSV="${OUT_CSV:-data/processed/preds_openface_affectnet_200.csv}"

DEVICE="${DEVICE:-cpu}"
OPENFACE_ROOT="${OPENFACE_ROOT:-/s/babbage/h/nobackup/nblancha/public-datasets/sifat/OpenFace-3.0}"
WEIGHTS_PATH="${WEIGHTS_PATH:-/s/babbage/h/nobackup/nblancha/public-datasets/sifat/OpenFace-3.0/weights/stage2_epoch_7_loss_1.1606_acc_0.5589.pth}"

"$PYTHON_BIN" -m src.inference.run_openface_affectnet \
  --index_csv "$INDEX_CSV" \
  --out_csv "$OUT_CSV" \
  --device "$DEVICE" \
  --openface_root "$OPENFACE_ROOT" \
  --weights_path "$WEIGHTS_PATH"