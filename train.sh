#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "  BANKING INTENT — FULL TRAINING PIPELINE"
echo "============================================================"

# Step 1: Prepare full dataset (download + stratified split)
echo "[1/2] Preprocessing data..."
python scripts/preprocess_data.py

# Step 2: Train with pre-tuned HPO params from train.yaml
# (Optuna HPO was already run on a subset — best params hardcoded in config)
echo "[2/2] Starting Unsloth fine-tuning (5 epochs, full dataset)..."
python scripts/train.py --config configs/train.yaml --output outputs/run
