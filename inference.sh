#!/usr/bin/env bash
set -euo pipefail

# Suppress Unsloth/Transformers verbose banners during inference
export UNSLOTH_SUPPRESS_WARNINGS=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "  BANKING INTENT INFERENCE PIPELINE"
echo "============================================================"

# Full test set evaluation (accuracy + F1)
echo "[1/2] Evaluating on hold-out test set (385 samples)..."
python scripts/inference.py --eval --config configs/inference.yaml

# Single message demo
echo ""
echo "[2/2] Single query prediction demo..."
python scripts/inference.py --config configs/inference.yaml --message "I accidentally lost my card yesterday"
