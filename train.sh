#!/usr/bin/env bash
set -euo pipefail
# Preprocess data + Optuna HPO + Final training
python scripts/preprocess_data.py
python scripts/train.py --tune --config configs/train.yaml --output outputs/run
