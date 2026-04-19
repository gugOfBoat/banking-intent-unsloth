#!/usr/bin/env bash
set -euo pipefail
# Full test set evaluation (accuracy + F1)
python scripts/inference.py --eval --config configs/inference.yaml
# Single message demo
python scripts/inference.py --config configs/inference.yaml --message "when will I get my refund?"
