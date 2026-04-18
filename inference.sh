#!/usr/bin/env bash
# Run inference with the trained model
python scripts/inference.py \
  --config configs/inference.yaml \
  --message "I want to top up my account"
