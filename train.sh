#!/usr/bin/env bash
# Standard training from configs/train.yaml
conda activate banking-intent
python scripts/train.py \
  --config configs/train.yaml \
  --output outputs/run
