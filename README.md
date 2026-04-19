# Project 2: Fine-Tuning Intent Detection Model with Banking Dataset

## Overview
Fine-tune **Qwen2.5-3B-Instruct** (4-bit quantized via [Unsloth](https://github.com/unslothai/unsloth)) on the [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) intent classification dataset using LoRA adapters and SFTTrainer.

## Project Structure
```
banking-intent-unsloth/
├── scripts/
│   ├── preprocess_data.py     # Data loading, stratified sampling, CSV export
│   ├── train.py               # Fine-tuning engine (Unsloth + Optuna HPO)
│   └── inference.py           # Standalone inference class
├── configs/
│   ├── train.yaml             # Training hyperparameters
│   └── inference.yaml         # Inference configuration
├── sample_data/
│   ├── train.csv              # 1925 samples (25/class × 77 classes)
│   ├── val.csv                # 385 samples (5/class × 77 classes)
│   ├── test.csv               # 385 samples (5/class × 77 classes)
│   └── label_map.json         # 77 intent label names
├── outputs/                   # Model checkpoints & logs (generated after training)
├── train.sh                   # Shell script for standard training
├── inference.sh               # Shell script for inference
├── requirements.txt           # Python dependencies
└── README.md
```

## Environment Setup

### Option A: Google Colab (Recommended — Free T4 GPU)
```python
# Cell 1: Install Unsloth + clone repo
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!git clone https://github.com/gugOfBoat/banking-intent-unsloth
%cd banking-intent-unsloth
!pip install -r requirements.txt
```

### Option B: Local / Conda
```bash
conda create -n banking-intent python=3.11 -y
conda activate banking-intent
pip install -r requirements.txt
# Install Unsloth separately (requires CUDA)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Data Preprocessing
```bash
python scripts/preprocess_data.py
```
Downloads the BANKING77 dataset from HuggingFace, performs stratified sampling (25 train / 5 val / 5 test per intent class), and saves clean CSVs to `sample_data/`.

## Training

### Standard Training (using `configs/train.yaml`)
```bash
bash train.sh
# or directly:
python scripts/train.py --config configs/train.yaml --output outputs/run
```

### Hyperparameter Tuning with Optuna
```bash
python scripts/train.py --tune --output outputs/run
```
Runs 5 Optuna trials to search over `learning_rate`, `lora_r`, and `lora_alpha`, then automatically trains a final model with the best parameters.

### Hyperparameters (documented in `configs/train.yaml`)
| Parameter | Value | Description |
|---|---|---|
| Model | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` | Instruct variant with ChatML template |
| LoRA rank (`r`) | 16 | Low-rank adapter dimension |
| LoRA alpha | 32 | Scaling factor |
| Learning rate | 2e-4 | Peak LR with cosine schedule |
| Batch size | 4 | Per-device batch size |
| Gradient accumulation | 4 | Effective batch = 16 |
| Epochs | 2 | Format-learning with Optuna tuning |
| Optimizer | AdamW 8-bit | Memory-efficient optimizer |
| Max sequence length | 512 | Input token limit |
| Gradient checkpointing | `"unsloth"` | Extreme VRAM savings |

### Training Outputs
After training, the `outputs/run/` directory contains:
- `checkpoint_final/` — Saved model + tokenizer
- `training_log.csv` — Per-step training metrics
- `training_summary.json` — All hyperparameters + final results
- `test_results.json` — Test set evaluation metrics

## Inference
```bash
bash inference.sh
# or directly:
python scripts/inference.py --config configs/inference.yaml --message "I want to top up my account"
```

### Python Usage
```python
from scripts.inference import IntentClassification

clf = IntentClassification("configs/inference.yaml")
label = clf("I want to top up my account")
print(label)  # → "top_up"
```

## Accuracy / Test Results
The final accuracy on the test set is recorded in `outputs/run/test_results.json` after training.

## Video Demonstration
[Link to Video](https://drive.google.com/your-video-link)

## References
- [BANKING77 Dataset](https://huggingface.co/datasets/PolyAI/banking77)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Qwen Models](https://huggingface.co/unsloth/Qwen2.5-3B)
