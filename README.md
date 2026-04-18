# Project 2: Banking Intent Detection (Unsloth)

This repository contains the solution for Fine-tuning Intent Detection Model with BANKING dataset using Unsloth.

## Setup
1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. For specific Unsloth installation, please refer to the official Unsloth GitHub.

## Usage
### 1. Preprocess Data
```bash
python scripts/preprocess_data.py
```

### 2. Train Model
Modify `configs/train.yaml` with your hyper-parameters, then run:
```bash
bash train.sh
```

### 3. Inference
Modify `configs/inference.yaml` with your checkpoint path, then run:
```bash
bash inference.sh
```

## Video Demonstration
[Link to Video]
