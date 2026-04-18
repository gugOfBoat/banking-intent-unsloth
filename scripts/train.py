"""
train.py  —  Fine-tune Qwen on BANKING77 intent classification with Unsloth.

Modes
-----
    python scripts/train.py                  # standard train from configs/train.yaml
    python scripts/train.py --tune           # Optuna HPO → best params → final train
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score
from trl import SFTTrainer, SFTConfig, train_on_responses_only

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "sample_data"
CFG     = ROOT / "configs" / "train.yaml"
OUT_DIR = ROOT / "outputs"

# ---------------------------------------------------------------------------
# Model / dataset constants
# ---------------------------------------------------------------------------
BASE_MODEL    = "unsloth/Qwen2.5-7B"          # 7B ≈ 9B tier – fits T4 in 4-bit
MAX_SEQ_LEN   = 512

# ChatML instruction / response delimiters (Qwen default)
INSTR_PART    = "<|im_start|>user\n"
RESP_PART     = "<|im_start|>assistant\n"

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _row_to_messages(row: dict) -> dict:
    """Convert a CSV row to ChatML messages + apply_chat_template."""
    messages = [
        {"role": "user",      "content": f"Classify the banking intent: {row['text']}"},
        {"role": "assistant", "content": row["intent"]},
    ]
    return {"messages": messages}


def load_hf_datasets(tokenizer):
    """Load CSVs → HF Datasets, apply chat template, return DatasetDict."""
    splits = {}
    for name in ("train", "val", "test"):
        csv_path = DATA / f"{name}.csv"
        df = pd.read_csv(csv_path)
        ds = Dataset.from_pandas(df)
        ds = ds.map(_row_to_messages)

        def apply_template(batch, tok=tokenizer):
            texts = []
            for msgs in batch["messages"]:
                texts.append(
                    tok.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
            return {"text": texts}

        ds = ds.map(apply_template, batched=True, remove_columns=ds.column_names)
        splits[name] = ds

    return DatasetDict({"train": splits["train"],
                        "validation": splits["val"],
                        "test": splits["test"]})

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    # Flatten (seq2seq models may add dim)
    preds  = preds.flatten()
    labels = labels.flatten()
    # Ignore padding token (-100)
    mask = labels != -100
    micro_f1 = f1_score(labels[mask], preds[mask], average="micro",
                        zero_division=0)
    return {"eval_f1": float(micro_f1)}

# ---------------------------------------------------------------------------
# Unsloth model factory
# ---------------------------------------------------------------------------

def build_model(lr=2e-4, lora_r=16, lora_alpha=32):
    """Load base model in 4-bit and wrap with LoRA PEFT."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,          # auto detect
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",   # extreme VRAM savings
        random_state=42,
        use_rslora=True,
    )
    return model, tokenizer

# ---------------------------------------------------------------------------
# Standard training (uses configs/train.yaml)
# ---------------------------------------------------------------------------

def run_standard_training(cfg: dict, out_dir: Path):
    print("=" * 60)
    print("  STANDARD TRAINING")
    print("=" * 60)

    lr         = cfg.get("learning_rate", 2e-4)
    lora_r     = cfg.get("r", 16)
    lora_alpha = cfg.get("lora_alpha", 32)
    epochs     = cfg.get("num_train_epochs", 3)
    batch_size = cfg.get("per_device_train_batch_size", 4)
    grad_acc   = cfg.get("gradient_accumulation_steps", 4)

    model, tokenizer = build_model(lr=lr, lora_r=lora_r, lora_alpha=lora_alpha)
    datasets         = load_hf_datasets(tokenizer)

    training_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",              # Unsloth 8-bit Adam
        fp16=not _has_bf16(),
        bf16=_has_bf16(),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        args=training_args,
        compute_metrics=compute_metrics,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTR_PART,
        response_part=RESP_PART,
    )

    trainer.train()
    trainer.save_model(str(out_dir / "checkpoint_final"))
    tokenizer.save_pretrained(str(out_dir / "checkpoint_final"))
    print(f"\n  Model saved to {out_dir / 'checkpoint_final'}")

# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------

def run_hpo(cfg: dict, out_dir: Path):
    print("=" * 60)
    print("  OPTUNA HYPERPARAMETER SEARCH")
    print("=" * 60)

    import optuna
    from unsloth import FastLanguageModel

    n_trials = cfg.get("optuna_trials", 5)
    epochs   = cfg.get("optuna_epochs", 1)          # 1 epoch per trial to save time
    batch    = cfg.get("per_device_train_batch_size", 4)
    grad_acc = cfg.get("gradient_accumulation_steps", 4)

    # We need a tokenizer early for dataset preparation
    # Use default params for tokenizer — it doesn't depend on HPO params
    _, tokenizer = build_model()
    datasets = load_hf_datasets(tokenizer)

    best_params_store = {}

    def objective(trial):
        lr         = trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True)
        lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
        lora_r     = trial.suggest_categorical("r", [8, 16, 32])

        trial_dir = out_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        model, _ = build_model(lr=lr, lora_r=lora_r, lora_alpha=lora_alpha)

        training_args = SFTConfig(
            output_dir=str(trial_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,
            gradient_accumulation_steps=grad_acc,
            learning_rate=lr,
            warmup_ratio=0.03,
            optim="adamw_8bit",
            fp16=not _has_bf16(),
            bf16=_has_bf16(),
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            packing=False,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            args=training_args,
            compute_metrics=compute_metrics,
        )
        trainer = train_on_responses_only(
            trainer,
            instruction_part=INSTR_PART,
            response_part=RESP_PART,
        )

        results = trainer.evaluate()
        f1 = results.get("eval_f1", 0.0)

        # Free VRAM between trials
        del model, trainer
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

        return f1

    study = optuna.create_study(direction="maximize",
                                study_name="banking77-intent")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print("\n  Best params:", best)
    best_params_store.update(best)

    # Final training with best params
    print("\n  Starting final training with best params …")
    final_cfg = {**cfg, **best}
    run_standard_training(final_cfg, out_dir / "best_model")

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _has_bf16() -> bool:
    try:
        import torch
        return torch.cuda.is_bf16_supported()
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train intent classifier with Unsloth")
    p.add_argument("--tune", action="store_true",
                   help="Run Optuna HPO before final training")
    p.add_argument("--config", default=str(CFG),
                   help="Path to train.yaml (default: configs/train.yaml)")
    p.add_argument("--output", default=str(OUT_DIR / "run"),
                   help="Output directory")
    return p.parse_args()


def main():
    args    = parse_args()
    cfg     = load_yaml(Path(args.config))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tune:
        run_hpo(cfg, out_dir)
    else:
        run_standard_training(cfg, out_dir)


if __name__ == "__main__":
    main()
