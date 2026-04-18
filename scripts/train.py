"""
train.py  —  Fine-tune Qwen on BANKING77 intent classification with Unsloth.

  - Clearly document all hyperparameters        → configs/train.yaml
  - Save model checkpoint after fine-tuning      → outputs/run/checkpoint_final
  - Evaluate on independent test set             → final test eval after training
  - VRAM / GPU monitoring                        → printed before & after training

Modes
-----
    python scripts/train.py                  # standard train from configs/train.yaml
    python scripts/train.py --tune           # Optuna HPO → best params → final train
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, accuracy_score, classification_report
from trl import SFTTrainer, SFTConfig

# train_on_responses_only lives in different places depending on trl version
try:
    from trl import train_on_responses_only
except ImportError:
    from unsloth.chat_templates import train_on_responses_only

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
BASE_MODEL    = "unsloth/Qwen3.5-9B"
MAX_SEQ_LEN   = 512

# ChatML instruction / response delimiters (Qwen default)
INSTR_PART    = "<|im_start|>user\n"
RESP_PART     = "<|im_start|>assistant\n"

# ---------------------------------------------------------------------------
# GPU / VRAM Monitoring
# ---------------------------------------------------------------------------

def print_gpu_status(tag: str = ""):
    """Print current GPU VRAM usage."""
    if not torch.cuda.is_available():
        print(f"  [{tag}] No GPU available")
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved()  / 1024**3
    total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
    name      = torch.cuda.get_device_name()
    print(f"  [{tag}] GPU: {name}")
    print(f"  [{tag}] VRAM: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved / {total:.2f} GB total")
    print(f"  [{tag}] VRAM Free: ~{total - reserved:.2f} GB")


def free_vram():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    """Convert a CSV row to ChatML messages."""
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
    preds  = preds.flatten()
    labels = labels.flatten()
    mask = labels != -100
    micro_f1 = f1_score(labels[mask], preds[mask], average="micro",
                        zero_division=0)
    return {"eval_f1": float(micro_f1)}

# ---------------------------------------------------------------------------
# Unsloth model factory
# ---------------------------------------------------------------------------

def build_model(lora_r=16, lora_alpha=32):
    """Load base model in 4-bit and wrap with LoRA PEFT."""
    from unsloth import FastLanguageModel

    print_gpu_status("BEFORE model load")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )

    print_gpu_status("AFTER base model load (4-bit)")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=True,
    )

    print_gpu_status("AFTER LoRA PEFT applied")
    return model, tokenizer

# ---------------------------------------------------------------------------
# Test set evaluation (Section 2.2: evaluate on independent test set)
# ---------------------------------------------------------------------------

def evaluate_on_test(trainer, datasets, out_dir: Path):
    """Run evaluation on the held-out test set and save report."""
    print("\n" + "=" * 60)
    print("  EVALUATING ON TEST SET")
    print("=" * 60)

    test_results = trainer.evaluate(eval_dataset=datasets["test"])
    print(f"  Test F1 (micro): {test_results.get('eval_f1', 'N/A')}")
    print(f"  Test Loss:       {test_results.get('eval_loss', 'N/A')}")

    # Save test results
    results_path = out_dir / "test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=4, default=str)
    print(f"  Results saved to {results_path}")

    return test_results

# ---------------------------------------------------------------------------
# Save training summary (hyperparameters + metrics)
# ---------------------------------------------------------------------------

def save_training_summary(cfg: dict, train_result, test_results, out_dir: Path):
    """Export a JSON file documenting all hyperparameters and results."""
    summary = {
        "model": BASE_MODEL,
        "max_seq_length": MAX_SEQ_LEN,
        "hyperparameters": {
            "learning_rate": cfg.get("learning_rate"),
            "lora_r": cfg.get("r"),
            "lora_alpha": cfg.get("lora_alpha"),
            "num_train_epochs": cfg.get("num_train_epochs"),
            "per_device_train_batch_size": cfg.get("per_device_train_batch_size"),
            "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps"),
            "effective_batch_size": (
                cfg.get("per_device_train_batch_size", 4)
                * cfg.get("gradient_accumulation_steps", 4)
            ),
            "optimizer": "adamw_8bit",
            "lr_scheduler": "cosine",
            "warmup_ratio": 0.03,
            "lora_dropout": 0.05,
            "gradient_checkpointing": "unsloth",
            "quantization": "4-bit (NF4)",
        },
        "training_metrics": {
            "total_steps": train_result.global_step if train_result else None,
            "train_loss": train_result.training_loss if train_result else None,
            "training_time_sec": train_result.metrics.get("train_runtime") if train_result else None,
        },
        "test_metrics": test_results,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2) if torch.cuda.is_available() else None,
    }

    summary_path = out_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, default=str)
    print(f"\n  Training summary saved to {summary_path}")

# ---------------------------------------------------------------------------
# Standard training (uses configs/train.yaml)
# ---------------------------------------------------------------------------

def run_standard_training(cfg: dict, out_dir: Path):
    print("\n" + "=" * 60)
    print("  STANDARD TRAINING")
    print("=" * 60)

    lr         = cfg.get("learning_rate", 2e-4)
    lora_r     = cfg.get("r", 16)
    lora_alpha = cfg.get("lora_alpha", 32)
    epochs     = cfg.get("num_train_epochs", 3)
    batch_size = cfg.get("per_device_train_batch_size", 4)
    grad_acc   = cfg.get("gradient_accumulation_steps", 4)

    print(f"\n  Hyperparameters:")
    print(f"    learning_rate          = {lr}")
    print(f"    lora_r                 = {lora_r}")
    print(f"    lora_alpha             = {lora_alpha}")
    print(f"    num_train_epochs       = {epochs}")
    print(f"    per_device_batch_size  = {batch_size}")
    print(f"    gradient_accumulation  = {grad_acc}")
    print(f"    effective_batch_size   = {batch_size * grad_acc}")
    print(f"    optimizer              = adamw_8bit")
    print(f"    max_seq_length         = {MAX_SEQ_LEN}")
    print(f"    model                  = {BASE_MODEL}")
    print()

    model, tokenizer = build_model(lora_r=lora_r, lora_alpha=lora_alpha)
    datasets         = load_hf_datasets(tokenizer)

    checkpoint_dir = out_dir / "checkpoint_final"

    training_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        fp16=not _has_bf16(),
        bf16=_has_bf16(),
        logging_steps=10,
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

    # ---- TRAIN ----
    print_gpu_status("BEFORE training")
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed/60:.1f} minutes")
    print_gpu_status("AFTER training")

    # ---- SAVE MODEL CHECKPOINT (Section 2.2) ----
    print(f"\n  Saving model checkpoint to {checkpoint_dir} …")
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    # Also save training log
    log_df = pd.DataFrame(trainer.state.log_history)
    log_df.to_csv(out_dir / "training_log.csv", index=False)
    print(f"  Training log saved to {out_dir / 'training_log.csv'}")

    # ---- EVALUATE ON TEST SET (Section 2.2) ----
    test_results = evaluate_on_test(trainer, datasets, out_dir)

    # ---- SAVE FULL SUMMARY (Section 2.2: document all hyperparameters) ----
    save_training_summary(cfg, train_result, test_results, out_dir)

    print(f"\n  ✅ Model checkpoint saved to: {checkpoint_dir}")

    return train_result

# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------

def run_hpo(cfg: dict, out_dir: Path):
    print("\n" + "=" * 60)
    print("  OPTUNA HYPERPARAMETER SEARCH")
    print("=" * 60)

    import optuna

    n_trials = cfg.get("optuna_trials", 5)
    epochs   = cfg.get("optuna_epochs", 1)
    batch    = cfg.get("per_device_train_batch_size", 4)
    grad_acc = cfg.get("gradient_accumulation_steps", 4)

    # Build tokenizer once (doesn't depend on HPO params)
    _, tokenizer = build_model()
    datasets = load_hf_datasets(tokenizer)

    # Free the initial model — we only needed the tokenizer
    free_vram()

    def objective(trial):
        lr         = trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True)
        lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
        lora_r     = trial.suggest_categorical("r", [8, 16, 32])

        print(f"\n  --- Trial {trial.number} ---")
        print(f"      lr={lr:.6f}, r={lora_r}, alpha={lora_alpha}")

        trial_dir = out_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        model, _ = build_model(lora_r=lora_r, lora_alpha=lora_alpha)

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

        # Train 1 epoch for trial
        trainer.train()
        results = trainer.evaluate()
        f1 = results.get("eval_f1", 0.0)
        print(f"      Trial {trial.number} → F1 = {f1:.4f}")

        # Free VRAM between trials
        del model, trainer
        free_vram()

        return f1

    study = optuna.create_study(direction="maximize",
                                study_name="banking77-intent")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Print Optuna summary
    print("\n" + "=" * 60)
    print("  OPTUNA RESULTS")
    print("=" * 60)
    print(f"  Best F1:     {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Save Optuna results
    optuna_results = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ],
    }
    with open(out_dir / "optuna_results.json", "w") as f:
        json.dump(optuna_results, f, indent=4, default=str)

    # Free VRAM before final training
    free_vram()

    # Final training with best params
    print("\n  Starting FINAL training with best params …")
    final_cfg = {**cfg, **study.best_params}
    run_standard_training(final_cfg, out_dir / "best_model")

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _has_bf16() -> bool:
    try:
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

    print(f"  Config:  {args.config}")
    print(f"  Output:  {out_dir}")
    print(f"  Mode:    {'OPTUNA HPO + FINAL TRAIN' if args.tune else 'STANDARD TRAIN'}")

    if args.tune:
        run_hpo(cfg, out_dir)
    else:
        run_standard_training(cfg, out_dir)

    print("\n  🎉 All done!")


if __name__ == "__main__":
    main()
