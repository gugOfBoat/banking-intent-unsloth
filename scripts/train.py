"""
train.py  —  Fine-tune Qwen on BANKING77 intent classification with Unsloth.

Config-Driven Development: ALL parameters come from configs/train.yaml.
No model name, no hyperparameters are hardcoded in this file.

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

import unsloth  # MUST be imported FIRST before trl/transformers/peft

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score
from trl import SFTTrainer, SFTConfig

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

# ChatML instruction / response delimiters (Qwen default)
INSTR_PART = "<|im_start|>user\n"
RESP_PART  = "<|im_start|>assistant\n"

# ---------------------------------------------------------------------------
# GPU / VRAM Monitoring
# ---------------------------------------------------------------------------

def print_gpu_status(tag: str = ""):
    if not torch.cuda.is_available():
        print(f"  [{tag}] No GPU available")
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    resv  = torch.cuda.memory_reserved()  / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    name  = torch.cuda.get_device_name()
    print(f"  [{tag}] GPU: {name} | "
          f"VRAM: {alloc:.2f}/{resv:.2f}/{total:.2f} GB "
          f"(alloc/reserved/total) | Free: ~{total-resv:.2f} GB")


def free_vram():
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
    return {"messages": [
        {"role": "user",      "content": f"Classify the banking intent: {row['text']}"},
        {"role": "assistant", "content": row["intent"]},
    ]}


def load_hf_datasets(tokenizer):
    splits = {}
    for name in ("train", "val", "test"):
        csv_path = DATA / f"{name}.csv"
        df = pd.read_csv(csv_path)
        ds = Dataset.from_pandas(df)
        ds = ds.map(_row_to_messages)

        def apply_template(batch, tok=tokenizer):
            texts = [
                tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=False)
                for msgs in batch["messages"]
            ]
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
    preds  = np.argmax(logits, axis=-1).flatten()
    labels = labels.flatten()
    mask   = labels != -100
    return {"eval_f1": float(f1_score(labels[mask], preds[mask],
                                      average="micro", zero_division=0))}

# ---------------------------------------------------------------------------
# Unsloth model factory (Config-Driven)
# ---------------------------------------------------------------------------

def build_model(cfg: dict, lora_r=None, lora_alpha=None):
    """Load model from config. lora_r/lora_alpha override for Optuna."""
    from unsloth import FastLanguageModel

    model_name = cfg["model_name"]
    max_seq    = cfg.get("max_seq_length", 512)
    load_4bit  = cfg.get("load_in_4bit", True)

    _r     = lora_r     or cfg.get("r", 16)
    _alpha = lora_alpha or cfg.get("lora_alpha", 32)
    _drop  = cfg.get("lora_dropout", 0.05)
    _rsl   = cfg.get("use_rslora", True)
    _gc    = cfg.get("gradient_checkpointing", "unsloth")

    print_gpu_status("BEFORE model load")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq,
        dtype=None,
        load_in_4bit=load_4bit,
    )

    print_gpu_status("AFTER base model load")

    model = FastLanguageModel.get_peft_model(
        model,
        r=_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=_alpha,
        lora_dropout=_drop,
        bias="none",
        use_gradient_checkpointing=_gc,
        random_state=42,
        use_rslora=_rsl,
    )

    print_gpu_status("AFTER LoRA PEFT applied")
    return model, tokenizer

# ---------------------------------------------------------------------------
# Test set evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(trainer, datasets, out_dir: Path):
    print("\n" + "=" * 60)
    print("  EVALUATING ON TEST SET")
    print("=" * 60)
    results = trainer.evaluate(eval_dataset=datasets["test"])
    print(f"  Test F1 (micro): {results.get('eval_f1', 'N/A')}")
    print(f"  Test Loss:       {results.get('eval_loss', 'N/A')}")

    with open(out_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=4, default=str)
    return results

# ---------------------------------------------------------------------------
# Save training summary (all hyperparameters documented)
# ---------------------------------------------------------------------------

def save_training_summary(cfg, train_result, test_results, out_dir):
    summary = {
        "model": cfg.get("model_name"),
        "max_seq_length": cfg.get("max_seq_length"),
        "hyperparameters": {k: v for k, v in cfg.items()},
        "training_metrics": {
            "total_steps": train_result.global_step if train_result else None,
            "train_loss": train_result.training_loss if train_result else None,
            "runtime_sec": train_result.metrics.get("train_runtime") if train_result else None,
        },
        "test_metrics": test_results,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2) if torch.cuda.is_available() else None,
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=str)
    print(f"  Summary saved to {out_dir / 'training_summary.json'}")

# ---------------------------------------------------------------------------
# Standard training
# ---------------------------------------------------------------------------

def run_standard_training(cfg: dict, out_dir: Path):
    print("\n" + "=" * 60)
    print("  STANDARD TRAINING")
    print("=" * 60)

    lr       = cfg.get("learning_rate", 2e-4)
    epochs   = cfg.get("num_train_epochs", 3)
    batch    = cfg.get("per_device_train_batch_size", 4)
    grad_acc = cfg.get("gradient_accumulation_steps", 4)
    optim    = cfg.get("optimizer", "adamw_8bit")
    sched    = cfg.get("lr_scheduler_type", "cosine")
    warmup   = cfg.get("warmup_ratio", 0.03)
    max_seq  = cfg.get("max_seq_length", 512)

    print(f"\n  Model:        {cfg.get('model_name')}")
    print(f"  LR:           {lr}")
    print(f"  LoRA r/alpha: {cfg.get('r')}/{cfg.get('lora_alpha')}")
    print(f"  Epochs:       {epochs}")
    print(f"  Eff. batch:   {batch * grad_acc}")
    print()

    model, tokenizer = build_model(cfg)
    datasets         = load_hf_datasets(tokenizer)
    ckpt_dir         = out_dir / "checkpoint_final"

    training_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        warmup_ratio=warmup,
        lr_scheduler_type=sched,
        optim=optim,
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
        max_seq_length=max_seq,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        args=training_args,
        compute_metrics=compute_metrics,
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTR_PART, response_part=RESP_PART,
    )

    # ---- TRAIN ----
    print_gpu_status("BEFORE training")
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed/60:.1f} min")
    print_gpu_status("AFTER training")

    # ---- SAVE CHECKPOINT ----
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    print(f"  Checkpoint → {ckpt_dir}")

    # ---- TRAINING LOG ----
    pd.DataFrame(trainer.state.log_history).to_csv(
        out_dir / "training_log.csv", index=False)

    # ---- TEST EVAL ----
    test_results = evaluate_on_test(trainer, datasets, out_dir)

    # ---- SUMMARY ----
    save_training_summary(cfg, train_result, test_results, out_dir)

    print(f"\n  ✅ Model saved to: {ckpt_dir}")
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
    optim    = cfg.get("optimizer", "adamw_8bit")
    max_seq  = cfg.get("max_seq_length", 512)

    # Tokenizer from first model load
    _, tokenizer = build_model(cfg)
    datasets = load_hf_datasets(tokenizer)
    free_vram()

    def objective(trial):
        lr     = trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True)
        alpha  = trial.suggest_categorical("lora_alpha", [16, 32, 64])
        r      = trial.suggest_categorical("r", [8, 16, 32])

        print(f"\n  --- Trial {trial.number}: lr={lr:.6f}, r={r}, alpha={alpha} ---")

        trial_dir = out_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        model, _ = build_model(cfg, lora_r=r, lora_alpha=alpha)

        args = SFTConfig(
            output_dir=str(trial_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,
            gradient_accumulation_steps=grad_acc,
            learning_rate=lr,
            warmup_ratio=0.03,
            optim=optim,
            fp16=not _has_bf16(),
            bf16=_has_bf16(),
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
            dataset_text_field="text",
            max_seq_length=max_seq,
            packing=False,
        )

        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            args=args,
            compute_metrics=compute_metrics,
        )
        trainer = train_on_responses_only(
            trainer, instruction_part=INSTR_PART, response_part=RESP_PART,
        )

        trainer.train()
        results = trainer.evaluate()
        f1 = results.get("eval_f1", 0.0)
        print(f"      Trial {trial.number} → F1 = {f1:.4f}")

        del model, trainer
        free_vram()
        return f1

    study = optuna.create_study(direction="maximize",
                                study_name="banking77-intent")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best F1:     {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Save Optuna results
    with open(out_dir / "optuna_results.json", "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
            "trials": [{"n": t.number, "v": t.value, "p": t.params}
                       for t in study.trials],
        }, f, indent=4, default=str)

    free_vram()

    # Final training with best params
    print("\n  Final training with best params …")
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
                   help="Path to train.yaml")
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
    print(f"  Model:   {cfg.get('model_name')}")
    print(f"  Mode:    {'OPTUNA HPO + FINAL TRAIN' if args.tune else 'STANDARD TRAIN'}")

    if args.tune:
        run_hpo(cfg, out_dir)
    else:
        run_standard_training(cfg, out_dir)

    print("\n  🎉 All done!")


if __name__ == "__main__":
    main()
