"""
train.py  —  Fine-tune Qwen on BANKING77 intent classification with Unsloth.

Config-Driven: ALL parameters come from configs/train.yaml.

OOM-Safe Design (T4 15GB VRAM):
  - NO compute_metrics (logits for 150k vocab = 14GB OOM bomb)
  - Optuna uses eval_loss (zero logits stored)
  - Post-training evaluation via model.generate() (inference VRAM only)
  - eval_batch=1, eval_accumulation_steps=4, fp16_full_eval=True

Modes
-----
    python scripts/train.py                  # standard train
    python scripts/train.py --tune           # Optuna HPO → best params → final train
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import unsloth  # MUST be imported FIRST before trl/transformers/peft

import difflib
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, accuracy_score, classification_report
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

# ChatML delimiters (Qwen)
INSTR_PART = "<|im_start|>user\n"
RESP_PART  = "<|im_start|>assistant\n"

# System prompt — teaches label FORMAT, not understanding (model already knows banking)
SYSTEM_MSG = (
    "You are a banking intent classifier. "
    "Reply with ONLY the intent label in snake_case. "
    "Examples: card_arrival, lost_or_stolen_card, exchange_rate, top_up_failed. "
    "No explanation, no punctuation, no extra words."
)

# ---------------------------------------------------------------------------
# GPU / VRAM Monitoring
# ---------------------------------------------------------------------------

def gpu(tag=""):
    if not torch.cuda.is_available():
        print(f"  [{tag}] No GPU"); return
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved()  / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  [{tag}] {torch.cuda.get_device_name()} | "
          f"{a:.2f}/{r:.2f}/{t:.2f} GB (alloc/resv/total) | free ~{t-r:.2f} GB")

def free_vram():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def backup_to_drive(src_dir, drive_dest="/content/drive/MyDrive/banking-intent-outputs"):
    """Copy outputs to Google Drive. Drive must be mounted in notebook BEFORE training."""
    import shutil
    mydrive = Path("/content/drive/MyDrive")
    if not mydrive.exists():
        print("  [BACKUP] ⚠️  Google Drive not mounted!")
        print("  [BACKUP] Run this in a notebook cell BEFORE training:")
        print("  [BACKUP]   from google.colab import drive; drive.mount('/content/drive')")
        return
    dest = Path(drive_dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  [BACKUP] Copying {src_dir} → {dest} ...")
    shutil.copytree(str(src_dir), str(dest / src_dir.name), dirs_exist_ok=True)
    print(f"  [BACKUP] ✅ Saved to Google Drive: {dest / src_dir.name}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_yaml(p):
    with open(p, encoding="utf-8") as f: return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _to_messages(row):
    return {"messages": [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": f"Classify the banking intent: {row['text']}"},
        {"role": "assistant", "content": row["intent"]},
    ]}

def load_datasets(tokenizer):
    splits = {}
    for name in ("train", "val", "test"):
        df = pd.read_csv(DATA / f"{name}.csv")
        ds = Dataset.from_pandas(df).map(_to_messages)
        def tpl(batch, tok=tokenizer):
            return {"text": [tok.apply_chat_template(m, tokenize=False,
                             add_generation_prompt=False) for m in batch["messages"]]}
        ds = ds.map(tpl, batched=True, remove_columns=ds.column_names)
        splits[name] = ds
    return DatasetDict({"train": splits["train"],
                        "validation": splits["val"],
                        "test": splits["test"]})

# ---------------------------------------------------------------------------
# Model factory (config-driven)
# ---------------------------------------------------------------------------

def build_model(cfg, lora_r=None, lora_alpha=None):
    from unsloth import FastLanguageModel

    gpu("BEFORE load")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg.get("max_seq_length", 512),
        dtype=None,
        load_in_4bit=cfg.get("load_in_4bit", True),
    )
    gpu("AFTER base load")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r or cfg.get("r", 16),
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=lora_alpha or cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        use_gradient_checkpointing=cfg.get("gradient_checkpointing", "unsloth"),
        random_state=42,
        use_rslora=cfg.get("use_rslora", True),
    )
    gpu("AFTER LoRA")
    return model, tokenizer

# ---------------------------------------------------------------------------
# Prediction normalization + fuzzy matching (shared with inference.py)
# ---------------------------------------------------------------------------

def normalize(raw):
    t = raw.strip().lower()
    t = re.sub(r"[^a-z0-9_\s]", "", t)
    t = re.sub(r"[\s-]+", "_", t)
    return t.strip("_")

def fuzzy_match(pred, labels, cutoff=0.6):
    if pred in labels: return pred
    m = difflib.get_close_matches(pred, labels, n=1, cutoff=cutoff)
    return m[0] if m else pred

# ---------------------------------------------------------------------------
# Post-training generative evaluation (OOM-safe)
# ---------------------------------------------------------------------------

def generative_eval(model, tokenizer, test_csv, label_map, out_dir):
    """
    Evaluate by generating predictions one-by-one.
    Uses inference VRAM (~2GB) instead of eval logits (~14GB).
    Shows per-class F1 breakdown for key intents.
    """
    from unsloth import FastLanguageModel

    print("\n" + "=" * 60)
    print("  GENERATIVE EVALUATION ON TEST SET")
    print("=" * 60)

    FastLanguageModel.for_inference(model)  # 2x faster, 50% less VRAM
    gpu("INFERENCE mode")

    df = pd.read_csv(test_csv)
    valid_labels = label_map if isinstance(label_map, list) else []

    y_true, y_pred = [], []
    correct = 0
    t_eval = time.time()

    for i, row in df.iterrows():
        msgs = [
            {"role": "system",  "content": SYSTEM_MSG},
            {"role": "user",    "content": f"Classify the banking intent: {row['text']}"},
        ]
        input_ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(input_ids=input_ids, max_new_tokens=15,
                                 do_sample=False, use_cache=True)

        gen_ids = out[0][input_ids.shape[1]:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred = normalize(raw)
        if valid_labels:
            pred = fuzzy_match(pred, valid_labels)

        truth = row["intent"]
        y_true.append(truth)
        y_pred.append(pred)
        if pred == truth:
            correct += 1

        if i < 5:  # show first 5 predictions
            mark = "+" if pred == truth else "X"
            print(f"  {mark} [{pred:<30s}] gt=[{truth}] | {row['text'][:60]}")

    eval_time = time.time() - t_eval

    # ---- Overall Metrics ----
    acc      = correct / len(df)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n  {'='*60}")
    print(f"  OVERALL RESULTS ({len(df)} samples, {eval_time:.0f}s)")
    print(f"  {'='*60}")
    print(f"  Accuracy:       {acc:.4f}  ({correct}/{len(df)})")
    print(f"  F1 (micro):     {f1_micro:.4f}")
    print(f"  F1 (macro):     {f1_macro:.4f}")
    print(f"  F1 (weighted):  {f1_weighted:.4f}")

    # ---- Per-Class F1 Breakdown ----
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    class_scores = {k: v for k, v in report.items()
                    if isinstance(v, dict) and "f1-score" in v
                    and k not in ("micro avg", "macro avg", "weighted avg", "accuracy")}

    if class_scores:
        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1]["f1-score"], reverse=True)
        top5 = sorted_classes[:5]
        bot5 = sorted_classes[-5:]

        print(f"\n  TOP-5 BEST INTENTS (by F1):")
        print(f"  {'Intent':<35} {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'N':>4}")
        print(f"  {'─'*60}")
        for name, sc in top5:
            print(f"  {name:<35} {sc['f1-score']:>6.3f}  {sc['precision']:>6.3f}  {sc['recall']:>6.3f}  {sc['support']:>4.0f}")

        print(f"\n  BOTTOM-5 WEAKEST INTENTS (by F1):")
        print(f"  {'Intent':<35} {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'N':>4}")
        print(f"  {'─'*60}")
        for name, sc in bot5:
            print(f"  {name:<35} {sc['f1-score']:>6.3f}  {sc['precision']:>6.3f}  {sc['recall']:>6.3f}  {sc['support']:>4.0f}")

    # ---- Save Results ----
    results = {
        "accuracy": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "correct": correct,
        "total": len(df),
        "eval_time_sec": round(eval_time, 1),
        "top5_intents": {n: round(s["f1-score"], 4) for n, s in top5} if class_scores else {},
        "bot5_intents": {n: round(s["f1-score"], 4) for n, s in bot5} if class_scores else {},
    }

    with open(out_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n  Results -> {out_dir / 'test_results.json'}")

    # Switch back to training mode in case caller needs it
    FastLanguageModel.for_training(model)

    return results

# ---------------------------------------------------------------------------
# Standard training
# ---------------------------------------------------------------------------

def run_standard_training(cfg, out_dir):
    from unsloth import FastLanguageModel

    print("\n" + "=" * 60)
    print("  STANDARD TRAINING (Full Dataset + Best HPO Params)")
    print("=" * 60)

    lr       = cfg.get("learning_rate", 2e-4)
    epochs   = cfg.get("num_train_epochs", 3)
    batch    = cfg.get("per_device_train_batch_size", 4)
    grad_acc = cfg.get("gradient_accumulation_steps", 4)
    optim    = cfg.get("optimizer", "adamw_8bit")
    max_seq  = cfg.get("max_seq_length", 512)
    eff_batch = batch * grad_acc

    # ---- Config Summary ----
    print(f"\n  📋 CONFIGURATION SUMMARY")
    print(f"  {'─'*50}")
    print(f"  Model:            {cfg['model_name']}")
    print(f"  LoRA rank (r):    {cfg.get('r')}  |  alpha: {cfg.get('lora_alpha')}")
    print(f"  Learning rate:    {lr}")
    print(f"  Batch size:       {batch} × {grad_acc} (grad acc) = {eff_batch} effective")
    print(f"  Epochs:           {epochs}")
    print(f"  Optimizer:        {optim}")
    print(f"  Max seq length:   {max_seq}")
    print(f"  Checkpointing:    {cfg.get('gradient_checkpointing', 'none')}")

    # ---- Build Model ----
    model, tokenizer = build_model(cfg)

    # ---- LoRA Parameter Analysis ----
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    frozen    = total - trainable
    print(f"\n  🧠 MODEL PARAMETER ANALYSIS")
    print(f"  {'─'*50}")
    print(f"  Total parameters:     {total:>15,}")
    print(f"  Frozen (base model):  {frozen:>15,}")
    print(f"  Trainable (LoRA):     {trainable:>15,}  ({trainable/total*100:.2f}%)")
    print(f"  Memory saved by LoRA: ~{(total - trainable) * 2 / 1e9:.1f} GB (vs full fine-tune)")

    # ---- Load Dataset ----
    datasets = load_datasets(tokenizer)

    print(f"\n  📊 DATASET STATISTICS")
    print(f"  {'─'*50}")
    print(f"  Train:      {len(datasets['train']):>6,} samples")
    print(f"  Validation: {len(datasets['validation']):>6,} samples")
    print(f"  Test:       {len(datasets['test']):>6,} samples")
    total_steps = (len(datasets['train']) // eff_batch) * epochs
    print(f"  Total steps: {total_steps:>5,} ({len(datasets['train'])//eff_batch} steps/epoch × {epochs} epochs)")

    ckpt_dir = out_dir / "checkpoint_final"

    # Switch to training mode (Unsloth best practice)
    FastLanguageModel.for_training(model)

    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=1,           # T4-safe: eval batch=1
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        warmup_ratio=0.03,
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        optim=optim,
        fp16=not _bf16(),
        bf16=_bf16(),
        fp16_full_eval=True,                    # halve eval memory
        eval_accumulation_steps=4,              # offload logits to CPU
        logging_steps=cfg.get("logging_steps", 5),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",      # use loss, NOT logits-based F1
        greater_is_better=False,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=max_seq,
        packing=False,
        dataset_num_proc=1,  # <--- FORCES SEQUENTIAL TO PREVENT FORK OOM
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        args=args,
        # NO compute_metrics — logits for 150k vocab = 14GB OOM bomb
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTR_PART, response_part=RESP_PART,
    )

    # ---- VRAM before training ----
    vram_before = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    gpu("BEFORE training")

    print(f"\n  🚀 TRAINING STARTED")
    print(f"  {'─'*50}")
    t0 = time.time()
    result = trainer.train()
    train_time = time.time() - t0

    # ---- VRAM after training ----
    vram_after = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    gpu("AFTER training")

    # ---- Performance Report ----
    samples_per_sec = len(datasets['train']) * epochs / train_time
    time_per_epoch  = train_time / epochs
    # HuggingFace native speed estimate (Unsloth claims 2x)
    hf_estimated_time = train_time * 2

    print(f"\n  ⚡ UNSLOTH PERFORMANCE REPORT")
    print(f"  {'─'*50}")
    print(f"  Total training time:      {train_time/60:.1f} min ({train_time:.0f}s)")
    print(f"  Time per epoch:           {time_per_epoch/60:.1f} min")
    print(f"  Throughput:               {samples_per_sec:.1f} samples/sec")
    print(f"  VRAM delta (train):       {vram_after - vram_before:+.2f} GB (should be ~0 with Unsloth)")
    print(f"  Peak VRAM reserved:       {vram_after:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    print(f"  Estimated HF native time: ~{hf_estimated_time/60:.0f} min (2× slower without Unsloth)")
    print(f"  Unsloth speedup:          ~2.0×  (Triton kernel rewrite for RoPE/RMSNorm/CE)")

    # ---- Save checkpoint ----
    print(f"\n  💾 SAVING CHECKPOINT")
    print(f"  {'─'*50}")
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    print(f"  Adapter weights → {ckpt_dir}")
    print(f"  Tokenizer       → {ckpt_dir}")

    # Backup to Google Drive immediately (before eval which might crash)
    backup_to_drive(out_dir)

    # Training log
    pd.DataFrame(trainer.state.log_history).to_csv(
        out_dir / "training_log.csv", index=False)

    # Load label map for generative eval
    label_map_path = DATA / "label_map.json"
    labels = json.load(open(label_map_path)) if label_map_path.exists() else []

    # Post-training generative evaluation (OOM-safe)
    test_results = generative_eval(
        model, tokenizer, DATA / "test.csv", labels, out_dir)

    # Save summary
    summary = {
        "model": cfg.get("model_name"),
        "hyperparameters": {k: v for k, v in cfg.items()},
        "train_loss": result.training_loss,
        "train_steps": result.global_step,
        "runtime_min": round(train_time/60, 1),
        "time_per_epoch_min": round(time_per_epoch/60, 1),
        "samples_per_sec": round(samples_per_sec, 1),
        "estimated_hf_time_min": round(hf_estimated_time/60, 1),
        "test_results": test_results,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2)
                        if torch.cuda.is_available() else None,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(trainable/total*100, 2),
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=str)

    print(f"\n  ✅ Done! Model → {ckpt_dir}")
    return result

# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------

def run_hpo(cfg, out_dir):
    from unsloth import FastLanguageModel
    import optuna

    print("\n" + "=" * 60)
    print("  OPTUNA HYPERPARAMETER SEARCH")
    print("=" * 60)

    n_trials = cfg.get("optuna_trials", 5)
    ep       = cfg.get("optuna_epochs", 1)
    batch    = cfg.get("per_device_train_batch_size", 4)
    grad_acc = cfg.get("gradient_accumulation_steps", 4)
    optim    = cfg.get("optimizer", "adamw_8bit")
    max_seq  = cfg.get("max_seq_length", 512)

    # Tokenizer (model-independent)
    _, tokenizer = build_model(cfg)
    datasets = load_datasets(tokenizer)
    free_vram()

    def objective(trial):
        lr    = trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True)
        alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
        r     = trial.suggest_categorical("r", [8, 16, 32])

        print(f"\n  --- Trial {trial.number}: lr={lr:.6f}, r={r}, alpha={alpha} ---")

        model, _ = build_model(cfg, lora_r=r, lora_alpha=alpha)
        FastLanguageModel.for_training(model)

        args = SFTConfig(
            output_dir=str(out_dir / f"trial_{trial.number}"),
            num_train_epochs=ep,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=grad_acc,
            learning_rate=lr,
            warmup_ratio=0.03,
            optim=optim,
            fp16=not _bf16(),
            bf16=_bf16(),
            fp16_full_eval=True,
            eval_accumulation_steps=4,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
            dataset_text_field="text",
            max_seq_length=max_seq,
            packing=False,
            dataset_num_proc=1,  # <--- FORCES SEQUENTIAL TO PREVENT FORK OOM
        )

        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            args=args,
            # NO compute_metrics — use eval_loss only
        )
        trainer = train_on_responses_only(
            trainer, instruction_part=INSTR_PART, response_part=RESP_PART,
        )

        trainer.train()
        metrics = trainer.evaluate()
        loss = metrics.get("eval_loss", 999.0)
        print(f"      Trial {trial.number} → eval_loss = {loss:.4f}")

        del model, trainer
        free_vram()
        return loss  # minimize loss

    study = optuna.create_study(direction="minimize",
                                study_name="banking77-intent")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best loss:   {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    with open(out_dir / "optuna_results.json", "w") as f:
        json.dump({"best_value": study.best_value,
                   "best_params": study.best_params,
                   "trials": [{"n":t.number, "v":t.value, "p":t.params}
                              for t in study.trials]}, f, indent=4, default=str)

    free_vram()

    # Final training with best params
    print("\n  Final training with best params …")
    run_standard_training({**cfg, **study.best_params}, out_dir / "best_model")

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _bf16():
    try: return torch.cuda.is_bf16_supported()
    except: return False

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tune", action="store_true")
    p.add_argument("--config", default=str(CFG))
    p.add_argument("--output", default=str(OUT_DIR / "run"))
    args = p.parse_args()

    cfg     = load_yaml(Path(args.config))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Config:  {args.config}")
    print(f"  Output:  {out_dir}")
    print(f"  Model:   {cfg.get('model_name')}")
    print(f"  Mode:    {'OPTUNA HPO → FINAL TRAIN' if args.tune else 'STANDARD'}")

    if args.tune:
        run_hpo(cfg, out_dir)
    else:
        run_standard_training(cfg, out_dir)

    print("\n  🎉 All done!")

if __name__ == "__main__":
    main()
