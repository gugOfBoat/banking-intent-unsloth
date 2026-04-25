"""
inference.py — Standalone inference for banking intent classification.

Features:
  - Config-driven (model path from YAML)
  - IntentClassification class with __init__ and __call__ (Section 2.3)
  - Smart prediction normalization + fuzzy label matching
    (handles LLM output quirks: extra punctuation, casing, typos)
"""

import json
import re
import difflib
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Prediction normalization (catches LLM "garbage" output)
# ---------------------------------------------------------------------------

def normalize_prediction(raw: str) -> str:
    """
    Clean up raw LLM output to a standard label format.
    Examples:
        "Card_arrival."   → "card_arrival"
        "  Top Up!  "     → "top_up"
        "card-arrival\n"  → "card_arrival"
    """
    text = raw.strip().lower()
    text = re.sub(r"[^a-z0-9_\s]", "", text)  # remove punctuation
    text = re.sub(r"[\s-]+", "_", text)        # spaces/hyphens → underscore
    text = re.sub(r"_+", "_", text)            # collapse multiple underscores
    text = text.strip("_")
    return text


def map_prediction_to_label(prediction: str, valid_labels: list,
                            cutoff: float = 0.6) -> str:
    """
    Fuzzy-match a (possibly noisy) prediction to the closest valid label.

    Uses difflib.get_close_matches for edit-distance based matching.
    If no match is found above the cutoff, returns the raw prediction.

    Parameters
    ----------
    prediction : str
        Normalized prediction string.
    valid_labels : list[str]
        The 77 canonical BANKING77 intent labels.
    cutoff : float
        Minimum similarity ratio (0–1). Default 0.6.

    Returns
    -------
    str
        The best-matching canonical label, or the raw prediction if no match.
    """
    if prediction in valid_labels:
        return prediction  # exact match, skip fuzzy search

    matches = difflib.get_close_matches(prediction, valid_labels,
                                        n=1, cutoff=cutoff)
    if matches:
        return matches[0]

    return prediction  # no close match found


# ---------------------------------------------------------------------------
# IntentClassification class (Section 2.3)
# ---------------------------------------------------------------------------

class IntentClassification:
    """Banking intent classifier powered by a fine-tuned Qwen model (Unsloth)."""

    def __init__(self, model_path: str):
        """
        Load configuration, tokenizer, model checkpoint, and label map.

        Parameters
        ----------
        model_path : str
            Path to a YAML config file containing:
              - model_checkpoint: path to saved model directory
              - max_new_tokens: max tokens to generate (default 32)
              - label_map: path to label_map.json (for fuzzy matching)
        """
        config_path = Path(model_path)
        root_dir    = config_path.resolve().parent.parent

        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        checkpoint     = root_dir / self.config["model_checkpoint"]
        self.max_new   = self.config.get("max_new_tokens", 32)

        # Load valid labels for fuzzy matching
        label_map_path = root_dir / self.config.get("label_map",
                                                     "sample_data/label_map.json")
        if label_map_path.exists():
            with open(label_map_path, encoding="utf-8") as f:
                self.valid_labels = json.load(f)
            print(f"[INFO] Loaded {len(self.valid_labels)} valid labels for fuzzy matching")
        else:
            self.valid_labels = []
            print("[WARN] No label_map.json found — fuzzy matching disabled")

        # Load model & tokenizer
        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(checkpoint),
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)  # 2x faster, 50% less VRAM

        print(f"[INFO] Model loaded from: {checkpoint}")

    def __call__(self, message: str) -> str:
        """
        Predict the intent label for a single text message.

        The raw LLM output is normalized (lowered, stripped of punctuation)
        and then fuzzy-matched to the closest canonical BANKING77 label.

        Parameters
        ----------
        message : str
            The user's banking query.

        Returns
        -------
        predicted_label : str
            The predicted (fuzzy-matched) intent class name.
        """
        # System prompt — must match the one used during training
        SYSTEM_MSG = (
            "You are a banking intent classifier. "
            "Reply with ONLY the intent label in snake_case. "
            "Examples: card_arrival, lost_or_stolen_card, exchange_rate, top_up_failed. "
            "No explanation, no punctuation, no extra words."
        )
        # Plain text messages (Qwen2.5 is a text model, no VL format needed)
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": f"Classify the banking intent: {message}"},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=15,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
        )

        # Decode only newly generated tokens
        gen_ids = outputs[0][input_ids.shape[1]:]
        raw_output = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Normalize + fuzzy match
        normalized = normalize_prediction(raw_output)

        if self.valid_labels:
            predicted_label = map_prediction_to_label(normalized, self.valid_labels)
        else:
            predicted_label = normalized

        return predicted_label


# ---------------------------------------------------------------------------
# Standalone usage + full test set evaluation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pandas as pd
    from sklearn.metrics import f1_score, accuracy_score, classification_report

    parser = argparse.ArgumentParser(description="Run intent classification inference")
    parser.add_argument("--config", default="configs/inference.yaml",
                        help="Path to inference config YAML")
    parser.add_argument("--message", default=None,
                        help="Single message to classify")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate on full test set (sample_data/test.csv)")
    args = parser.parse_args()

    clf = IntentClassification(args.config)

    if args.eval:
        # ---- Full test set evaluation ----
        test_path = Path(args.config).resolve().parent.parent / "sample_data" / "test.csv"
        df = pd.read_csv(test_path)
        print(f"\n{'='*60}")
        print(f"  EVALUATING ON TEST SET: {len(df)} samples")
        print(f"{'='*60}")

        import time as _time
        t0 = _time.time()
        y_true, y_pred = [], []
        for i, row in df.iterrows():
            pred = clf(row["text"])
            truth = row["intent"]
            y_true.append(truth)
            y_pred.append(pred)

            mark = "+" if pred == truth else "X"
            if i < 8 or pred != truth:  # show first 8 + all errors
                print(f"  {mark} [{pred:<30s}] gt=[{truth}] | {row['text'][:60]}")

        eval_time = _time.time() - t0
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        acc = accuracy_score(y_true, y_pred)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"\n{'='*60}")
        print(f"  OVERALL RESULTS ({len(df)} samples, {eval_time:.0f}s)")
        print(f"{'='*60}")
        print(f"  Accuracy:       {acc:.4f}  ({correct}/{len(df)})")
        print(f"  F1 (micro):     {f1_micro:.4f}")
        print(f"  F1 (macro):     {f1_macro:.4f}")
        print(f"  F1 (weighted):  {f1_weighted:.4f}")

        # ---- Per-Class F1 Breakdown ----
        from sklearn.metrics import classification_report
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
            print(f"  {'-'*60}")
            for name, sc in top5:
                print(f"  {name:<35} {sc['f1-score']:>6.3f}  {sc['precision']:>6.3f}  {sc['recall']:>6.3f}  {sc['support']:>4.0f}")

            print(f"\n  BOTTOM-5 WEAKEST INTENTS (by F1):")
            print(f"  {'Intent':<35} {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'N':>4}")
            print(f"  {'-'*60}")
            for name, sc in bot5:
                print(f"  {name:<35} {sc['f1-score']:>6.3f}  {sc['precision']:>6.3f}  {sc['recall']:>6.3f}  {sc['support']:>4.0f}")

        # Save results
        results = {"accuracy": acc, "f1_micro": f1_micro, "f1_macro": f1_macro,
                   "f1_weighted": f1_weighted,
                   "total": len(df), "correct": correct,
                   "eval_time_sec": round(eval_time, 1)}
        out_path = Path(args.config).resolve().parent.parent / "outputs" / "eval_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\n  Results saved -> {out_path}")

    elif args.message:
        # ---- Single prediction ----
        result = clf(args.message)
        print(f"\n  Input:   {args.message}")
        print(f"  Intent:  {result}")

    else:
        # ---- Demo examples ----
        examples = [
            "My card just expired, when will I get a new one?",
            "Why was I charged twice?",
            "How do I change my PIN?",
            "I want to transfer money to another account",
            "What is my current balance?",
        ]
        print("\n  --- Demo examples ---")
        for msg in examples:
            label = clf(msg)
            print(f"  [{label:<30s}] {msg}")

