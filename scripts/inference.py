"""
inference.py — Standalone inference for banking intent classification.

Section 2.3 Requirements:
  - IntentClassification class with __init__ and __call__
  - __init__: loads config, tokenizer, and model checkpoint
  - __call__: receives input message, returns predicted label
  - model_path points to a config file with checkpoint path

Usage:
    from scripts.inference import IntentClassification

    clf = IntentClassification("configs/inference.yaml")
    label = clf("I want to top up my account")
    print(label)  # → "top_up"
"""

import yaml
from pathlib import Path


class IntentClassification:
    """Banking intent classifier powered by a fine-tuned Qwen model (Unsloth)."""

    def __init__(self, model_path: str):
        """
        Load the configuration file, tokenizer, and model checkpoint.

        Parameters
        ----------
        model_path : str
            Path to a YAML configuration file that contains at least:
              - model_checkpoint: path to saved model directory
              - max_new_tokens: (optional) max tokens to generate, default 32
        """
        # Load config
        config_path = Path(model_path)
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        checkpoint    = self.config["model_checkpoint"]
        max_new_tokens = self.config.get("max_new_tokens", 32)
        self.max_new_tokens = max_new_tokens

        # Load model & tokenizer via Unsloth (4-bit inference)
        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )

        # Switch to inference mode (2x faster, 50% less VRAM)
        FastLanguageModel.for_inference(self.model)

        print(f"[INFO] Model loaded from: {checkpoint}")

    def __call__(self, message: str) -> str:
        """
        Predict the intent label for a single text message.

        Parameters
        ----------
        message : str
            The user's banking query, e.g. "I want to top up my account"

        Returns
        -------
        predicted_label : str
            The predicted intent class name, e.g. "top_up"
        """
        # Build ChatML prompt (same format used during training)
        messages = [
            {"role": "user", "content": f"Classify the banking intent: {message}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # adds <|im_start|>assistant\n
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,           # greedy for classification
            temperature=1.0,
            use_cache=True,
        )

        # Decode only the newly generated tokens (skip the prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        predicted_label = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return predicted_label


# ---------------------------------------------------------------------------
# Standalone usage example (Section 2.3: provide a short usage example)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run intent classification inference")
    parser.add_argument("--config", default="configs/inference.yaml",
                        help="Path to inference config YAML")
    parser.add_argument("--message", default="I want to top up my account",
                        help="Input message to classify")
    args = parser.parse_args()

    # Initialize
    clf = IntentClassification(args.config)

    # Predict
    result = clf(args.message)
    print(f"\n  Input:   {args.message}")
    print(f"  Intent:  {result}")

    # Test with a few more examples
    test_messages = [
        "My card just expired, when will I get a new one?",
        "Why was I charged twice?",
        "How do I change my PIN?",
    ]
    print("\n  --- Additional examples ---")
    for msg in test_messages:
        label = clf(msg)
        print(f"  [{label}] {msg}")
