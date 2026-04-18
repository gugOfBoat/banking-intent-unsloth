"""
preprocess_data.py
-------------------
Load the PolyAI/banking77 dataset from Hugging Face, perform stratified
sampling across all 77 intent classes, map numeric labels to their string
names, and export clean train/test CSV files.
"""

import os
import pandas as pd
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_SAMPLES_PER_CLASS = 30
TEST_SAMPLES_PER_CLASS = 6
RANDOM_SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_data")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_banking77():
    """Download the PolyAI/banking77 dataset and return both splits."""
    print("[INFO] Loading PolyAI/banking77 from Hugging Face …")
    ds = load_dataset("PolyAI/banking77")
    return ds


def build_label_map(ds):
    """Extract the id-to-string mapping from the dataset features."""
    label_names = ds["train"].features["label"].names
    label_map = {idx: name for idx, name in enumerate(label_names)}
    print(f"[INFO] Found {len(label_map)} intent classes.")
    return label_map


def process_and_sample_data(ds, label_map):
    """
    Merge train+test splits into one pool, map numeric labels to intent
    strings, then perform stratified sampling to create balanced subsets.

    Returns
    -------
    train_df, test_df : pd.DataFrame
        Each contains columns ``text`` and ``intent``.
    """
    # Combine all available rows into a single pool
    frames = []
    for split_name in ds.keys():
        split_df = ds[split_name].to_pandas()
        frames.append(split_df)

    pool = pd.concat(frames, ignore_index=True)
    pool["intent"] = pool["label"].map(label_map)
    pool = pool[["text", "intent"]]
    print(f"[INFO] Total pool size: {len(pool)} rows across {pool['intent'].nunique()} classes.")

    # Stratified sampling
    total_needed = TRAIN_SAMPLES_PER_CLASS + TEST_SAMPLES_PER_CLASS

    sampled = (
        pool
        .groupby("intent", group_keys=False)
        .apply(lambda g: g.sample(n=min(total_needed, len(g)), random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )

    # Split the sampled pool per class
    train_parts, test_parts = [], []
    for intent, group in sampled.groupby("intent"):
        train_parts.append(group.iloc[:TRAIN_SAMPLES_PER_CLASS])
        test_parts.append(group.iloc[TRAIN_SAMPLES_PER_CLASS:TRAIN_SAMPLES_PER_CLASS + TEST_SAMPLES_PER_CLASS])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, test_df


def save_csv(df, filepath):
    """Save a DataFrame to CSV with utf-8 encoding."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"[INFO] Saved {filepath}  —  shape {df.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ds = load_banking77()
    label_map = build_label_map(ds)
    train_df, test_df = process_and_sample_data(ds, label_map)

    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    save_csv(train_df, train_path)
    save_csv(test_df, test_path)

    print("\n========== Summary ==========")
    print(f"  Train samples : {len(train_df)}  ({train_df['intent'].nunique()} classes)")
    print(f"  Test  samples : {len(test_df)}  ({test_df['intent'].nunique()} classes)")
    print("  Columns       :", list(train_df.columns))
    print("=============================")


if __name__ == "__main__":
    main()
