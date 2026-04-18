"""
preprocess_data.py
-------------------
Load the PolyAI/banking77 dataset (raw CSV from GitHub), perform
stratified sampling across all 77 intent classes, and export clean
train/test CSV files.

Uses urllib3 Retry adapter for resilient downloads on flaky networks.
"""

import io
import json
import requests
import pandas as pd
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_SAMPLES_PER_CLASS = 30
TEST_SAMPLES_PER_CLASS = 6
RANDOM_SEED = 42

METADATA_URL = (
    "https://huggingface.co/datasets/PolyAI/banking77"
    "/resolve/main/dataset_infos.json"
)


# ---------------------------------------------------------------------------
# Network helper
# ---------------------------------------------------------------------------

def _robust_session():
    """Return a requests.Session with automatic retry on network errors."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _download_text(url: str) -> str:
    """Download text content with retry."""
    session = _robust_session()
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def fetch_metadata():
    """Download dataset_infos.json and extract CSV URLs + label list."""
    print("[1/5] Fetching metadata from Hugging Face …")
    raw = _download_text(METADATA_URL)
    info = json.loads(raw)["default"]

    labels = info["features"]["label"]["names"]
    urls = list(info.get("download_checksums", {}).keys())

    train_url = next(u for u in urls if "train" in u)
    test_url = next(u for u in urls if "test" in u)

    print(f"      → {len(labels)} intent classes found.")
    return labels, train_url, test_url


def download_csv(url: str, step: str) -> pd.DataFrame:
    """Download a CSV split into a DataFrame."""
    print(f"[{step}] Downloading {url.split('/')[-1]} …")
    csv_text = _download_text(url)
    return pd.read_csv(io.StringIO(csv_text))


def stratified_sample(df: pd.DataFrame, group_col: str, n_per_class: int):
    """Take exactly n_per_class samples per group (or fewer if not enough)."""
    parts = []
    for _, group in df.groupby(group_col):
        parts.append(group.sample(n=min(len(group), n_per_class),
                                  random_state=RANDOM_SEED))
    return pd.concat(parts, ignore_index=True)


def process_and_sample(labels, train_url, test_url):
    """Download → detect columns → stratified sample → clean output."""
    df_train = download_csv(train_url, "2/5")
    df_test = download_csv(test_url, "3/5")

    # The raw CSV has columns: ['text', 'category']
    # 'category' is already the string intent name
    print(f"      Detected columns: {list(df_train.columns)}")

    # Determine which column is the label
    label_col = "category" if "category" in df_train.columns else "label"
    text_col = "text"

    print("[4/5] Stratified sampling …")
    df_train = stratified_sample(df_train, label_col, TRAIN_SAMPLES_PER_CLASS)
    df_test = stratified_sample(df_test, label_col, TEST_SAMPLES_PER_CLASS)

    # If label is already a string, just rename to 'intent'
    # If label is an integer ID, map it
    if pd.api.types.is_integer_dtype(df_train[label_col]):
        df_train["intent"] = df_train[label_col].apply(lambda x: labels[int(x)])
        df_test["intent"] = df_test[label_col].apply(lambda x: labels[int(x)])
    else:
        df_train = df_train.rename(columns={label_col: "intent"})
        df_test = df_test.rename(columns={label_col: "intent"})

    # Keep only clean columns
    df_train = df_train[[text_col, "intent"]]
    df_test = df_test[[text_col, "intent"]]

    return df_train, df_test


def save_outputs(df_train, df_test, labels, output_dir: Path):
    """Save CSVs and label map."""
    print("[5/5] Saving to sample_data/ …")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(output_dir / "train.csv", index=False, encoding="utf-8")
    df_test.to_csv(output_dir / "test.csv", index=False, encoding="utf-8")

    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

    print(f"      train.csv     — {df_train.shape}")
    print(f"      test.csv      — {df_test.shape}")
    print(f"      label_map.json — {len(labels)} labels")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root_dir = Path(__file__).resolve().parent.parent
    output_dir = root_dir / "sample_data"

    labels, train_url, test_url = fetch_metadata()
    df_train, df_test = process_and_sample(labels, train_url, test_url)
    save_outputs(df_train, df_test, labels, output_dir)

    print("\n========== Summary ==========")
    print(f"  Train : {len(df_train)} rows  ({df_train['intent'].nunique()} classes)")
    print(f"  Test  : {len(df_test)} rows  ({df_test['intent'].nunique()} classes)")
    print(f"  Cols  : {list(df_train.columns)}")
    print("=============================")


if __name__ == "__main__":
    main()