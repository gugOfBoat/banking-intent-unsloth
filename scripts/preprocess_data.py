"""
preprocess_data.py
-------------------
Load PolyAI/banking77 raw CSVs and export train / val / test splits.

Mode: FULL DATASET
  - Val: 10 per class | Test: 10 per class | Train: ALL REMAINING
  - This yields ~9,200 train / 770 val / 770 test samples
Output columns  →  text, intent   (no prompt formatting)
"""

import io
import json
import requests
import pandas as pd
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VAL_PER_CLASS   = 10
TEST_PER_CLASS  = 10
RANDOM_SEED     = 42

METADATA_URL = (
    "https://huggingface.co/datasets/PolyAI/banking77"
    "/resolve/main/dataset_infos.json"
)

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def _session():
    s = requests.Session()
    r = Retry(total=5, backoff_factor=2,
              status_forcelist=[429, 500, 502, 503, 504],
              allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s


def _get(url: str) -> str:
    return _session().get(url, timeout=60).text

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def fetch_metadata():
    print("[1/5] Fetching metadata …")
    info = json.loads(_get(METADATA_URL))["default"]
    labels = info["features"]["label"]["names"]
    urls   = list(info["download_checksums"].keys())
    train_url = next(u for u in urls if "train" in u)
    test_url  = next(u for u in urls if "test"  in u)
    print(f"      {len(labels)} intent classes found.")
    return labels, train_url, test_url


def load_csv(url: str, tag: str) -> pd.DataFrame:
    print(f"[{tag}] Downloading {url.split('/')[-1]}")
    return pd.read_csv(io.StringIO(_get(url)))


def stratified_sample(df: pd.DataFrame, group_col: str, n: int) -> pd.DataFrame:
    parts = [
        g.sample(min(len(g), n), random_state=RANDOM_SEED)
        for _, g in df.groupby(group_col)
    ]
    return pd.concat(parts, ignore_index=True)


def process(labels, train_url, test_url):
    raw_train = load_csv(train_url, "2/5")
    raw_test  = load_csv(test_url,  "3/5")

    label_col = "category" if "category" in raw_train.columns else "label"
    text_col  = "text"
    print(f"      Columns: {list(raw_train.columns)}")

    # Pool all rows, reset to a clean 0-N index
    pool = (
        pd.concat([raw_train, raw_test], ignore_index=True)
        .drop_duplicates(subset=[text_col])
        .reset_index(drop=True)
    )

    train_parts, val_parts, test_parts = [], [], []

    print("[4/5] Full-dataset split (holdout val+test, rest -> train) ...")
    for _, group in pool.groupby(label_col):
        g = group.sample(frac=1, random_state=RANDOM_SEED)  # shuffle group
        # Reserve fixed holdout for val & test, everything else → train
        val_parts.append(g.iloc[:VAL_PER_CLASS])
        test_parts.append(g.iloc[VAL_PER_CLASS: VAL_PER_CLASS + TEST_PER_CLASS])
        train_parts.append(g.iloc[VAL_PER_CLASS + TEST_PER_CLASS:])

    def collect(parts):
        df = pd.concat(parts, ignore_index=True)
        if label_col == "category":
            df = df.rename(columns={"category": "intent"})
        else:
            df = df.copy()
            df["intent"] = df[label_col].apply(lambda x: labels[int(x)])
        return df[[text_col, "intent"]].rename(columns={text_col: "text"})

    return collect(train_parts), collect(val_parts), collect(test_parts)



def save(train_df, val_df, test_df, labels, out_dir: Path):
    print("[5/5] Saving …")
    out_dir.mkdir(parents=True, exist_ok=True)
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
        print(f"      {name}.csv — {df.shape}  ({df['intent'].nunique()} classes)")
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent.parent
    labels, train_url, test_url = fetch_metadata()
    train_df, val_df, test_df   = process(labels, train_url, test_url)
    save(train_df, val_df, test_df, labels, root / "sample_data")

    print("\n========== Summary ==========")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"  {name:<6}: {len(df):>4} rows | {df['intent'].nunique()} classes")
    print("=============================")


if __name__ == "__main__":
    main()