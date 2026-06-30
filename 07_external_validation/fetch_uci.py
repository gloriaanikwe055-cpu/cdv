"""fetch_uci.py — acquire the REAL UCI Heart Disease dataset (Stage 07 input).

ONE job: download the processed Cleveland Heart Disease data from the UCI
repository and write it to `data/real_cvd.csv` with proper UCI column headers.

This is acquisition only. The file written here is the read-only real dataset
(rule R8): the validation stage never writes to it and never trains on it. The
column harmonisation that aligns these raw UCI columns to our synthetic feature
names lives in `data/column_map.yaml`, consumed by `cvd.external_validate`.

Run once:
    python 07_external_validation/fetch_uci.py
"""
from __future__ import annotations
import io
import urllib.request
from pathlib import Path
import pandas as pd

# Processed Cleveland data — the canonical 303-row, 14-attribute subset used in
# virtually all UCI Heart Disease ML papers.
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")

# Official attribute order for the processed.*.data files (UCI documentation).
UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

OUT = Path(__file__).resolve().parent / "data" / "real_cvd.csv"


def fetch() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_uci] downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")

    # The file is headerless CSV with '?' for missing values.
    df = pd.read_csv(io.StringIO(raw), header=None, names=UCI_COLUMNS,
                     na_values="?")
    df.to_csv(OUT, index=False)
    print(f"[fetch_uci] wrote {len(df)} real rows -> {OUT}")
    print(f"[fetch_uci] columns: {list(df.columns)}")
    print(f"[fetch_uci] target 'num' distribution:\n{df['num'].value_counts().sort_index()}")
    return OUT


if __name__ == "__main__":
    fetch()
