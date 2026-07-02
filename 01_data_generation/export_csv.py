"""export_csv.py — lossless CSV export of the Stage 01 synthetic dataset.

The canonical artefact is the parquet written by `cvd.generate`. This helper
just re-serialises those exact rows to CSV so they can be opened in Excel or
shared. It does NOT modify, filter, or regenerate any data — same rows, same
values, different file format.

Run:  python 01_data_generation/export_csv.py
Output: 01_data_generation/output/synthetic_cvd.csv  (+ a copy in the repo root)
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402
from cvd.config import load_config  # noqa: E402


def export() -> Path:
    cfg = load_config()
    parquet = ROOT / cfg["data_generation"]["output"]
    if not parquet.exists():
        raise SystemExit(
            f"Dataset not found: {parquet}\nRun `python -m cvd.run --stage 01` first.")
    df = pd.read_parquet(parquet)

    csv_out = parquet.with_suffix(".csv")
    df.to_csv(csv_out, index=False)

    root_copy = ROOT / "synthetic_cvd_dataset.csv"
    df.to_csv(root_copy, index=False)

    print(f"[export_csv] {len(df)} rows x {df.shape[1]} cols")
    print(f"[export_csv] -> {csv_out}")
    print(f"[export_csv] -> {root_copy}")
    return csv_out


if __name__ == "__main__":
    export()
