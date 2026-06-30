"""external_validate.py — Stage 07: real-data hold-out validation (one job).

This is the project's scientific backbone (CLAUDE.md Sec 4). The frozen best
model — trained ONLY on synthetic data — is scored against a REAL public dataset
it has never seen. Real data is read-only and never trains anything (rule R8).

A performance drop here is EXPECTED and must be reported honestly. The gap
between Gate D (synthetic) and Gate F (real) is the dissertation's headline
finding, not a failure to hide (rule R1).

Because every real dataset has different column names AND encodings, this module
reads `column_map.yaml`, which declares three things:
  columns: <real column> -> <our feature name>
  values:  per-feature value remaps (e.g. UCI sex 1/0 -> male/female)
  target:  how to binarise the real label column

The real UCI Heart Disease set only shares a SUBSET of our 17 synthetic
features. Features it does not provide are imputed from the synthetic training
distribution (numeric -> median, categorical -> mode) so the frozen pipeline can
score every real row. Exactly which features were imputed is recorded in the
output — this is a documented limitation, not a hidden assumption.

If the real dataset or mapping is absent, the stage records that honestly and
skips — it does not invent numbers.
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
import yaml

from .config import load_config, root
from .data import load_dataset, TARGET
from .metrics import score


def _impute_stats(synth: pd.DataFrame) -> dict:
    """Per-feature fill values from the SYNTHETIC training distribution."""
    feats = synth.drop(columns=[TARGET])
    stats = {}
    for col in feats.columns:
        s = feats[col]
        if pd.api.types.is_numeric_dtype(s):
            stats[col] = {"kind": "numeric", "fill": float(s.median())}
        else:
            stats[col] = {"kind": "categorical", "fill": str(s.mode().iloc[0])}
    return stats


def _harmonise(real: pd.DataFrame, colmap: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Rename, value-remap, and binarise the real dataset per column_map.yaml.

    Returns (features_df_with_mapped_columns_only, binary_target)."""
    columns = colmap["columns"]
    values = colmap.get("values", {}) or {}
    target = colmap["target"]

    df = real.rename(columns=columns)
    # Keep only the columns we explicitly mapped (drop unmapped UCI columns).
    keep = [c for c in columns.values() if c in df.columns]
    df = df[keep].copy()

    # Value remaps (e.g. sex 1/0 -> male/female). Keys may be int while the
    # real column is float/int — match on int where possible.
    for feat, mapping in values.items():
        if feat not in df.columns:
            continue
        norm = {}
        for k, v in mapping.items():
            norm[k] = v
            try:
                norm[int(k)] = v
                norm[float(k)] = v
            except (TypeError, ValueError):
                pass
        df[feat] = df[feat].map(lambda x: norm.get(x, norm.get(_as_int(x), x)))

    # Target binarisation.
    tcol = target["column"]
    gt = target.get("positive_when_gt", 0)
    y = (pd.to_numeric(df[tcol], errors="coerce") > gt).astype(int)
    X = df.drop(columns=[tcol])
    return X, y


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def run() -> dict:
    cfg = load_config()
    ev = cfg["external_validation"]
    out = root() / ev["output"]
    out.mkdir(parents=True, exist_ok=True)
    data_path = root() / ev["data_path"]
    map_path = data_path.parent / "column_map.yaml"

    if not data_path.exists() or not map_path.exists():
        msg = {
            "status": "SKIPPED",
            "reason": "Real dataset and/or column_map.yaml not present.",
            "how_to_enable": (
                f"Run `python 07_external_validation/fetch_uci.py` to place a real "
                f"CVD CSV at {ev['data_path']}, with a column_map.yaml mapping its "
                "columns to our feature names + 'cvd' target."),
            "honesty_note": "No numbers fabricated (rule R1).",
        }
        (out / "external_metrics.json").write_text(json.dumps(msg, indent=2))
        print("[stage 07] SKIPPED — real dataset not provided. Recorded honestly.")
        return msg

    real = pd.read_csv(data_path)
    colmap = yaml.safe_load(open(map_path))
    X_real, y = _harmonise(real, colmap)

    # Drop real rows whose mapped feature/target is unusable.
    valid = y.notna()
    X_real, y = X_real[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    # Align to the model's expected feature set: impute features the real data
    # does not provide, from the synthetic training distribution.
    synth = load_dataset(root() / cfg["data_generation"]["output"])
    stats = _impute_stats(synth)
    expected = [c for c in synth.columns if c != TARGET]

    shared, imputed = [], []
    X = pd.DataFrame(index=X_real.index)
    for col in expected:
        if col in X_real.columns:
            s = X_real[col]
            if stats[col]["kind"] == "numeric":
                s = pd.to_numeric(s, errors="coerce")
                if s.isna().any():
                    s = s.fillna(stats[col]["fill"])
            else:
                s = s.where(s.notna(), stats[col]["fill"]).astype(str)
            X[col] = s
            shared.append(col)
        else:
            X[col] = stats[col]["fill"]
            imputed.append(col)

    pipe = joblib.load(root() / cfg["evaluation"]["output"] / "best_model.pkl")
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    real_metrics = score(y, pred, proba)

    # load synthetic test metrics for the headline gap
    comp = json.loads((root() / cfg["evaluation"]["output"] / "comparison.json").read_text())
    best = comp[0]
    result = {
        "status": "DONE",
        "dataset": ev["dataset_name"],
        "real_n_rows": int(len(y)),
        "real_positive_prevalence": float(y.mean()),
        "shared_features": shared,
        "imputed_features": imputed,
        "imputed_note": (
            f"{len(imputed)} of {len(expected)} features absent from the real "
            "dataset were imputed from the synthetic training distribution "
            "(numeric=median, categorical=mode). Documented limitation (R1)."),
        "best_model": best["model"],
        "synthetic_test": {k: best[f"test_{k}"] for k in cfg["evaluation"]["metrics"]},
        "real_holdout": real_metrics,
        "roc_auc_gap": float(best["test_roc_auc"] - real_metrics["roc_auc"]),
        "interpretation": (
            "Gap = synthetic minus real ROC-AUC. A positive gap quantifies "
            "optimism from training on self-generated data. Discuss in reports/."),
    }
    (out / "external_metrics.json").write_text(json.dumps(result, indent=2))
    print(f"[stage 07] real ROC-AUC={real_metrics['roc_auc']:.3f} "
          f"(gap {result['roc_auc_gap']:+.3f}); "
          f"{len(shared)} shared / {len(imputed)} imputed features")
    return result


if __name__ == "__main__":
    run()
