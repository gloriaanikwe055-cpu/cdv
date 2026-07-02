"""data.py — shared loading + preprocessing (one job: turn a dataframe into X, y).

Used by stages 02, 03, 04, 07 so the split and encoding are defined ONCE
(rule R2). A consistent preprocessor is what makes baseline vs advanced a fair
comparison.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from .config import load_config, root, seed

TARGET = "cvd"


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def split(df: pd.DataFrame):
    cfg = load_config()["split"]
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y,
        test_size=cfg["test_size"],
        stratify=y if cfg["stratify"] else None,
        random_state=seed(),
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # NOTE: several category features (smoker, diabetes, family_history_cvd) arrive
    # as BOOLEAN dtype because YAML parses `[no, yes]` as `[False, True]`. bool is
    # excluded by BOTH np.number and object/category selectors, so it MUST be named
    # explicitly here or those features get silently dropped by the ColumnTransformer.
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = X.select_dtypes(include=["number"]).columns.tolist()
    num = [c for c in num if c not in cat]  # np.number never includes bool, but be explicit
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


def make_pipeline(estimator, X: pd.DataFrame) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(X)), ("model", estimator)])
