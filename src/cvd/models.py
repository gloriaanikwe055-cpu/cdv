"""models.py — model factory + training (one job: build and fit estimators).

Baseline and advanced estimators are defined here so both stages train through
the SAME code path with the SAME preprocessing and seed — a fair comparison
(rules R2, R4).
"""
from __future__ import annotations
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from .config import seed

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False


def build_estimator(name: str):
    s = seed()
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=s)
    if name == "naive_bayes":
        return GaussianNB()
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1)
    if name == "xgboost":
        if not _HAS_XGB:
            raise RuntimeError("xgboost not installed — pip install xgboost")
        return XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=s, n_jobs=-1,
        )
    raise ValueError(f"Unknown model '{name}' (rule R1: fail loud)")
