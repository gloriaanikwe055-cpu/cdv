"""metrics.py — scoring (one job: turn predictions into the proposal's 5 metrics).

Metrics are defined once (proposal Sec 3/5): accuracy, precision, recall, F1,
ROC-AUC. Used by every model-touching stage so numbers are comparable.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import load_config, seed


def score(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def cv_score(pipeline, X, y) -> dict:
    """Stratified k-fold CV (proposal Sec 5). Reproducible via seed."""
    folds = load_config()["split"]["cv_folds"]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed())
    proba = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = score(y, pred, proba)
    out["cv_folds"] = folds
    return out
