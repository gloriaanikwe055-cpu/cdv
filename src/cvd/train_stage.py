"""train_stage.py — shared trainer for baseline (02) and advanced (03) stages.

One job: given a list of model names, fit each through the shared pipeline,
CV-score on train, evaluate on the held-out test set, persist fitted pipelines
and metrics.json. Stages 02 and 03 call this with different model lists so the
two stages are identical except for which estimators they run (rule R1/R2).

Advanced models are hyperparameter-tuned via GridSearchCV when a grid is declared
for them in config.yaml `tuning.grids` (proposal Sec 5). Baselines have no grid
and train at their defaults. The tuning CV and the reported CV both use the
seed-controlled StratifiedKFold, so the whole stage stays reproducible (R4).
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .config import load_config, root, seed
from .data import load_dataset, split, make_pipeline
from .models import build_estimator
from .metrics import score, cv_score


def _tune(pipe, grid, X_tr, y_tr):
    """GridSearchCV over `grid`; returns (best_pipeline, best_params, best_cv)."""
    cfg = load_config()["tuning"]
    skf = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed())
    gs = GridSearchCV(pipe, grid, scoring=cfg["scoring"], cv=skf, n_jobs=-1, refit=True)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)


def run(model_names: list[str], out_dir: str) -> dict:
    cfg = load_config()
    tuning = cfg.get("tuning", {})
    grids = tuning.get("grids", {}) if tuning.get("enabled") else {}

    df = load_dataset(root() / cfg["data_generation"]["output"])
    X_tr, X_te, y_tr, y_te = split(df)

    out = root() / out_dir
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    for name in model_names:
        pipe = make_pipeline(build_estimator(name), X_tr)
        best_params, best_search_score = None, None

        if name in grids:
            pipe, best_params, best_search_score = _tune(pipe, grids[name], X_tr, y_tr)
            print(f"[train] {name}: tuned best CV {cfg['tuning']['scoring']}="
                  f"{best_search_score:.3f} params={best_params}")
        else:
            pipe.fit(X_tr, y_tr)

        # Honest, comparable CV on train (same procedure for every model).
        cv = cv_score(pipe, X_tr, y_tr)
        # best_estimator_ from GridSearchCV is already fit on full train; baselines
        # were fit above. Score the held-out test set.
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        test = score(y_te, pred, proba)

        joblib.dump(pipe, out / f"{name}.pkl")
        results[name] = {"cv": cv, "test": test}
        if best_params is not None:
            results[name]["tuning"] = {
                "scoring": cfg["tuning"]["scoring"],
                "best_cv_score": best_search_score,
                "best_params": best_params,
            }
        print(f"[train] {name}: test ROC-AUC={test['roc_auc']:.3f} "
              f"F1={test['f1']:.3f}")

    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    return results
