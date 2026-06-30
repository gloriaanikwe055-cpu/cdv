"""explain.py — Stage 05: explainability (one job: make the model interpretable).

Generates SHAP global importance AND a LIME single-instance explanation for the
frozen best model. Directly answers the proposal's explainable-AI objective and
the "black box" gap in the literature review.

If SHAP fails it falls back to the model's feature_importances_ and records the
failure honestly (R1). LIME explains one held-out instance the way the GUI would.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_config, root
from .data import load_dataset, split


def _to_dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def _shap(model, X_te_t, feat_names, out) -> bool:
    import shap
    sample = X_te_t[:500]
    explainer = shap.Explainer(model, sample)
    sv = explainer(sample)
    plt.figure()
    shap.summary_plot(sv, sample, feature_names=list(feat_names), show=False)
    plt.tight_layout()
    plt.savefig(out / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    # bar (mean |SHAP|) for an unambiguous global ranking
    plt.figure()
    shap.summary_plot(sv, sample, feature_names=list(feat_names),
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[stage 05] SHAP summary -> {out / 'shap_summary.png'}")
    return True


def _lime(model, X_tr_t, X_te_t, feat_names, out) -> bool:
    from lime.lime_tabular import LimeTabularExplainer
    explainer = LimeTabularExplainer(
        training_data=_to_dense(X_tr_t),
        feature_names=list(feat_names),
        class_names=["no_cvd", "cvd"],
        mode="classification",
        random_state=load_config()["random_seed"],
    )
    instance = _to_dense(X_te_t)[0]
    exp = explainer.explain_instance(
        instance, lambda a: model.predict_proba(a), num_features=12)
    fig = exp.as_pyplot_figure()
    fig.tight_layout()
    fig.savefig(out / "lime_instance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    exp.save_to_file(str(out / "lime_instance.html"))
    print(f"[stage 05] LIME instance -> {out / 'lime_instance.png'}")
    return True


def run() -> Path:
    cfg = load_config()
    out = root() / cfg["explainability"]["output"]
    out.mkdir(parents=True, exist_ok=True)

    pipe = joblib.load(root() / cfg["evaluation"]["output"] / "best_model.pkl")
    df = load_dataset(root() / cfg["data_generation"]["output"])
    X_tr, X_te, y_tr, y_te = split(df)

    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    X_tr_t = _to_dense(prep.transform(X_tr))
    X_te_t = _to_dense(prep.transform(X_te))
    try:
        feat_names = prep.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(X_te_t.shape[1])]

    # SHAP (global)
    try:
        _shap(model, X_te_t, feat_names, out)
    except Exception as e:  # report honestly, don't fake it (R1)
        (out / "shap_ERROR.txt").write_text(
            f"SHAP failed: {e}\nFalling back to model feature_importances_ if available.")
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feat_names)
            imp.sort_values().tail(15).plot.barh(figsize=(7, 5))
            plt.title("Feature importance (fallback)")
            plt.tight_layout()
            plt.savefig(out / "feature_importance.png", dpi=150)
            plt.close()
            print(f"[stage 05] SHAP failed; wrote feature_importances_ fallback.")

    # LIME (single instance)
    try:
        _lime(model, X_tr_t, X_te_t, feat_names, out)
    except Exception as e:  # report honestly (R1)
        (out / "lime_ERROR.txt").write_text(f"LIME failed: {e}")
        print(f"[stage 05] LIME failed: {e}")

    return out


if __name__ == "__main__":
    run()
