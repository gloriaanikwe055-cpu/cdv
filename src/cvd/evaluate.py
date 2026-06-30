"""evaluate.py — Stage 04: comparison + best-model selection (one job).

Reads metrics from stages 02 and 03, builds the baseline-vs-advanced comparison
table the dissertation needs, picks the best model by the configured metric,
freezes it to best_model.pkl, and writes comparison.json + a figure.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_config, root


def _load(stage_out: str) -> dict:
    p = root() / stage_out / "metrics.json"
    with open(p) as f:
        return json.load(f)


def run() -> dict:
    cfg = load_config()
    metric = cfg["evaluation"]["select_best_by"]
    baseline = _load(cfg["baseline_output"])
    advanced = _load(cfg["advanced_output"])

    rows = []
    for tier, group in (("baseline", baseline), ("advanced", advanced)):
        for name, res in group.items():
            row = {"tier": tier, "model": name}
            row.update({f"test_{k}": v for k, v in res["test"].items()})
            rows.append(row)
    table = pd.DataFrame(rows).sort_values(f"test_{metric}", ascending=False)

    out = root() / cfg["evaluation"]["output"]
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison.csv", index=False)
    table.to_json(out / "comparison.json", orient="records", indent=2)

    best = table.iloc[0]
    best_tier_dir = ("02_baseline_models/output" if best["tier"] == "baseline"
                     else "03_advanced_models/output")
    src = root() / best_tier_dir / f"{best['model']}.pkl"
    shutil.copy(src, out / "best_model.pkl")
    joblib.dump({"model": best["model"], "tier": best["tier"],
                 f"test_{metric}": float(best[f"test_{metric}"])},
                out / "best_model_meta.pkl")

    # figure: ROC-AUC by model
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0" if t == "baseline" else "#C44E52" for t in table["tier"]]
    ax.barh(table["model"], table["test_roc_auc"], color=colors)
    ax.set_xlabel("Test ROC-AUC")
    ax.set_title("Baseline vs Advanced — CVD detection")
    ax.set_xlim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(out / "comparison_roc_auc.png", dpi=150)

    print(f"[stage 04] best model: {best['model']} "
          f"({best['tier']}, {metric}={best[f'test_{metric}']:.3f})")
    print(f"[stage 04] frozen -> {out / 'best_model.pkl'}")
    return {"best": best["model"], "table": rows}


if __name__ == "__main__":
    run()
