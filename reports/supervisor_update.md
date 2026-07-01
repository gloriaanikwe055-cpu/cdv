# Supervisor update — CVD detection pipeline (2026-07-02)

*Status report only. Not dissertation text. Every number below is reproduced from a file the pipeline regenerates.*

## Headline status: all 7 ICM gates pass, end-to-end, reproducibly

| Gate | Stage | Artefact | Status |
|---|---|---|---|
| A | 01 Data generation | `synthetic_cvd.parquet` + `data_card.json` (10,000 rows, seed 42) | PASS |
| B | 02 Baselines | LogReg + NaiveBayes `metrics.json` | PASS |
| C | 03 Advanced (tuned) | RandomForest + XGBoost `metrics.json` | PASS |
| D | 04 Evaluation | `comparison.json` + frozen `best_model.pkl` | PASS |
| E | 05 Explainability | SHAP + LIME plots | PASS |
| F | 07 External validation | real UCI hold-out `external_metrics.json` | PASS |
| — | 06 GUI | Streamlit app loads frozen model, returns prediction | Works (smoke-tested) |

**Reproducibility (R4):** the full pipeline was run twice from seed 42; the dataset, all metrics files, the comparison table, the external-validation result and `results.md` are **byte-for-byte identical** across runs. `pytest` = 3 passed.

## Result 1 — Baseline vs Advanced (held-out synthetic test)

| tier | model | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|---|
| baseline | logistic_regression | 0.8745 | 0.8941 | 0.7217 | 0.7987 | **0.8570** |
| advanced | xgboost (tuned) | 0.8720 | 0.8931 | 0.7145 | 0.7939 | 0.8567 |
| baseline | naive_bayes | 0.8780 | 0.8805 | 0.7478 | 0.8088 | 0.8565 |
| advanced | random_forest (tuned) | 0.8720 | 0.8931 | 0.7145 | 0.7939 | 0.8564 |

- Synthetic ROC-AUC sits in the defensible **0.80–0.92** band by design (label noise 0.10, feature overlap 0.50).
- Advanced models were hyperparameter-tuned via GridSearchCV (5-fold); baselines were not.
- Observed result: advanced models do **not** outperform baselines here — all four cluster at ROC-AUC ≈ 0.856–0.857. Reported as-is (R1).

## Result 2 — Synthetic vs Real gap (UCI Heart Disease, Stage 07)

| metric | synthetic test | real hold-out (UCI, n=303) |
|---|---|---|
| accuracy | 0.8745 | 0.6997 |
| precision | 0.8941 | 0.6818 |
| recall | 0.7217 | 0.6475 |
| f1 | 0.7987 | 0.6642 |
| roc_auc | 0.8570 | **0.7597** |

- **ROC-AUC gap (synthetic − real) = +0.097.** This is the honesty-gate finding (CLAUDE.md Sec 4), reported not hidden.
- 7 features shared with UCI (age, sex, systolic BP, total cholesterol, ST-depression, max HR, diabetes); 10 features absent from UCI were imputed from the synthetic distribution — a documented limitation carried in `external_metrics.json`.

## Figures / files to open in the meeting

- `reports/results.md` — full raw results (also CV scores + tuned hyperparameters)
- `04_evaluation/output/comparison_roc_auc.png` — baseline vs advanced bar chart
- `05_explainability/output/shap_summary.png` and `shap_bar.png` — global feature importance
- `05_explainability/output/lime_instance.png` — single-patient explanation
- `07_external_validation/output/external_metrics.json` — the synthetic→real gap

## How to reproduce live (if asked)

```
$env:PYTHONPATH = "src"
python -m cvd.run --stage all     # regenerates every number above from seed 42
pytest                            # gate + reproducibility tests
streamlit run 06_gui/app.py       # clinician GUI
```

## Outstanding (not yet done)

- `ranges/SOURCES.md` still holds placeholder citations — needs the exact editions from the proposal reference list.
- `00_governance/ETHICS.md` ethics path to be confirmed.
- Stage 07 currently validates on UCI Cleveland only; a second real dataset (Kaggle CVD / Framingham) would strengthen the gap analysis.
