# Supervisor update — CVD detection pipeline (2026-07-03)

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
| — | 06 GUI | Streamlit predictor + interactive feature dashboard | Works |

**Reproducibility (R4):** the full pipeline was run twice from seed 42; the dataset, all metrics files, the comparison table, the external-validation result and `results.md` are **byte-for-byte identical** across runs. `pytest` = 3 passed.

**Data quality:** every numeric feature is re-clipped to its cited clinical range, so the dataset contains **0 out-of-range values**.

**Feature completeness (fixed):** a dtype bug was silently dropping `smoker`, `diabetes` and `family_history_cvd` from every model (they parsed as boolean and were excluded by the column selector). Now all **17 features** reach the models (23 encoded inputs). Numbers below reflect the corrected models.

## Result 1 — Baseline vs Advanced (held-out synthetic test)

| tier | model | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|---|
| advanced | random_forest (tuned) | 0.8760 | 0.8961 | 0.7246 | 0.8013 | **0.8579** |
| advanced | xgboost (tuned) | 0.8725 | 0.8905 | 0.7188 | 0.7955 | 0.8570 |
| baseline | logistic_regression | 0.8770 | 0.8964 | 0.7275 | 0.8032 | 0.8569 |
| baseline | naive_bayes | 0.8610 | 0.8323 | 0.7478 | 0.7878 | 0.8546 |

- Synthetic ROC-AUC sits in the defensible **0.80–0.92** band by design (label noise 0.10, feature overlap 0.50).
- Advanced models were hyperparameter-tuned via GridSearchCV (5-fold); baselines were not.
- Best model is **random_forest** (advanced), narrowly ahead of the baselines — all four cluster at ROC-AUC ≈ 0.855–0.858.

## Result 2 — Synthetic vs Real gap (UCI Heart Disease, Stage 07)

| metric | synthetic test | real hold-out (UCI, n=303) |
|---|---|---|
| accuracy | 0.8760 | 0.6997 |
| precision | 0.8961 | 0.7222 |
| recall | 0.7246 | 0.5612 |
| f1 | 0.8013 | 0.6316 |
| roc_auc | 0.8579 | **0.7540** |

- **ROC-AUC gap (synthetic − real) = +0.1039.** This is the honesty-gate finding (CLAUDE.md Sec 4), reported not hidden.
- 7 features shared with UCI (age, sex, systolic BP, total cholesterol, ST-depression, max HR, diabetes); 10 features absent from UCI were imputed from the synthetic distribution — a documented limitation carried in `external_metrics.json`.

## Figures / files to open in the meeting

- **Interactive dashboard:** `streamlit run 06_gui/dashboard.py` → http://localhost:8501 (5 tabs: Overview, Feature Explorer, Correlations, Class Separation, Live Risk Predictor)
- `synthetic_cvd_dataset.csv` (project root) — the full 10,000-patient synthetic dataset
- `reports/results.md` — full raw results (test, CV, tuned hyperparameters)
- `04_evaluation/output/comparison_roc_auc.png` — baseline vs advanced bar chart
- `05_explainability/output/shap_summary.png` + `lime_instance.png` — explainability
- `07_external_validation/output/external_metrics.json` — the synthetic→real gap

## How to reproduce live (if asked)

```
$env:PYTHONPATH = "src"
python -m cvd.run --stage all     # regenerates every number above from seed 42
pytest                            # gate + reproducibility tests
streamlit run 06_gui/dashboard.py # interactive feature dashboard
```

## Outstanding (not yet done)

- `ranges/SOURCES.md` still holds placeholder citations — needs the exact editions from the proposal reference list.
- `00_governance/ETHICS.md` ethics path to be confirmed.
- Stage 07 currently validates on UCI Cleveland only; a second real dataset (Kaggle CVD / Framingham) would strengthen the gap analysis.
