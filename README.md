# Machine Learning for Cardiovascular Disease Detection

MSc dissertation build. A reproducible, explainable ML pipeline that detects
cardiovascular disease (CVD) from **medically-grounded synthetic patient data**,
compares baseline vs. advanced classifiers, explains its predictions, and serves
them through a clinician GUI — then validates the frozen model against **real**
public data.

> Read `CLAUDE.md` first. It is the single source of truth and governs every stage.

## The pipeline (folders are states — ICM)

| Stage | Folder | Job | Gate artefact |
|------|--------|-----|---------------|
| 01 | `01_data_generation/` | Synthesise data from cited clinical ranges | `synthetic_cvd.parquet` + `data_card.json` |
| 02 | `02_baseline_models/` | Logistic Regression, Naive Bayes | `metrics.json` |
| 03 | `03_advanced_models/` | Random Forest, XGBoost | `metrics.json` |
| 04 | `04_evaluation/` | Compare, pick + freeze best model | `best_model.pkl` + `comparison.json` |
| 05 | `05_explainability/` | SHAP / LIME on best model | `shap_summary.png` |
| 06 | `06_gui/` | Streamlit clinician app | running app |
| 07 | `07_external_validation/` | Real-data hold-out (read-only) | `external_metrics.json` |

Stages run in order. A stage is blocked until the previous gate's artefact exists.

## Why this design defends the grade

The standard examiner attack on synthetic-data projects is *"the model only
learned the rules you wrote."* This build answers it structurally:

1. **Every range is cited** (`01_data_generation/ranges/SOURCES.md`).
2. **The whole pipeline is reproducible from one seed** (`config.yaml`, tested).
3. **Stage 07 validates on real data** and reports the synthetic-vs-real gap
   honestly as a headline finding rather than hiding it.

## Quickstart

```bash
pip install -r requirements.txt
python -m cvd.run --stage all       # 01 -> 07, gates enforced
pytest                               # reproducibility + gate tests
streamlit run 06_gui/app.py         # clinician GUI
```

Run a single stage: `python -m cvd.run --stage 01`.

## Mapping to the proposal

- Synthetic dataset from medically-validated ranges + probabilistic correlations → Stage 01
- Baseline vs advanced comparative modelling → Stages 02–04
- accuracy / precision / recall / F1 / ROC-AUC, stratified k-fold CV → `src/cvd/metrics.py`
- Explainable AI (SHAP / LIME) → Stage 05
- Clinician GUI → Stage 06
- (Strengthening) real-data external validation → Stage 07

## Adding the real dataset for Stage 07

1. Put a public CVD CSV at `07_external_validation/data/real_cvd.csv`.
2. Add `07_external_validation/data/column_map.yaml` mapping its columns to our
   feature names and the `cvd` target.
3. `python -m cvd.run --stage 07`.

Without these, Stage 07 records `SKIPPED` honestly and invents no numbers.

## Engineering rules

See `CLAUDE.md` Sec 3 (R1–R8). The short version: truth over hype, one source of
truth, measure don't assume, reproducible from one seed, change one thing at a
time, secrets in `.env`, synthetic labelled synthetic, real data read-only.

## Ethics

See `00_governance/ETHICS.md` — resolves the proposal's blank ethics tick-box
before submission.
