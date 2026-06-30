# CLAUDE.md — Machine Learning for Cardiovascular Disease Detection

> **This file is the single source of truth.** Claude Code reads the nearest
> `CLAUDE.md` automatically. Every stage folder also has its own `CLAUDE.md`
> that inherits these rules and narrows them. When a stage rule conflicts with
> this root file, **this root file wins**.

---

## 1. What this project is

A reproducible, explainable machine-learning pipeline that detects
cardiovascular disease (CVD) from **medically-grounded synthetic patient data**,
compares baseline vs. advanced classifiers, and exposes predictions through a
clinician-facing GUI.

This is an **MSc dissertation deliverable**. The approved Project Proposal is
the canonical scope. Do not invent scope beyond it. The proposal commits to:

1. A synthetic dataset built from medically-validated ranges (demographic,
   clinical, lifestyle, ECG-derived features) with probabilistic correlations.
2. Baseline models (Logistic Regression, Naive Bayes) vs. advanced models
   (Random Forest, XGBoost).
3. Evaluation on accuracy, precision, recall, F1, ROC-AUC with stratified
   cross-validation.
4. Explainable AI (SHAP / LIME) for transparency.
5. A user-friendly GUI for clinicians/researchers.

## 2. The pipeline is a state machine (ICM)

Folders are states. Each runs **in order**. A stage may only start when the
previous stage's `output/` contains a valid, gate-passed artefact.

```
01_data_generation   → synthetic_cvd.parquet + data_card.json   [GATE A]
02_baseline_models   → baseline_models.pkl + metrics.json       [GATE B]
03_advanced_models   → advanced_models.pkl + metrics.json       [GATE C]
04_evaluation        → comparison.json + figures/              [GATE D]
05_explainability    → shap_values + plots                      [GATE E]
06_gui               → running app (loads frozen best model)
07_external_validation → real-data hold-out metrics            [GATE F]
```

**One module, one job.** A file that generates data does not also train models.
A file that trains does not also plot. Cross-stage logic lives in `src/cvd/`.

## 3. Non-negotiable engineering rules (apply to every file, every stage)

- **R1 — Truth over hype.** Report real metrics. Never fabricate a number, a
  citation, or a result. If a model underperforms, that is a finding, not a bug
  to hide.
- **R2 — One source of truth.** Config lives in `config.yaml`. Ranges live in
  `01_data_generation/ranges/`. Nothing is hard-coded twice.
- **R3 — Measure, don't assume.** Every claim in the dissertation must trace to
  a file in an `output/` folder that a marker could re-run.
- **R4 — Reproducible from one seed.** `config.yaml` holds `random_seed`. The
  entire pipeline, end to end, must reproduce identical artefacts from that
  seed. This is the project's defence against the "validated against your own
  assumptions" critique. Treat seed-reproducibility as a test that must pass.
- **R5 — Change one thing at a time.** Git checkpoint after every passing gate.
  Commit message names the gate (e.g. `gate A passed: synthetic data v1`).
- **R6 — Secrets only in `.env`.** No keys, no paths-to-private-data committed.
- **R7 — Synthetic data is labelled synthetic.** Every dataset artefact carries
  a `data_card.json` stating it is synthetic, the seed, the generator version,
  and the ranges source. No synthetic row is ever presented as real patient data.
- **R8 — Real data is read-only.** Stage 07's external dataset is never written
  to, never used for training — hold-out validation only.

## 4. The honesty gate that protects the grade

The single biggest examiner attack on a synthetic-data project is:
*"Your model only learned the rules you wrote into the generator."*

We answer it structurally, not rhetorically:

- **Stage 01** documents every range and correlation with a peer-reviewed or
  clinical-guideline citation in `ranges/SOURCES.md`. No magic numbers.
- **Stage 07** validates the frozen best model against **at least one real
  public dataset** (UCI Heart Disease / Kaggle CVD / Framingham-style). A drop
  in performance here is expected and **must be reported honestly** — it is the
  most scientifically valuable result in the dissertation.
- The gap between synthetic-test performance (Gate D) and real-data performance
  (Gate F) is a headline finding, discussed in `reports/`, not buried.

## 5. Decision rule when blocked

Before adding anything not in the proposal, ask: does this serve **(1) finishing
the dissertation, (2) defensible results, (3) reproducibility**? If not, it does
not go in. No extra frameworks, no multi-agent orchestration, no scope creep.

## 6. Tech stack (frozen)

- Python 3.11, `uv` or `pip` + `requirements.txt`
- numpy, pandas, scikit-learn, xgboost, shap, lime, matplotlib, pyyaml, pyarrow
- GUI: Streamlit (fast, clinician-friendly, defensible in a viva)
- Testing: pytest
- One config: `config.yaml`

## 7. How to run (full pipeline)

```bash
pip install -r requirements.txt
python -m cvd.run --stage all          # runs 01→07 in order, honouring gates
python -m cvd.run --stage 01           # run a single stage
pytest                                  # gate tests + seed-reproducibility test
streamlit run 06_gui/app.py            # launch clinician GUI
```

## 8. Definition of done

- [ ] All seven gates pass from a single seed, twice, identically (R4).
- [ ] `ranges/SOURCES.md` cites every threshold (≥ the proposal's references).
- [ ] Comparison table (baseline vs advanced) reproduced in `reports/`.
- [ ] SHAP/LIME plots generated for the best model.
- [ ] GUI loads the frozen model and returns a prediction + explanation.
- [ ] Stage 07 real-data validation run and **honestly reported**.
- [ ] Ethics path resolved (see `00_governance/ETHICS.md`).
