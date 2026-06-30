# CLAUDE.md — Stage 02: Baseline Models

Inherits root. ONE job: train Logistic Regression + Naive Bayes through the
shared pipeline, CV-score on train, evaluate on held-out test, write metrics.json.

- Cannot start until GATE A (Stage 01 output) exists.
- Uses src/cvd/train_stage.py — do not duplicate training logic here (R1/R2).
- GATE B: metrics.json + per-model .pkl present.
