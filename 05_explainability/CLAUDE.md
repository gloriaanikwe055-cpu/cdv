# CLAUDE.md — Stage 05: Explainability

Inherits root. ONE job: SHAP global importance (+ LIME for single instances) on
the frozen best model. Answers the proposal's explainable-AI objective.

- Cannot start until GATE D exists.
- If SHAP fails, report honestly + fall back to feature_importances_ (R1).
