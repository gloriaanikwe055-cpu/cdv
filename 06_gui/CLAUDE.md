# CLAUDE.md — Stage 06: GUI

Inherits root. ONE job: Streamlit app that loads the FROZEN best model and
returns a prediction + plain-language explanation for clinicians.

- Loads model from Stage 04 output. Never trains. Never sees raw data.
- Must display the "synthetic data / not for clinical use" disclaimer.
