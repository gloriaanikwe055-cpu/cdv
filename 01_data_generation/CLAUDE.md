# CLAUDE.md — Stage 01: Data Generation

Inherits root CLAUDE.md. This stage has ONE job: produce synthetic_cvd.parquet +
data_card.json from medically-validated ranges. It does not train or evaluate.

- Ranges live in `ranges/feature_ranges.yaml`; every threshold cited in `ranges/SOURCES.md` (honesty gate).
- Output must carry a data_card.json marked synthetic with the seed (R7, R4).
- GATE A: stage is "done" only when output parquet AND data_card.json exist.
- Edit ranges or correlations here; never hard-code numbers in generate.py (R2).
