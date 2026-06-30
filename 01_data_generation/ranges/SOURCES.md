# SOURCES.md — citations for every medical range

> **Honesty gate (CLAUDE.md Sec 4).** Every threshold in `feature_ranges.yaml`
> traces to a `ref:` key listed here. This file is what lets an examiner verify
> the synthetic data is clinically grounded rather than invented. Replace any
> placeholder DOIs with the exact editions your literature review cites, and
> keep this list aligned with the 20+ references in your proposal.

| ref key        | Source (fill in exact citation from your reference list)                          |
|----------------|------------------------------------------------------------------------------------|
| WHO2023        | World Health Organization (2023) Cardiovascular diseases fact sheet.                |
| GBD2020        | Global Burden of Disease Study — CVD risk-factor incidence by sex/smoking.          |
| ESC2018_HTN    | ESC/ESH Guidelines for the management of arterial hypertension (BP thresholds).     |
| AHA2021        | American Heart Association — heart-rate and max-HR reference values.                 |
| NCEP_ATP3      | NCEP ATP III — total and HDL cholesterol risk thresholds.                           |
| ADA2023        | American Diabetes Association — fasting glucose / diabetes criteria.                 |
| WHO_BMI        | WHO BMI classification (normal / overweight / obese cut-offs).                       |
| WHO_PA2020     | WHO Guidelines on physical activity and sedentary behaviour (2020).                  |
| AHA_Sleep2022  | AHA "Life's Essential 8" — sleep duration as a cardiovascular metric (2022).         |
| Lancet_Alcohol2018 | GBD Alcohol Collaborators (2018), The Lancet — alcohol and CVD risk.            |
| Awasthi2023    | Awasthi et al. (2023), EClinicalMedicine — AI-enabled ECG, ST-depression features.  |

## Method note (put a version of this in the dissertation methodology)

Each feature is sampled from a class-conditional distribution (separate
parameters for CVD-negative and CVD-positive patients), with parameters chosen
to sit inside the cited clinical ranges and to reproduce the directional risk
relationships those sources report (e.g. lower HDL, shorter sleep, and higher
ST-depression in the positive class). Probabilistic correlations defined in
`config.yaml` are then applied so that combinations remain physiologically
plausible (e.g. high BMI co-occurring with elevated blood pressure and glucose)
rather than independently random. Distributional plausibility is checked against
population epidemiology in Stage 01's validation gate.
