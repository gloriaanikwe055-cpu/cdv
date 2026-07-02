"""generate.py — Stage 01: synthetic data generation (one job only).

Reads medically-validated ranges, samples class-conditional distributions,
applies probabilistic correlations, writes a parquet dataset + a data_card.json
that labels it synthetic and records the seed (rules R4, R7).

This module does NOT train, score, or plot. It only generates.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from .config import load_config, root, seed


def _load_ranges() -> dict:
    cfg = load_config()
    ranges_path = root() / cfg["data_generation"]["ranges_dir"] / "feature_ranges.yaml"
    with open(ranges_path) as f:
        return yaml.safe_load(f)


def _sample_numeric(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    dist = spec["dist"]
    if dist == "normal":
        x = rng.normal(spec["mean"], spec["sd"], n)
    elif dist == "gamma":
        x = rng.gamma(spec["shape"], spec["scale"], n)
    else:
        raise ValueError(f"Unknown dist '{dist}' (rule R1: fail loud, don't guess)")
    return np.clip(x, spec.get("min", -np.inf), spec.get("max", np.inf))


def _sample_categorical(spec: dict, probs_key: str, n: int, rng: np.random.Generator):
    return rng.choice(spec["values"], size=n, p=spec[probs_key])


def _generate_class(ranges: dict, n: int, cls: str, rng: np.random.Generator) -> pd.DataFrame:
    """cls is 'negative' or 'positive'."""
    cols = {}
    for _group, feats in ranges.items():
        for fname, spec in feats.items():
            if spec["unit"] == "category":
                probs_key = f"{cls}_probs"
                cols[fname] = _sample_categorical(spec, probs_key, n, rng)
            else:
                cols[fname] = _sample_numeric(spec[cls], n, rng)
    return pd.DataFrame(cols)


def _apply_correlations(df: pd.DataFrame, corr: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Nudge correlated features so combinations are physiologically plausible.
    Kept deliberately simple and documented: each rule shifts a target feature
    proportional to a z-scored driver. This is transparent for the viva."""
    def z(col):
        s = df[col].astype(float)
        return (s - s.mean()) / (s.std() + 1e-9)

    if "bmi" in df and "systolic_bp" in df:
        df["systolic_bp"] += corr["bmi_to_bp"] * 10 * z("bmi")
    if "bmi" in df and "fasting_glucose" in df:
        df["fasting_glucose"] += corr["bmi_to_glucose"] * 12 * z("bmi")
    if "age" in df and "systolic_bp" in df:
        df["systolic_bp"] += corr["age_to_bp"] * 8 * z("age")
    return df


def _clip_to_ranges(df: pd.DataFrame, ranges: dict) -> pd.DataFrame:
    """Re-apply the cited clinical bounds after correlations + realism injection.

    Those two steps add/shift values and can push a small fraction of samples
    past their sampled floor/ceiling (e.g. negative alcohol units). Clipping each
    numeric feature back to the union of its per-class [min, max] guarantees every
    value stays inside a range cited in SOURCES.md — no impossible values, and the
    ranges file remains the single source of truth (R2/R7)."""
    for _group, feats in ranges.items():
        for fname, spec in feats.items():
            if spec["unit"] == "category" or fname not in df.columns:
                continue
            lo = min(spec["negative"].get("min", -np.inf),
                     spec["positive"].get("min", -np.inf))
            hi = max(spec["negative"].get("max", np.inf),
                     spec["positive"].get("max", np.inf))
            df[fname] = df[fname].clip(lo, hi)
    return df


def _inject_realism(df: pd.DataFrame, dg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Make synthetic data clinically believable, not perfectly separable.

    Real diagnosis is ambiguous: two patients with similar vitals can differ in
    outcome. Without this, classifiers hit ~0.999 ROC-AUC, which examiners
    rightly distrust. We (a) blend each numeric feature toward the global mean to
    increase class overlap, and (b) flip a fraction of labels to simulate
    diagnostic ambiguity. Both are seed-controlled and documented (R3/R4).
    """
    overlap = float(dg.get("feature_overlap", 0.0))
    noise = float(dg.get("label_noise", 0.0))

    if overlap > 0:
        num_cols = df.select_dtypes(include="number").columns.drop("cvd")
        for c in num_cols:
            global_mean = df[c].mean()
            df[c] = (1 - overlap) * df[c] + overlap * global_mean \
                + rng.normal(0, df[c].std() * overlap * 0.5, len(df))

    if noise > 0:
        flip = rng.random(len(df)) < noise
        df.loc[flip, "cvd"] = 1 - df.loc[flip, "cvd"]

    return df


def generate() -> Path:
    cfg = load_config()
    dg = cfg["data_generation"]
    rng = np.random.default_rng(seed())
    ranges = _load_ranges()

    n_total = dg["n_samples"]
    n_pos = int(round(n_total * dg["positive_prevalence"]))
    n_neg = n_total - n_pos

    neg = _generate_class(ranges, n_neg, "negative", rng)
    neg["cvd"] = 0
    pos = _generate_class(ranges, n_pos, "positive", rng)
    pos["cvd"] = 1

    df = pd.concat([neg, pos], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed()).reset_index(drop=True)
    df = _apply_correlations(df, dg["correlations"], rng)
    df = _inject_realism(df, dg, rng)
    df = _clip_to_ranges(df, ranges)  # no value escapes its cited clinical range

    out = root() / dg["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    card = {
        "synthetic": True,
        "note": "SYNTHETIC DATA — not real patients (rule R7).",
        "generator_version": cfg["project"]["version"],
        "random_seed": seed(),
        "n_samples": int(len(df)),
        "positive_prevalence_actual": float(df["cvd"].mean()),
        "n_features": int(df.shape[1] - 1),
        "feature_names": [c for c in df.columns if c != "cvd"],
        "ranges_source": "01_data_generation/ranges/feature_ranges.yaml + SOURCES.md",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    card_path = root() / dg["data_card"]
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)

    print(f"[stage 01] wrote {len(df)} synthetic rows -> {out}")
    print(f"[stage 01] data card -> {card_path}")
    return out


if __name__ == "__main__":
    generate()
