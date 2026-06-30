"""test_pipeline.py — gate + reproducibility tests (rule R4).

The seed-reproducibility test is the most important: it proves the whole
pipeline is deterministic, which is the project's defence against the
"validated against your own assumptions" critique.
"""
import sys
from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cvd.generate import generate  # noqa: E402
from cvd.config import load_config  # noqa: E402


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generation_runs_and_labels_synthetic():
    out = generate()
    assert out.exists()
    card = ROOT / load_config()["data_generation"]["data_card"]
    import json
    meta = json.loads(card.read_text())
    assert meta["synthetic"] is True          # rule R7
    assert meta["random_seed"] == load_config()["random_seed"]


def test_seed_reproducibility():
    """Rule R4: two runs from the same seed produce identical data."""
    p1 = generate()
    h1 = _hash(p1)
    p2 = generate()
    h2 = _hash(p2)
    assert h1 == h2, "Pipeline not reproducible from seed — R4 violated."


def test_class_balance_reasonable():
    df = pd.read_parquet(ROOT / load_config()["data_generation"]["output"])
    prev = df["cvd"].mean()
    assert 0.2 < prev < 0.4, f"Prevalence {prev} outside plausible band."
