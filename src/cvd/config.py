"""config.py — load the single source of truth (rule R2).

Every other module reads config through here. Nothing reads config.yaml directly.
"""
from __future__ import annotations
import functools
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def root() -> Path:
    return ROOT


def seed() -> int:
    """Rule R4: one seed governs the whole pipeline."""
    return int(load_config()["random_seed"])
