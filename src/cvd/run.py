"""run.py — pipeline orchestrator (the state machine driver).

Runs stages 01→07 in order. Each stage is gated: a stage only proceeds if the
previous stage's required artefact exists (ICM enforcement, CLAUDE.md Sec 2).

Usage:
    python -m cvd.run --stage all
    python -m cvd.run --stage 01
"""
from __future__ import annotations
import argparse
from pathlib import Path

from .config import load_config, root


def _exists(rel: str) -> bool:
    return (root() / rel).exists()


def gate(name: str, artefact: str):
    if not _exists(artefact):
        raise SystemExit(
            f"[GATE {name}] BLOCKED: required artefact missing: {artefact}\n"
            f"Run the prior stage first (ICM: stages run in order).")
    print(f"[GATE {name}] passed.")


def stage_01():
    from .generate import generate
    generate()


def stage_02():
    gate("A", load_config()["data_generation"]["output"])
    from .train_stage import run
    run(load_config()["baseline_models"], load_config()["baseline_output"])


def stage_03():
    gate("A", load_config()["data_generation"]["output"])
    from .train_stage import run
    run(load_config()["advanced_models"], load_config()["advanced_output"])


def stage_04():
    gate("B", "02_baseline_models/output/metrics.json")
    gate("C", "03_advanced_models/output/metrics.json")
    from .evaluate import run
    run()


def stage_05():
    gate("D", "04_evaluation/output/best_model.pkl")
    from .explain import run
    run()


def stage_07():
    gate("D", "04_evaluation/output/best_model.pkl")
    from .external_validate import run
    run()


def stage_report():
    """Assemble reports/results.md from gate-passed artefacts (raw tables only)."""
    gate("D", "04_evaluation/output/best_model.pkl")
    gate("F", "07_external_validation/output/external_metrics.json")
    from .report import run
    run()


STAGES = {
    "01": stage_01, "02": stage_02, "03": stage_03,
    "04": stage_04, "05": stage_05, "07": stage_07,
    "report": stage_report,
}
ORDER = ["01", "02", "03", "04", "05", "07", "report"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", help="01..07 or 'all'")
    args = ap.parse_args()
    targets = ORDER if args.stage == "all" else [args.stage.zfill(2)]
    for s in targets:
        if s not in STAGES:
            raise SystemExit(f"Unknown stage '{s}'. Valid: {ORDER} or 'all'.")
        print(f"\n=== STAGE {s} ===")
        STAGES[s]()
    print("\n[run] complete.")


if __name__ == "__main__":
    main()
