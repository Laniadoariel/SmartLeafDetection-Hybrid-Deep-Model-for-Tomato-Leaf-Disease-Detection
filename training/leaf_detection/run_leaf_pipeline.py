#!/usr/bin/env python3
"""Orchestrator for the leaf-detection improvement workflow.

Coordinates the individual stages (each a standalone, testable module) and
aggregates their results:

    prepare  -> train(baseline) -> train(improved) -> evaluate(both) -> summary

Every stage runs in-process by importing the stage's ``main`` with argv, so a
single command reproduces the full before/after comparison. Stages can also be
skipped to resume after a crash (results already on disk are reused).

Usage:
    # Full run (heavy — trains two models):
    python training/leaf_detection/run_leaf_pipeline.py --device mps

    # Just (re)build the dataset:
    python training/leaf_detection/run_leaf_pipeline.py --only prepare

    # Skip baseline (e.g. it is already trained) and only do improved+eval:
    python training/leaf_detection/run_leaf_pipeline.py \
        --skip baseline --device mps

    # Quick smoke test with tiny epoch counts:
    python training/leaf_detection/run_leaf_pipeline.py \
        --baseline-epochs 3 --improved-epochs 3 --device mps
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE / "reports"

STAGES = ["prepare", "baseline", "improved", "evaluate", "summary"]


def _run_module(script: str, argv: list[str]) -> None:
    """Run a sibling stage script as ``__main__`` with the given argv."""
    full = HERE / script
    print(f"\n>>> {script} {' '.join(argv)}\n", flush=True)
    old_argv = sys.argv
    sys.argv = [str(full), *argv]
    try:
        runpy.run_path(str(full), run_name="__main__")
    finally:
        sys.argv = old_argv


def _checkpoint(name: str, payload: dict) -> None:
    """Incrementally persist orchestrator progress so a crash loses nothing."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cp = REPORTS_DIR / "pipeline_progress.json"
    data = {}
    if cp.exists():
        try:
            data = json.loads(cp.read_text())
        except Exception:
            data = {}
    data[name] = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    cp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[checkpoint] {name} -> {cp}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=None, help="'mps', 'cpu', '0', ...")
    p.add_argument("--data", default="datasets/leaves_yolo/data.yaml")
    p.add_argument("--only", choices=STAGES, default=None,
                   help="Run only this stage")
    p.add_argument("--skip", nargs="*", default=[], choices=STAGES,
                   help="Stages to skip")
    p.add_argument("--baseline-epochs", type=int, default=None)
    p.add_argument("--improved-epochs", type=int, default=None)
    p.add_argument("--imgsz-eval", type=int, default=960)
    args = p.parse_args()

    todo = [args.only] if args.only else [s for s in STAGES if s not in args.skip]
    print(f"Stages to run: {todo}")
    dev = ["--device", args.device] if args.device else []

    # --- prepare ----------------------------------------------------------
    if "prepare" in todo:
        _run_module("prepare_leaf_dataset.py", [])
        _checkpoint("prepare", {"data": args.data})

    # --- baseline training ------------------------------------------------
    if "baseline" in todo:
        extra = ["--epochs", str(args.baseline_epochs)] if args.baseline_epochs else []
        _run_module("train_leaf_detector.py",
                    ["--preset", "baseline", "--data", args.data,
                     "--name", "leaf_baseline", *dev, *extra])
        _checkpoint("baseline", {"weights": "weights/leaf_baseline.pt"})

    # --- improved training ------------------------------------------------
    if "improved" in todo:
        extra = ["--epochs", str(args.improved_epochs)] if args.improved_epochs else []
        _run_module("train_leaf_detector.py",
                    ["--preset", "improved", "--data", args.data,
                     "--name", "leaf_improved", *dev, *extra])
        _checkpoint("improved", {"weights": "weights/leaf_improved.pt"})

    # --- evaluate both ----------------------------------------------------
    if "evaluate" in todo:
        # Evaluate each model at the input size it was trained on (fairest
        # "best vs best"): baseline=640, improved=960.
        for tag, wname, eval_imgsz, sweep in (
            ("baseline", "weights/leaf_baseline.pt", 640, []),
            ("improved", "weights/leaf_improved.pt", 960, ["--sweep-conf"]),
        ):
            if (PROJECT_ROOT / wname).exists():
                _run_module("evaluate_leaf_detector.py",
                            ["--weights", wname, "--tag", tag,
                             "--data", args.data, "--split", "test",
                             "--imgsz", str(eval_imgsz), *dev, *sweep])
            else:
                print(f"(skip eval '{tag}': {wname} not found)")
        _checkpoint("evaluate", {"csv": "training/leaf_detection/reports/metrics_comparison.csv"})

    # --- summary ----------------------------------------------------------
    if "summary" in todo:
        _build_summary()
        _checkpoint("summary", {"summary": "training/leaf_detection/reports/comparison_summary.json"})

    print("\nOrchestration complete.")


def _build_summary() -> None:
    """Combine baseline/improved metrics into one before/after summary."""
    base = REPORTS_DIR / "metrics_baseline.json"
    imp = REPORTS_DIR / "metrics_improved.json"
    if not (base.exists() and imp.exists()):
        print("(summary skipped: need both metrics_baseline.json and metrics_improved.json)")
        return

    b = json.loads(base.read_text())["metrics"]
    i = json.loads(imp.read_text())["metrics"]
    keys = ["precision", "recall", "f1", "mAP50", "mAP50_95"]
    summary = {
        "baseline": b,
        "improved": i,
        "delta": {k: round(i[k] - b[k], 5) for k in keys},
        "relative_pct": {
            k: (round(100 * (i[k] - b[k]) / b[k], 2) if b[k] else None) for k in keys
        },
    }
    out = REPORTS_DIR / "comparison_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print("BEFORE / AFTER  (held-out TEST split)")
    print("=" * 64)
    print(f"{'metric':<12}{'baseline':>12}{'improved':>12}{'delta':>12}")
    for k in keys:
        print(f"{k:<12}{b[k]:>12.4f}{i[k]:>12.4f}{summary['delta'][k]:>+12.4f}")
    print("=" * 64)
    print(f"Summary written to {out}")


if __name__ == "__main__":
    main()
