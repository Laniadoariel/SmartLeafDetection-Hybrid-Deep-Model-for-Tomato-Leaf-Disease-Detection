#!/usr/bin/env python3
"""Evaluate a trained leaf detector on the held-out TEST split.

Computes the standard object-detection metrics (precision, recall, mAP@0.5 and
mAP@0.5:0.95) on the ``test`` split defined in the dataset ``data.yaml`` and:

* writes a per-model metrics JSON,
* appends a row to a shared comparison CSV so baseline vs improved sit
  side-by-side,
* optionally sweeps the confidence threshold to recommend the operating point
  with the best F1 (useful for tuning ``leaf_confidence_threshold``).

Usage:
    python training/leaf_detection/evaluate_leaf_detector.py \
        --weights weights/leaf_baseline.pt --tag baseline

    python training/leaf_detection/evaluate_leaf_detector.py \
        --weights weights/leaf_improved.pt --tag improved --sweep-conf
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "training" / "leaf_detection" / "reports"


def _extract_metrics(res) -> dict[str, float]:
    """Pull the headline detection metrics out of an Ultralytics results obj."""
    box = res.box
    p = float(box.mp)            # mean precision across classes
    r = float(box.mr)            # mean recall
    map50 = float(box.map50)
    map5095 = float(box.map)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {
        "precision": round(p, 5),
        "recall": round(r, 5),
        "f1": round(f1, 5),
        "mAP50": round(map50, 5),
        "mAP50_95": round(map5095, 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to model .pt")
    parser.add_argument("--data", default="datasets/leaves_yolo/data.yaml")
    parser.add_argument("--tag", required=True, help="Label for this run, e.g. baseline/improved")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Conf threshold for mAP computation (low = standard mAP)")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sweep-conf", action="store_true",
                        help="Also sweep confidence to find best-F1 operating point")
    args = parser.parse_args()

    weights = (PROJECT_ROOT / args.weights).resolve() if not Path(args.weights).is_absolute() else Path(args.weights)
    data_path = (PROJECT_ROOT / args.data).resolve()
    if not weights.exists():
        raise SystemExit(f"ERROR: weights not found: {weights}")
    if not data_path.exists():
        raise SystemExit(f"ERROR: data.yaml not found: {data_path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    # Resolve device so an explicit but unavailable backend degrades to CPU.
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT))
    from smart_leaf_detection.device_utils import resolve_ultralytics_device
    resolved_device = resolve_ultralytics_device(args.device)

    print(f"Evaluating '{args.tag}' on split='{args.split}' using {weights.name}")
    model = YOLO(str(weights))
    res = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=resolved_device,
        plots=True,
        verbose=False,
    )
    metrics = _extract_metrics(res)
    print("  " + "  ".join(f"{k}={v}" for k, v in metrics.items()))

    record = {
        "tag": args.tag,
        "evaluated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "weights": str(weights),
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "iou": args.iou,
        "conf_for_map": args.conf,
        "metrics": metrics,
    }

    # --- Optional confidence sweep ---------------------------------------
    if args.sweep_conf:
        sweep = []
        best = None
        for c in [round(0.05 * i, 2) for i in range(1, 19)]:  # 0.05 .. 0.90
            r = model.val(
                data=str(data_path), split=args.split, imgsz=args.imgsz,
                conf=c, iou=args.iou,
                device=resolved_device,
                plots=False, verbose=False,
            )
            m = _extract_metrics(r)
            sweep.append({"conf": c, **m})
            if best is None or m["f1"] > best["f1"]:
                best = {"conf": c, **m}
        record["conf_sweep"] = sweep
        record["recommended_conf_by_f1"] = best
        print(f"  Recommended conf (best F1): {best['conf']} -> F1={best['f1']}")

    # --- Persist per-model JSON ------------------------------------------
    out_json = REPORTS_DIR / f"metrics_{args.tag}.json"
    out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"  Wrote {out_json}")

    # --- Append to shared comparison CSV ---------------------------------
    csv_path = REPORTS_DIR / "metrics_comparison.csv"
    header = ["tag", "split", "weights", "precision", "recall", "f1",
              "mAP50", "mAP50_95", "evaluated_at"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([
            args.tag, args.split, weights.name,
            metrics["precision"], metrics["recall"], metrics["f1"],
            metrics["mAP50"], metrics["mAP50_95"], record["evaluated_at"],
        ])
    print(f"  Appended row to {csv_path}")


if __name__ == "__main__":
    main()
