#!/usr/bin/env python3
"""Evaluate a trained leaf-disease classifier on the held-out TEST split and
(optionally) compare it against the existing YOLO disease detector on the same
crops.

Reports (per model): accuracy, macro precision/recall/F1, per-class
precision/recall/F1, confusion matrix, **average inference time per image**,
**parameter count**, and **model size**. All metrics are computed with NumPy
(no scikit-learn dependency) so the protocol is identical for every model.

Outputs (reports dir):
    evaluation_metrics[_<tag>].json
    confusion_matrix[_<tag>].png
    per_class_f1[_<tag>].png

Usage:
    python training/disease_classification/evaluate_classifier.py
    python training/disease_classification/evaluate_classifier.py \
        --weights weights/benchmark/resnet50.pt --tag resnet50 \
        --reports-dir training/disease_classification/reports/resnet50 \
        --compare-yolo weights/best.pt --device mps
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "disease_classification"))


def per_class_metrics(conf: np.ndarray, classes: list[str]) -> dict:
    """Precision/recall/F1 per class + macro and accuracy from a confusion matrix."""
    tp = np.diag(conf).astype(np.float64)
    pred_tot = conf.sum(0).astype(np.float64)
    true_tot = conf.sum(1).astype(np.float64)
    precision = np.divide(tp, pred_tot, out=np.zeros_like(tp), where=pred_tot > 0)
    recall = np.divide(tp, true_tot, out=np.zeros_like(tp), where=true_tot > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    per_class = {
        c: {"precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(true_tot[i])}
        for i, c in enumerate(classes)
    }
    accuracy = float(tp.sum() / conf.sum()) if conf.sum() else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(float(precision.mean()), 4),
        "macro_recall": round(float(recall.mean()), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "per_class": per_class,
    }


def failure_analysis(conf: np.ndarray, classes: list[str], top_k: int = 2) -> dict:
    """For each class, the classes it is most often confused with (off-diagonal)."""
    out = {}
    for i, c in enumerate(classes):
        support = int(conf[i].sum())
        row = conf[i].astype(np.float64).copy()
        row[i] = 0.0  # ignore correct predictions
        order = np.argsort(row)[::-1]
        confused = [
            {"class": classes[j], "count": int(row[j]),
             "rate": round(float(row[j] / max(support, 1)), 4)}
            for j in order if row[j] > 0
        ][:top_k]
        out[c] = {"support": support, "most_confused_with": confused}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets/leaf_clf")
    ap.add_argument("--weights", default="weights/leaf_classifier.pt")
    ap.add_argument("--split", default="test", choices=["test", "validation", "train"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default=None, help="Suffix for output filenames (e.g. the arch name)")
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--timing-samples", type=int, default=200,
                    help="Number of test crops to time for avg inference (after warmup)")
    ap.add_argument("--compare-yolo", default=None,
                    help="Path to YOLO disease weights for a side-by-side comparison")
    args = ap.parse_args()

    import cv2
    import torch
    from torchvision import datasets, transforms
    from smart_leaf_detection.device_utils import resolve_torch_device
    from smart_leaf_detection.leaf_disease_classifier import LeafDiseaseClassifier

    reports_dir = (Path(args.reports_dir).resolve() if args.reports_dir
                   else PROJECT_ROOT / "training" / "disease_classification" / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    split_dir = (PROJECT_ROOT / args.data / args.split).resolve()
    if not split_dir.is_dir():
        raise SystemExit(f"Split not found: {split_dir} (run build_classification_dataset.py)")

    weights_path = PROJECT_ROOT / args.weights
    clf = LeafDiseaseClassifier(str(weights_path), device=args.device or "auto")
    classes = clf.classes
    cidx = {c: i for i, c in enumerate(classes)}
    n = len(classes)

    # Model footprint
    num_params = int(sum(p.numel() for p in clf._model.parameters()))
    model_size_mb = round(os.path.getsize(weights_path) / (1024 * 1024), 2)

    ds = datasets.ImageFolder(str(split_dir), transform=transforms.Lambda(lambda im: im))
    folder_idx_to_name = {v: k for k, v in ds.class_to_idx.items()}

    conf = np.zeros((n, n), dtype=np.int64)
    infer_times: list[float] = []

    # YOLO comparison setup
    do_yolo = args.compare_yolo is not None
    if do_yolo:
        from ultralytics import YOLO
        from class_mapping import map_label
        ydev = resolve_torch_device(args.device) if args.device else None
        ymodel = YOLO(str(PROJECT_ROOT / args.compare_yolo))
        ynames = ymodel.names
        yolo_correct = yolo_total = 0
        yolo_conf = np.zeros((n, n), dtype=np.int64)
        yolo_no_pred = 0

    warmup = 5
    for k, (img_path, folder_label) in enumerate(ds.samples):
        true_name = folder_idx_to_name[folder_label]
        if true_name not in cidx:
            continue
        ti = cidx[true_name]
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        t0 = time.perf_counter()
        pred_name, _conf, _ = clf.classify(bgr)
        dt = time.perf_counter() - t0
        if k >= warmup and len(infer_times) < args.timing_samples:
            infer_times.append(dt)
        conf[ti, cidx[pred_name]] += 1

        if do_yolo:
            yolo_total += 1
            res = ymodel(bgr, conf=0.25, verbose=False, device=ydev)
            boxes = res[0].boxes
            mapped = None
            if boxes is not None and len(boxes) > 0:
                bi = int(boxes.conf.argmax())
                try:
                    canon = map_label(ynames[int(boxes.cls[bi])])
                except KeyError:
                    canon = None
                mapped = canon if canon in cidx else None
            if mapped is None:
                yolo_no_pred += 1
            else:
                yolo_conf[ti, cidx[mapped]] += 1
                if mapped == true_name:
                    yolo_correct += 1

    metrics = per_class_metrics(conf, classes)
    avg_ms = round(1000 * float(np.mean(infer_times)), 3) if infer_times else None
    median_ms = round(1000 * float(np.median(infer_times)), 3) if infer_times else None

    result = {
        "tag": args.tag, "arch": clf.arch, "weights": args.weights, "split": args.split,
        "device": str(clf.device),
        "num_params": num_params,
        "num_params_millions": round(num_params / 1e6, 2),
        "model_size_mb": model_size_mb,
        "avg_inference_ms": avg_ms,
        "median_inference_ms": median_ms,
        "timing_samples": len(infer_times),
        **metrics,
    }
    result["failure_analysis"] = failure_analysis(conf, classes)
    if do_yolo:
        result["yolo_comparison"] = {
            "yolo_weights": args.compare_yolo,
            "yolo_accuracy_on_crops": round(yolo_correct / max(yolo_total, 1), 4),
            "classifier_accuracy_on_crops": metrics["accuracy"],
            "yolo_no_prediction_rate": round(yolo_no_pred / max(yolo_total, 1), 4),
            "note": "YOLO labels mapped to canonical via class_mapping; no-detection/unmapped counted as wrong.",
        }

    out_json = reports_dir / f"evaluation_metrics{suffix}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_json}")

    # ---- Plots (best-effort) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = conf.astype(np.float64)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n * 0.9)))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(classes)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"Confusion matrix ({clf.arch}, {args.split})")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(reports_dir / f"confusion_matrix{suffix}.png", dpi=130); plt.close(fig)

        f1s = [metrics["per_class"][c]["f1"] for c in classes]
        fig, ax = plt.subplots(figsize=(max(7, n * 0.9), 4))
        ax.bar(range(n), f1s, color="#16a34a")
        ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_ylim(0, 1); ax.set_ylabel("F1"); ax.set_title(f"Per-class F1 ({clf.arch})")
        fig.tight_layout(); fig.savefig(reports_dir / f"per_class_f1{suffix}.png", dpi=130); plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"(plots skipped: {exc})")


if __name__ == "__main__":
    main()
