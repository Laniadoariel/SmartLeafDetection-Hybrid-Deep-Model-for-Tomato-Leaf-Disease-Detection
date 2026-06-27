#!/usr/bin/env python3
"""Train a YOLOv11 leaf-detection model on the cleaned leaf dataset.

Two presets are provided so a fair before/after comparison can be made on the
SAME held-out test split:

* ``baseline``  — reproduces the project's original leaf-training recipe
                  (YOLOv11-nano, 640 px, the default aggressive augmentation
                  from ``training/train_yolo_leaves.py``). This is the "before".

* ``improved``  — the tuned recipe (YOLOv11-small, larger input size for small
                  / overlapping leaves, cosine LR, AdamW, mosaic with
                  ``close_mosaic`` near the end, lighter mixup, copy-paste for
                  dense scenes, longer schedule with early stopping). This is
                  the "after".

The script copies the best checkpoint to ``weights/<run-name>.pt`` and, for the
improved preset, also to ``weights/leaf_best.pt`` (the artifact the app uses).

Usage:
    python training/leaf_detection/train_leaf_detector.py --preset baseline
    python training/leaf_detection/train_leaf_detector.py --preset improved

    # override anything:
    python training/leaf_detection/train_leaf_detector.py \
        --preset improved --epochs 150 --imgsz 1024 --batch 8 --device mps
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Preset hyper-parameters. Only keys understood by Ultralytics' ``train`` are
# forwarded; ``model`` is popped out before the call.
PRESETS: dict[str, dict] = {
    "baseline": {
        "model": "yolo11n.pt",
        "imgsz": 640,
        "epochs": 80,
        "batch": 8,
        "optimizer": "auto",
        "cos_lr": False,
        "patience": 20,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 15.0, "translate": 0.1, "scale": 0.5,
        "fliplr": 0.5, "flipud": 0.2,
        "mosaic": 1.0, "mixup": 0.1, "close_mosaic": 0,
    },
    "improved": {
        "model": "yolo11s.pt",
        "imgsz": 960,          # leaves are small & dense in drone frames
        "epochs": 150,
        "batch": 8,
        "optimizer": "AdamW",
        "lr0": 0.002, "lrf": 0.01, "cos_lr": True,
        "warmup_epochs": 3.0,
        "weight_decay": 0.0005,
        "patience": 40,
        # Augmentation tuned for many small, overlapping objects.
        # NOTE on the annotation policy: only CLEAR, camera-facing leaves are
        # labelled; blurry / edge-on leaves are intentionally left unlabelled
        # (background). So we deliberately AVOID blur/erasing augmentation that
        # would teach the model to fire on unclear leaves. HSV/geom/flip keep
        # leaves sharp and only change colour/pose, which is consistent with the
        # "salient, front-facing leaf" target.
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 10.0, "translate": 0.1, "scale": 0.5, "shear": 2.0,
        "perspective": 0.0, "fliplr": 0.5, "flipud": 0.5,
        "mosaic": 1.0, "close_mosaic": 15,   # disable mosaic for last 15 epochs
        "mixup": 0.05,
        "erasing": 0.0,        # no random erasing (would mimic occluded/unclear leaves)
        "box": 7.5, "cls": 0.5, "dfl": 1.5,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="improved")
    parser.add_argument("--data", default="datasets/leaves_yolo/data.yaml")
    parser.add_argument("--name", default=None, help="Run name (default: leaf_<preset>)")
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "leaves"),
                        help="Output dir (absolute, to avoid Ultralytics runs_dir nesting)")
    parser.add_argument("--device", default=None, help="'mps', 'cpu', '0', ... (auto if unset)")
    # Optional overrides (None => use preset value)
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--cache", default=None,
                        help="Cache images to speed up epochs and reduce disk "
                             "I/O throttling: 'ram', 'disk', or omit for none.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted run from its last.pt "
                             "(continues to the original epoch target).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_path = (PROJECT_ROOT / args.data).resolve()
    if not data_path.exists():
        raise SystemExit(
            f"ERROR: data.yaml not found at {data_path}\n"
            "Run prepare_leaf_dataset.py first."
        )

    cfg = dict(PRESETS[args.preset])
    model_name = args.model or cfg.pop("model")
    cfg.pop("model", None)

    # Apply CLI overrides.
    for key in ("epochs", "imgsz", "batch"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    if args.cache:
        cfg["cache"] = args.cache

    run_name = args.name or f"leaf_{args.preset}"

    from ultralytics import YOLO  # imported lazily so --help works without it

    # Resolve device so an explicit but unavailable backend (e.g. --device cuda
    # on a Mac) degrades to CPU instead of crashing. None => Ultralytics auto.
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT))
    from smart_leaf_detection.device_utils import resolve_ultralytics_device
    resolved_device = resolve_ultralytics_device(args.device)

    # --- Resume an interrupted run from its last.pt -------------------------
    if args.resume:
        resume_run = Path(args.project) / run_name
        if not resume_run.is_absolute():
            resume_run = PROJECT_ROOT / resume_run
        last_ckpt = resume_run / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise SystemExit(
                f"ERROR: --resume requested but no checkpoint at {last_ckpt}\n"
                "Start a fresh run (without --resume) first."
            )
        print("=" * 60)
        print(f"RESUMING leaf detector from {last_ckpt}")
        print("=" * 60)
        model = YOLO(str(last_ckpt))
        results = model.train(resume=True)
    else:
        print("=" * 60)
        print(f"Training leaf detector — preset='{args.preset}'")
        print(f"  base model : {model_name}")
        print(f"  data       : {data_path}")
        print(f"  run        : {args.project}/{run_name}")
        print(f"  device     : {resolved_device or 'auto'}")
        print(f"  key cfg    : imgsz={cfg.get('imgsz')} epochs={cfg.get('epochs')} "
              f"batch={cfg.get('batch')} optimizer={cfg.get('optimizer')} "
              f"cache={cfg.get('cache', 'none')}")
        print("=" * 60)

        model = YOLO(model_name)
        results = model.train(
            data=str(data_path),
            project=args.project,
            name=run_name,
            exist_ok=True,
            seed=args.seed,
            device=resolved_device,
            plots=True,
            val=True,
            **cfg,
        )

    # Locate best.pt and copy it to weights/. Use the trainer's authoritative
    # save_dir (Ultralytics may relocate the run via its own runs_dir setting,
    # so we must not assume project/name is the final path).
    run_dir = None
    trainer = getattr(model, "trainer", None)
    if trainer is not None and getattr(trainer, "save_dir", None):
        run_dir = Path(trainer.save_dir)
    if run_dir is None and getattr(results, "save_dir", None):
        run_dir = Path(results.save_dir)
    if run_dir is None:
        run_dir = Path(args.project) / run_name
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir

    best = run_dir / "weights" / "best.pt"
    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(exist_ok=True)
    print(f"\nResolved run dir: {run_dir}")

    if best.exists():
        named = weights_dir / f"{run_name}.pt"
        shutil.copy2(best, named)
        print(f"\nBest weights copied to: {named}")
        if args.preset == "improved":
            deployed = weights_dir / "leaf_best.pt"
            shutil.copy2(best, deployed)
            print(f"Deployed leaf model to: {deployed}")
    else:
        print(f"\nWARNING: best.pt not found at {best}")

    # Persist the resolved config next to the run for traceability.
    try:
        summary = {
            "preset": args.preset,
            "model": model_name,
            "data": str(data_path),
            "run_dir": str(run_dir),
            "config": cfg,
            "best_weights": str(best) if best.exists() else None,
        }
        (run_dir / "leaf_train_config.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # non-fatal
        print(f"(could not write config summary: {exc})")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
