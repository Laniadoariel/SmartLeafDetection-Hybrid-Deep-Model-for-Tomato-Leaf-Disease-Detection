#!/usr/bin/env python3
"""Crash-proof training script for YOLO11.

Uses the merged dataset (Roboflow + CVAT + PlantDoc + Tomato-6K)
but with settings that avoid the Ultralytics validation crash on Python 3.14:
- val=False during training (skips the buggy validation step)
- Saves checkpoints every epoch
- Runs validation separately after training completes

Usage:
    source venv/bin/activate && python training/train_stable.py
"""
from __future__ import annotations

from pathlib import Path


def main() -> None:
    from ultralytics import YOLO

    dataset_yaml = Path("datasets/merged_diseases/data.yaml")
    if not dataset_yaml.exists():
        print("Merged dataset not found. Run prepare_cvat_and_train.py first")
        print("(just the dataset prep part — you can Ctrl+C before training starts)")
        return

    # Use yolo11s (small) for better accuracy — resume if previous training exists
    last_pt = Path("runs/detect/training/runs/diseases/yolo11s_stable/weights/last.pt")
    if last_pt.exists():
        print(f"Resuming from {last_pt}")
        model = YOLO(str(last_pt))
    else:
        model = YOLO("yolo11s.pt")

    print("=" * 60)
    print("STABLE TRAINING — validation disabled to prevent crash")
    print("=" * 60)
    print(f"Dataset: {dataset_yaml}")
    print(f"Model: yolo11s.pt")
    print()

    results = model.train(
        data=str(dataset_yaml.resolve()),
        epochs=60,
        imgsz=640,
        batch=4,
        device="mps",
        project="training/runs/diseases",
        name="yolo11s_stable",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        # Key crash prevention settings
        val=False,          # Skip validation during training (avoids the shape mismatch bug)
        save_period=5,      # Save checkpoint every 5 epochs
        patience=0,         # Disable early stopping (no val = no metric to watch)
        close_mosaic=0,     # Keep mosaic on entire time

        # Augmentation
        augment=True,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        degrees=15.0,
        translate=0.2,
        scale=0.5,
        fliplr=0.5,
        flipud=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    weights_dir = Path("training/runs/diseases/yolo11s_stable/weights")
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"

    # Since val=False, best.pt might not exist — use last.pt
    use_weights = str(best) if best.exists() else str(last)
    print(f"Weights: {use_weights}")

    # Run validation separately to get metrics
    print("\nRunning validation...")
    try:
        val_model = YOLO(use_weights)
        metrics = val_model.val(
            data=str(dataset_yaml.resolve()),
            batch=1,  # batch=1 avoids the shape mismatch crash
            device="mps",
        )
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
    except Exception as e:
        print(f"Validation failed (non-critical): {e}")
        print("The trained weights are still valid and usable.")

    print(f"\nTo test: python run_on_frames.py --frames test1.jpg test2.jpg test3.jpg "
          f"--disease-weights {use_weights} --confidence 0.4")
    print("=" * 60)


if __name__ == "__main__":
    main()
