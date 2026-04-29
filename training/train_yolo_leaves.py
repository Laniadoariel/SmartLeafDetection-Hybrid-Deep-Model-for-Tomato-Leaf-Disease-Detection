"""Fine-tune YOLOv11 for leaf detection on drone imagery.

This script fine-tunes a pretrained YOLOv11 model on the extracted leaf
annotations. Since we only have ~3 images, this serves as a proof-of-concept.
For production use, you'd need 500+ annotated images.

Usage:
    # First extract annotations:
    python training/extract_annotations.py --images 1.jpeg 2.jpeg 3.jpeg

    # Then train:
    python training/train_yolo_leaves.py \
        --data training/dataset/data.yaml \
        --epochs 50 \
        --model yolo11n.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11 for leaf detection")
    parser.add_argument("--data", default="training/dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained YOLO model to fine-tune")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (small for few images)")
    parser.add_argument("--project", default="training/runs/leaves", help="Output directory")
    parser.add_argument("--name", default="yolo11_leaves", help="Experiment name")
    parser.add_argument("--device", default="", help="Device: '' for auto, 'cpu', '0', etc.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data.yaml not found at {data_path}")
        print("Run extract_annotations.py first to create the dataset.")
        return

    from ultralytics import YOLO

    print(f"Loading pretrained model: {args.model}")
    model = YOLO(args.model)

    print(f"Starting training for {args.epochs} epochs...")
    print(f"  Dataset: {args.data}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device if args.device else None,
        # Augmentation settings — aggressive for small datasets
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.2,
        mosaic=1.0,
        mixup=0.1,
        # Early stopping
        patience=20,
        # Save best model
        save=True,
    )

    # Copy best weights to project root for pipeline use
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    target = Path("yolo11_leaves.pt")
    if best_weights.exists():
        import shutil
        shutil.copy2(best_weights, target)
        print(f"\nBest weights copied to: {target}")
    else:
        print(f"\nWARNING: best.pt not found at {best_weights}")

    print("Training complete!")
    print(f"Results saved to: {args.project}/{args.name}/")


if __name__ == "__main__":
    main()
