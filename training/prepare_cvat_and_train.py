#!/usr/bin/env python3
"""Convert CVAT polyline annotations to YOLO format and train YOLO11.

This script:
1. Parses the CVAT XML annotations (polyline → bounding box)
2. Copies images + labels into YOLO directory structure
3. Merges with the existing Roboflow tomato diseases dataset
4. Adds PlantDoc tomato images (multi-leaf field scenes)
5. Trains YOLO11 on the combined data

Usage:
    python training/prepare_cvat_and_train.py

The CVAT dataset contains "Tomato leaf late blight" images with polyline
annotations around individual leaves.  These are converted to axis-aligned
bounding boxes and assigned class index 3 (Late_blight) to match the
Roboflow dataset class mapping.
"""

from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CVAT_DIR = Path("task_2114473_annotations_2026_03_25_20_00_47_cvat for images 1.1")
CVAT_XML = CVAT_DIR / "annotations.xml"
CVAT_IMAGES = CVAT_DIR / "Tomato leaf late blight"

ROBOFLOW_DIR = Path("datasets/tomato_diseases")
PLANTDOC_DIR = Path("PlantDoc.v1-resize-416x416.yolov11")
TOMATO_6K_DIR = Path("datasets/tomato_6k/sample_16_12_2023")

# Output merged dataset
MERGED_DIR = Path("datasets/merged_diseases")

# Class mapping (same as Roboflow data.yaml)
# 0: Bacterial Spot, 1: Early_Blight, 2: Healthy, 3: Late_blight,
# 4: Leaf Mold, 5: Target_Spot, 6: black spot
LATE_BLIGHT_CLASS_ID = 3

TRAIN_SPLIT = 0.85  # 85% train, 15% val for CVAT data


def parse_cvat_xml(xml_path: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    """Parse CVAT XML and return {filename: [(class_id, x_center, y_center, w, h), ...]}
    
    Coordinates are normalized to [0, 1] (YOLO format).
    Polylines are converted to axis-aligned bounding boxes.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations: dict[str, list[tuple[int, float, float, float, float]]] = {}

    for image_elem in root.findall("image"):
        filename = image_elem.get("name", "")
        img_w = int(image_elem.get("width", "0"))
        img_h = int(image_elem.get("height", "0"))
        if img_w == 0 or img_h == 0:
            continue

        boxes = []
        for poly in image_elem.findall("polyline"):
            points_str = poly.get("points", "")
            if not points_str:
                continue

            # Parse "x1,y1;x2,y2;..." into list of (x, y)
            coords = []
            for pt in points_str.split(";"):
                parts = pt.strip().split(",")
                if len(parts) == 2:
                    coords.append((float(parts[0]), float(parts[1])))

            if len(coords) < 3:
                continue

            # Compute axis-aligned bounding box
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Clamp to image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_w, x_max)
            y_max = min(img_h, y_max)

            # Convert to YOLO format (normalized center + size)
            bw = x_max - x_min
            bh = y_max - y_min
            if bw < 5 or bh < 5:
                continue

            x_center = (x_min + bw / 2) / img_w
            y_center = (y_min + bh / 2) / img_h
            norm_w = bw / img_w
            norm_h = bh / img_h

            boxes.append((LATE_BLIGHT_CLASS_ID, x_center, y_center, norm_w, norm_h))

        if boxes:
            annotations[filename] = boxes

    return annotations


def setup_merged_dataset(annotations: dict[str, list[tuple[int, float, float, float, float]]]) -> Path:
    """Create merged dataset directory with CVAT + Roboflow + PlantDoc data."""

    # Clean old merged data and create output dirs
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    for split in ("train", "valid"):
        (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # --- 1. Copy Roboflow data as-is ---
    roboflow_copied = 0
    for split in ("train", "valid", "test"):
        src_img_dir = ROBOFLOW_DIR / split / "images"
        src_lbl_dir = ROBOFLOW_DIR / split / "labels"
        dst_split = "valid" if split == "test" else split

        if not src_img_dir.exists():
            continue

        for img_path in src_img_dir.glob("*.*"):
            dst_img = MERGED_DIR / dst_split / "images" / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                dst_lbl = MERGED_DIR / dst_split / "labels" / lbl_path.name
                if not dst_lbl.exists():
                    shutil.copy2(lbl_path, dst_lbl)
                roboflow_copied += 1

    print(f"Copied {roboflow_copied} Roboflow annotations")

    # --- 2. Convert and copy CVAT data ---
    filenames = list(annotations.keys())
    random.seed(42)
    random.shuffle(filenames)
    split_idx = int(len(filenames) * TRAIN_SPLIT)
    train_files = set(filenames[:split_idx])

    cvat_copied = 0
    cvat_boxes = 0
    for filename, boxes in annotations.items():
        split = "train" if filename in train_files else "valid"

        src_img = CVAT_IMAGES / filename
        if not src_img.exists():
            print(f"  WARNING: Image not found: {src_img}")
            continue

        safe_name = "cvat_" + filename.replace(" ", "_").replace("(", "").replace(")", "")
        stem = Path(safe_name).stem

        dst_img = MERGED_DIR / split / "images" / safe_name
        shutil.copy2(src_img, dst_img)

        dst_lbl = MERGED_DIR / split / "labels" / (stem + ".txt")
        with open(dst_lbl, "w") as f:
            for cls_id, xc, yc, w, h in boxes:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        cvat_copied += 1
        cvat_boxes += len(boxes)

    print(f"Converted {cvat_copied} CVAT images ({cvat_boxes} leaf bboxes)")

    # --- 3. Copy PlantDoc tomato data with class remapping ---
    # PlantDoc has 30 classes; we only take tomato-related ones and remap
    # to our 7-class system:
    #   PlantDoc 19 (Tomato Early blight leaf)              → our 1 (Early_Blight)
    #   PlantDoc 20 (Tomato Septoria leaf spot)              → our 5 (Target_Spot)
    #   PlantDoc 21 (Tomato leaf / healthy)                  → our 2 (Healthy)
    #   PlantDoc 22 (Tomato leaf bacterial spot)             → our 0 (Bacterial Spot)
    #   PlantDoc 23 (Tomato leaf late blight)                → our 3 (Late_blight)
    #   PlantDoc 26 (Tomato mold leaf)                       → our 4 (Leaf Mold)
    # Others (24=mosaic virus, 25=yellow virus, 27=spider mites) → skip
    PLANTDOC_REMAP: dict[int, int] = {
        19: 1,  # Early blight
        20: 5,  # Septoria → Target_Spot
        21: 2,  # Healthy
        22: 0,  # Bacterial spot
        23: 3,  # Late blight
        26: 4,  # Leaf Mold
    }

    plantdoc_copied = 0
    plantdoc_boxes = 0
    plantdoc_skipped = 0

    for split_name in ("train", "test"):
        src_img_dir = PLANTDOC_DIR / split_name / "images"
        src_lbl_dir = PLANTDOC_DIR / split_name / "labels"
        dst_split = "valid" if split_name == "test" else "train"

        if not src_img_dir.exists():
            continue

        for img_path in src_img_dir.glob("*.*"):
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            # Read and remap labels — keep only tomato classes
            remapped_lines = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    old_cls = int(parts[0])
                    if old_cls in PLANTDOC_REMAP:
                        new_cls = PLANTDOC_REMAP[old_cls]
                        remapped_lines.append(f"{new_cls} {' '.join(parts[1:])}\n")

            if not remapped_lines:
                plantdoc_skipped += 1
                continue

            # Use prefix to avoid name collisions
            safe_name = "pd_" + img_path.name
            stem = Path(safe_name).stem

            dst_img = MERGED_DIR / dst_split / "images" / safe_name
            shutil.copy2(img_path, dst_img)

            dst_lbl = MERGED_DIR / dst_split / "labels" / (stem + ".txt")
            with open(dst_lbl, "w") as f:
                f.writelines(remapped_lines)

            plantdoc_copied += 1
            plantdoc_boxes += len(remapped_lines)

    print(f"Copied {plantdoc_copied} PlantDoc tomato images "
          f"({plantdoc_boxes} leaf bboxes, skipped {plantdoc_skipped} non-tomato)")

    # --- 4. Copy Tomato 6K dataset with class remapping ---
    # 6K classes: 0=healthy, 1=bacterial_spot, 2=early_blight, 3=late_blight, 4=powdery_mildew
    # Our classes: 0=Bacterial Spot, 1=Early_Blight, 2=Healthy, 3=Late_blight, 4=Leaf Mold, 5=Target_Spot, 6=black spot
    TOMATO_6K_REMAP: dict[int, int] = {
        0: 2,  # healthy → Healthy
        1: 0,  # bacterial_spot → Bacterial Spot
        2: 1,  # early_blight → Early_Blight
        3: 3,  # late_blight → Late_blight
        4: 5,  # powdery_mildew → Target_Spot (closest match)
    }

    t6k_copied = 0
    t6k_boxes = 0

    if TOMATO_6K_DIR.exists():
        for split_name in ("train", "valid", "test"):
            src_img_dir = TOMATO_6K_DIR / "images" / split_name
            src_lbl_dir = TOMATO_6K_DIR / "labels" / split_name
            dst_split = "valid" if split_name in ("valid", "test") else "train"

            if not src_img_dir.exists():
                continue

            for img_path in src_img_dir.glob("*.*"):
                lbl_path = src_lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    continue

                # Remap class IDs
                remapped_lines = []
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        old_cls = int(parts[0])
                        if old_cls in TOMATO_6K_REMAP:
                            new_cls = TOMATO_6K_REMAP[old_cls]
                            remapped_lines.append(f"{new_cls} {' '.join(parts[1:])}\n")

                if not remapped_lines:
                    continue

                safe_name = "t6k_" + img_path.name
                stem = Path(safe_name).stem

                dst_img = MERGED_DIR / dst_split / "images" / safe_name
                shutil.copy2(img_path, dst_img)

                dst_lbl = MERGED_DIR / dst_split / "labels" / (stem + ".txt")
                with open(dst_lbl, "w") as f:
                    f.writelines(remapped_lines)

                t6k_copied += 1
                t6k_boxes += len(remapped_lines)

        print(f"Copied {t6k_copied} Tomato-6K images ({t6k_boxes} leaf bboxes)")
    else:
        print("Tomato-6K dataset not found, skipping")

    # --- Write data.yaml ---
    yaml_path = MERGED_DIR / "data.yaml"
    yaml_path.write_text(
        f"train: train/images\n"
        f"val: valid/images\n"
        f"\n"
        f"nc: 7\n"
        f"names: ['Bacterial Spot', 'Early_Blight', 'Healthy', 'Late_blight', "
        f"'Leaf Mold', 'Target_Spot', 'black spot']\n"
    )
    print(f"Dataset YAML: {yaml_path}")

    # Count totals
    train_imgs = len(list((MERGED_DIR / "train" / "images").glob("*.*")))
    val_imgs = len(list((MERGED_DIR / "valid" / "images").glob("*.*")))
    print(f"\nTotal: {train_imgs} train images, {val_imgs} val images")

    return yaml_path


def train(yaml_path: Path) -> None:
    """Train YOLO11 on the merged dataset."""
    from ultralytics import YOLO

    # Resume from last checkpoint if available, otherwise start fresh
    last_pt = Path("runs/detect/training/runs/diseases/yolo11s_plantdoc/weights/last.pt")
    if last_pt.exists():
        print(f"Resuming from {last_pt}")
        model = YOLO(str(last_pt))
    else:
        model = YOLO("yolo11s.pt")

    print("\n" + "=" * 60)
    print("Starting YOLO11 training on merged dataset")
    print("=" * 60)

    results = model.train(
        data=str(yaml_path.resolve()),
        epochs=30,
        imgsz=640,
        batch=4,
        patience=10,
        device="mps",
        project="training/runs/diseases",
        name="yolo11s_plantdoc",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        augment=True,
        mosaic=1.0,
        close_mosaic=0,
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
    print(f"Results saved to: training/runs/diseases/yolo11s_plantdoc/")
    print(f"Best weights: training/runs/diseases/yolo11s_plantdoc/weights/best.pt")
    print("=" * 60)


def main() -> None:
    print("Parsing CVAT annotations...")
    annotations = parse_cvat_xml(CVAT_XML)
    print(f"Found {len(annotations)} annotated images, "
          f"{sum(len(v) for v in annotations.values())} total leaf boxes")

    print("\nSetting up merged dataset...")
    yaml_path = setup_merged_dataset(annotations)

    print("\nStarting training...")
    train(yaml_path)


if __name__ == "__main__":
    main()
