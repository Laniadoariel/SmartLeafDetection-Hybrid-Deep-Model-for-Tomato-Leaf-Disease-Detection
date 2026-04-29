"""Extract leaf bounding box annotations from red-marked images.

The annotated images have red bounding boxes drawn on them marking
individual leaves. This script:
  1. Detects red pixels (R>180, G<100, B<100) to find bounding box outlines
  2. Extracts bounding box coordinates via connected-component analysis
  3. Copies the ORIGINAL (clean) images into the YOLO dataset structure
  4. Saves YOLO-format annotation .txt files

Usage:
    python training/extract_annotations.py \
        --annotated 1.jpeg 2.jpeg 3.jpeg \
        --originals originals/1.jpeg originals/2.jpeg originals/3.jpeg \
        --output-dir training/dataset \
        --class-id 0

If --originals is not provided, falls back to inpainting the red marks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def detect_red_mask(image: np.ndarray) -> np.ndarray:
    """Create a binary mask of red-drawn pixels (BGR input)."""
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    mask = (r > 180) & (g < 100) & (b < 100)
    return mask.astype(np.uint8) * 255


def extract_bboxes_from_mask(
    mask: np.ndarray, min_area: int = 200,
) -> list[tuple[int, int, int, int]]:
    """Find bounding boxes of connected red regions.

    Returns list of (x1, y1, x2, y2) in pixel coordinates.
    Small regions below min_area are filtered as noise.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, x + w, y + h))
    return bboxes


def bbox_to_yolo(
    bbox: tuple[int, int, int, int], img_w: int, img_h: int, class_id: int,
) -> str:
    """Convert (x1, y1, x2, y2) pixel bbox to YOLO format.

    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1].
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def inpaint_red_regions(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fallback: remove red markings via inpainting when originals unavailable."""
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    return cv2.inpaint(image, dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)


def process_image(
    annotated_path: Path,
    original_path: Path | None,
    output_dir: Path,
    class_id: int,
    min_area: int = 200,
) -> int:
    """Process a single annotated image.

    Extracts bounding boxes from the red-marked annotated image,
    then copies the original clean image into the dataset.

    Returns the number of bounding boxes extracted.
    """
    annotated = cv2.imread(str(annotated_path))
    if annotated is None:
        print(f"  WARNING: Could not read {annotated_path}, skipping.")
        return 0

    h, w = annotated.shape[:2]
    stem = annotated_path.stem

    # Detect red markings and extract bounding boxes
    red_mask = detect_red_mask(annotated)
    bboxes = extract_bboxes_from_mask(red_mask, min_area=min_area)
    print(f"  {annotated_path.name}: found {len(bboxes)} leaf bounding boxes")

    # Save YOLO annotation file
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_path = labels_dir / f"{stem}.txt"
    with open(label_path, "w") as f:
        for bbox in bboxes:
            f.write(bbox_to_yolo(bbox, w, h, class_id) + "\n")

    # Save clean training image
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    clean_path = images_dir / f"{stem}.jpeg"

    if original_path and original_path.exists():
        # Use the original unmarked image
        original = cv2.imread(str(original_path))
        if original is not None:
            cv2.imwrite(str(clean_path), original)
            print(f"    Using original clean image: {original_path.name}")
        else:
            print(f"    WARNING: Could not read original {original_path}, falling back to inpainting")
            clean = inpaint_red_regions(annotated, red_mask)
            cv2.imwrite(str(clean_path), clean)
    else:
        # Fallback: inpaint the red markings
        print(f"    No original provided, inpainting red marks")
        clean = inpaint_red_regions(annotated, red_mask)
        cv2.imwrite(str(clean_path), clean)

    # Save visualization with detected bboxes drawn in green on the clean image
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    viz_img = cv2.imread(str(clean_path))
    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(viz_img, "leaf", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(viz_dir / f"{stem}_viz.jpeg"), viz_img)

    return len(bboxes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract leaf annotations from red-marked images"
    )
    parser.add_argument(
        "--annotated", nargs="+", required=True,
        help="Annotated image files (with red bounding boxes drawn on them)",
    )
    parser.add_argument(
        "--originals", nargs="+", default=None,
        help="Original clean image files (same order as --annotated). "
             "If not provided, red marks will be inpainted.",
    )
    parser.add_argument("--output-dir", default="training/dataset", help="Output directory")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO class ID (0=leaf)")
    parser.add_argument("--min-area", type=int, default=1000, help="Min contour area to keep (filters noise)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Pair annotated with originals
    originals = args.originals or [None] * len(args.annotated)
    if len(originals) != len(args.annotated):
        print("ERROR: --originals must have the same number of files as --annotated")
        return

    total = 0
    print("Extracting leaf annotations from red-marked images...\n")
    for ann_file, orig_file in zip(args.annotated, originals):
        orig_path = Path(orig_file) if orig_file else None
        count = process_image(Path(ann_file), orig_path, output_dir, args.class_id, args.min_area)
        total += count

    # Create data.yaml for YOLO training
    data_yaml = output_dir / "data.yaml"
    abs_images = str((output_dir / "images").resolve())
    data_yaml.write_text(
        f"train: {abs_images}\n"
        f"val: {abs_images}\n"
        f"nc: 1\n"
        f"names: ['leaf']\n"
    )

    print(f"\nDone! Extracted {total} leaf annotations total.")
    print(f"  Clean images:    {output_dir / 'images'}/")
    print(f"  YOLO labels:     {output_dir / 'labels'}/")
    print(f"  Visualizations:  {output_dir / 'visualizations'}/")
    print(f"  data.yaml:       {data_yaml}")
    print(f"\nNext step: train YOLO with:")
    print(f"  python training/train_yolo_leaves.py --data {data_yaml}")


if __name__ == "__main__":
    main()
