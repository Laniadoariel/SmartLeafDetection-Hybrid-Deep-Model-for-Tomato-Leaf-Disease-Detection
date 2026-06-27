"""Draw ground-truth leaf boxes from a YOLO label file onto an image.

Use this to visually verify that the annotations line up with the actual
leaves before/while training. Handles the CVAT quirk where some label lines
carry a 6th column (a track id) by keeping only the first 5 tokens.

The raw image and label files are treated as READ-ONLY; output is written to a
separate file so nothing original is overwritten.

Usage
-----
    python training/leaf_detection/draw_labels.py \
        --image cvat_frames_000000_000905/frame_000000.jpg

    # explicit label path + custom output
    python training/leaf_detection/draw_labels.py \
        --image cvat_frames_000000_000905/frame_000000.jpg \
        --labels leaf_labels/labels/train/frame_000000.txt \
        --out training/leaf_detection/reports/label_preview_frame_000000.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_DIR = PROJECT_ROOT / "leaf_labels" / "labels" / "train"
DEFAULT_OUT_DIR = PROJECT_ROOT / "training" / "leaf_detection" / "reports"


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Return [(cls, xc, yc, w, h), ...] in normalized YOLO coords.

    Keeps only the first 5 whitespace-separated tokens per line, so an extra
    CVAT track-id column is ignored. Blank/short lines are skipped.
    """
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = (float(t) for t in parts[1:5])
        boxes.append((cls, xc, yc, w, h))
    return boxes


def draw_boxes(image_path: Path, label_path: Path, out_path: Path) -> int:
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"ERROR: could not read image: {image_path}")
    h, w = img.shape[:2]

    boxes = parse_label_file(label_path)
    for i, (_cls, xc, yc, bw, bh) in enumerate(boxes, start=1):
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"leaf {i}", (x1, max(y1 - 5, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # Header banner with the leaf count.
    cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(img, f"GT leaves: {len(boxes)}  ({image_path.name})",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return len(boxes)


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw GT leaf boxes on an image.")
    ap.add_argument("--image", required=True, help="Path to the image.")
    ap.add_argument("--labels", default=None,
                    help="Path to the YOLO label file (default: derive from "
                         "image stem under leaf_labels/labels/train/).")
    ap.add_argument("--out", default=None,
                    help="Output image path (default: reports/label_preview_<stem>.jpg).")
    args = ap.parse_args()

    image_path = _resolve(args.image)
    if args.labels:
        label_path = _resolve(args.labels)
    else:
        label_path = DEFAULT_LABELS_DIR / f"{image_path.stem}.txt"
    if args.out:
        out_path = _resolve(args.out)
    else:
        out_path = DEFAULT_OUT_DIR / f"label_preview_{image_path.stem}.jpg"

    n = draw_boxes(image_path, label_path, out_path)
    print(f"image : {image_path}")
    print(f"labels: {label_path} ({'found' if label_path.exists() else 'MISSING'})")
    print(f"drew  : {n} leaf boxes")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
