#!/usr/bin/env python3
"""Prepare a clean YOLO leaf-detection dataset from the raw CVAT export.

This script is the first stage of the leaf-detection training pipeline. It:

1. Reads the RAW images from ``cvat_frames_000000_000905/`` and the RAW
   per-image YOLO labels from ``leaf_labels/labels/train/``.
2. Cleans every label file:
     - strips the trailing 6th column (a CVAT *track id*) so each box is the
       standard 5-column YOLO detection format ``class x_center y_center w h``;
     - validates the class index (only ``0`` == ``leaf`` is allowed);
     - clamps normalized coordinates into ``[0, 1]`` and drops degenerate
       (zero/negative width or height) boxes.
3. Pairs images with labels, reporting unlabeled images and orphan labels.
4. Splits the data into train/val/test using a **grouped temporal block
   split** (consecutive frames stay together) so that near-duplicate adjacent
   video frames do not leak across splits and inflate the metrics.
5. Writes a fresh YOLO dataset under ``datasets/leaves_yolo/`` (images +
   cleaned labels + ``data.yaml``) WITHOUT modifying any raw input file.
6. Saves a JSON + CSV report describing the dataset and every fix applied.

The raw inputs are treated as read-only. Re-running the script rebuilds the
output dataset from scratch (idempotent).

Usage:
    python training/leaf_detection/prepare_leaf_dataset.py \
        --images cvat_frames_000000_000905 \
        --labels leaf_labels/labels/train \
        --out datasets/leaves_yolo
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# Project root = three levels up from this file
# (.../SmartLeafDetection/training/leaf_detection/prepare_leaf_dataset.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
LEAF_CLASS_ID = 0
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class LabelStats:
    """Accumulated statistics describing the label-cleaning pass."""

    files_seen: int = 0
    files_with_boxes: int = 0
    empty_files: int = 0
    total_boxes_raw: int = 0
    total_boxes_kept: int = 0
    lines_with_trackid: int = 0          # 6-column lines fixed
    boxes_dropped_bad_class: int = 0
    boxes_dropped_degenerate: int = 0
    boxes_dropped_malformed: int = 0
    boxes_clamped: int = 0
    tiny_boxes: int = 0                  # area < 0.1% of image (for info only)
    boxes_per_image: list[int] = field(default_factory=list)


def _parse_clean_label_line(
    line: str,
) -> tuple[tuple[int, float, float, float, float] | None, dict[str, bool]]:
    """Parse one raw label line and return a cleaned 5-tuple plus fix flags.

    Returns ``(None, flags)`` when the line should be dropped. ``flags`` keys:
    ``trackid`` (extra columns stripped), ``bad_class``, ``degenerate``,
    ``malformed``, ``clamped``.
    """
    flags = {
        "trackid": False,
        "bad_class": False,
        "degenerate": False,
        "malformed": False,
        "clamped": False,
    }

    tokens = line.split()
    if len(tokens) < 5:
        # Blank or truncated line — not a usable box.
        if tokens:
            flags["malformed"] = True
        return None, flags

    if len(tokens) > 5:
        # Extra column(s) — CVAT appends a track id. Keep only the first five.
        flags["trackid"] = True

    try:
        cls = int(float(tokens[0]))
        xc, yc, w, h = (float(t) for t in tokens[1:5])
    except ValueError:
        flags["malformed"] = True
        return None, flags

    if cls != LEAF_CLASS_ID:
        flags["bad_class"] = True
        return None, flags

    # Clamp center/size into valid normalized range.
    def _clamp(v: float) -> tuple[float, bool]:
        cv = min(max(v, 0.0), 1.0)
        return cv, (cv != v)

    xc, c1 = _clamp(xc)
    yc, c2 = _clamp(yc)
    w, c3 = _clamp(w)
    h, c4 = _clamp(h)
    if c1 or c2 or c3 or c4:
        flags["clamped"] = True

    if w <= 0.0 or h <= 0.0:
        flags["degenerate"] = True
        return None, flags

    return (cls, xc, yc, w, h), flags


def clean_label_file(path: Path, stats: LabelStats) -> list[str]:
    """Read a raw label file and return cleaned 5-column YOLO lines."""
    stats.files_seen += 1
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"  WARNING: cannot read label '{path}': {exc}")
        return []

    cleaned: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        stats.total_boxes_raw += 1
        box, flags = _parse_clean_label_line(line)

        if flags["trackid"]:
            stats.lines_with_trackid += 1
        if flags["clamped"]:
            stats.boxes_clamped += 1

        if box is None:
            if flags["bad_class"]:
                stats.boxes_dropped_bad_class += 1
            elif flags["degenerate"]:
                stats.boxes_dropped_degenerate += 1
            elif flags["malformed"]:
                stats.boxes_dropped_malformed += 1
            continue

        cls, xc, yc, w, h = box
        if w * h < 0.001:  # < 0.1% of image area
            stats.tiny_boxes += 1
        cleaned.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    stats.total_boxes_kept += len(cleaned)
    stats.boxes_per_image.append(len(cleaned))
    if cleaned:
        stats.files_with_boxes += 1
    else:
        stats.empty_files += 1
    return cleaned


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    """Return the image file matching ``stem`` regardless of extension."""
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def grouped_block_split(
    stems: list[str],
    block_size: int,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[str]]:
    """Assign consecutive-frame blocks to train/val/test.

    Frames are sorted, grouped into contiguous blocks of ``block_size``
    frames, and whole blocks are round-robin assigned to splits according to
    the requested ratios. Keeping adjacent (near-identical) frames in the same
    split prevents train/test leakage that would otherwise inflate metrics.
    """
    stems = sorted(stems)
    blocks: list[list[str]] = [
        stems[i : i + block_size] for i in range(0, len(stems), block_size)
    ]
    n_blocks = len(blocks)
    n_test = max(1, round(n_blocks * test_ratio)) if test_ratio > 0 else 0
    n_val = max(1, round(n_blocks * val_ratio)) if val_ratio > 0 else 0

    # Spread val/test blocks evenly across the timeline instead of taking a
    # single contiguous tail, so each split covers the whole flight.
    assignment: dict[int, str] = {i: "train" for i in range(n_blocks)}
    if n_val + n_test > 0 and n_blocks > n_val + n_test:
        holdout_every = n_blocks / (n_val + n_test)
        picks = [int(round(k * holdout_every)) for k in range(n_val + n_test)]
        picks = sorted({min(p, n_blocks - 1) for p in picks})
        for idx, b in enumerate(picks):
            assignment[b] = "val" if idx % 2 == 0 and n_val > 0 else "test"

    out: dict[str, list[str]] = {s: [] for s in SPLIT_NAMES}
    for i, block in enumerate(blocks):
        out[assignment[i]].extend(block)
    return out


def _materialize(src: Path, dst: Path, copy: bool) -> None:
    """Place an image at ``dst`` either by copy or symlink."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="cvat_frames_000000_000905")
    parser.add_argument("--labels", default="leaf_labels/labels/train")
    parser.add_argument("--out", default="datasets/leaves_yolo")
    parser.add_argument("--block-size", type=int, default=20,
                        help="Consecutive frames per split block (anti-leakage)")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images instead of symlinking them")
    parser.add_argument("--include-unlabeled", action="store_true",
                        help="Include images without any label as background")
    args = parser.parse_args()

    images_dir = (PROJECT_ROOT / args.images).resolve()
    labels_dir = (PROJECT_ROOT / args.labels).resolve()
    out_dir = (PROJECT_ROOT / args.out).resolve()
    reports_dir = PROJECT_ROOT / "training" / "leaf_detection" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.is_dir():
        raise SystemExit(f"ERROR: images dir not found: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"ERROR: labels dir not found: {labels_dir}")

    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Output: {out_dir}")

    # --- Index raw inputs -------------------------------------------------
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    image_stems = {p.stem for p in image_paths}
    label_paths = sorted(labels_dir.glob("*.txt"))
    label_stems = {p.stem for p in label_paths}

    unlabeled = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)
    print(f"Found {len(image_paths)} images, {len(label_paths)} label files")
    print(f"  Unlabeled images: {len(unlabeled)} | Orphan labels: {len(orphan_labels)}")

    # --- Clean labels (incremental: write a manifest as we go) ------------
    stats = LabelStats()
    cleaned_by_stem: dict[str, list[str]] = {}
    manifest_path = reports_dir / "clean_manifest_partial.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["stem", "raw_label_exists", "boxes_kept"])
        for stem in sorted(image_stems):
            lbl = labels_dir / f"{stem}.txt"
            if lbl.exists():
                cleaned = clean_label_file(lbl, stats)
                cleaned_by_stem[stem] = cleaned
                writer.writerow([stem, True, len(cleaned)])
            else:
                cleaned_by_stem[stem] = []
                writer.writerow([stem, False, 0])
    print(f"  Partial clean manifest -> {manifest_path}")

    # --- Decide which stems make it into the dataset ----------------------
    usable_stems = [
        s for s in sorted(image_stems)
        if cleaned_by_stem[s] or args.include_unlabeled
    ]
    if not usable_stems:
        raise SystemExit("ERROR: no usable labeled images found.")

    # --- Split ------------------------------------------------------------
    splits = grouped_block_split(
        usable_stems, args.block_size, args.val_ratio, args.test_ratio
    )
    print("Split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # --- Build output tree ------------------------------------------------
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in SPLIT_NAMES:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    per_split_boxes = {s: 0 for s in SPLIT_NAMES}
    for split, stems in splits.items():
        for stem in stems:
            src_img = find_image_for_stem(images_dir, stem)
            if src_img is None:
                continue
            dst_img = out_dir / "images" / split / src_img.name
            _materialize(src_img, dst_img, args.copy_images)

            label_lines = cleaned_by_stem.get(stem, [])
            per_split_boxes[split] += len(label_lines)
            dst_lbl = out_dir / "labels" / split / f"{stem}.txt"
            dst_lbl.write_text(
                ("\n".join(label_lines) + "\n") if label_lines else "",
                encoding="utf-8",
            )

    # --- data.yaml --------------------------------------------------------
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "# Auto-generated by prepare_leaf_dataset.py — do not edit by hand.\n"
        f"path: {out_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: leaf\n",
        encoding="utf-8",
    )
    print(f"Wrote {data_yaml}")

    # --- Report -----------------------------------------------------------
    bpi = stats.boxes_per_image
    boxes_per_image_stats = {
        "min": min(bpi) if bpi else 0,
        "max": max(bpi) if bpi else 0,
        "mean": round(sum(bpi) / len(bpi), 3) if bpi else 0,
    }
    report = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "num_images": len(image_paths),
            "num_label_files": len(label_paths),
            "num_unlabeled_images": len(unlabeled),
            "num_orphan_labels": len(orphan_labels),
            "orphan_labels_sample": orphan_labels[:20],
            "unlabeled_images_sample": unlabeled[:20],
        },
        "label_cleaning": {
            "files_seen": stats.files_seen,
            "files_with_boxes": stats.files_with_boxes,
            "empty_after_clean": stats.empty_files,
            "total_boxes_raw": stats.total_boxes_raw,
            "total_boxes_kept": stats.total_boxes_kept,
            "lines_with_trackid_fixed": stats.lines_with_trackid,
            "boxes_dropped_bad_class": stats.boxes_dropped_bad_class,
            "boxes_dropped_degenerate": stats.boxes_dropped_degenerate,
            "boxes_dropped_malformed": stats.boxes_dropped_malformed,
            "boxes_clamped_to_unit_range": stats.boxes_clamped,
            "tiny_boxes_lt_0.1pct_area": stats.tiny_boxes,
            "boxes_per_image": boxes_per_image_stats,
        },
        "split": {
            "strategy": "grouped temporal block split (anti-leakage)",
            "block_size": args.block_size,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "counts": {s: len(v) for s, v in splits.items()},
            "boxes": per_split_boxes,
            "include_unlabeled_as_background": args.include_unlabeled,
        },
        "assumptions": [
            "Class 0 == 'leaf'; any other class index is dropped.",
            "A 6th column in a raw label line is a CVAT track id and is removed.",
            "Coordinates are normalized; values outside [0,1] are clamped.",
            "Boxes with non-positive width/height are dropped.",
            "Adjacent video frames are near-duplicates, so splitting is done by "
            "contiguous frame blocks to avoid train/test leakage.",
            "By default unlabeled frames are excluded (no negative/background "
            "supervision); pass --include-unlabeled to keep them as background.",
        ],
        "output": {
            "dataset_dir": str(out_dir),
            "data_yaml": str(data_yaml),
        },
    }
    report_json = reports_dir / "dataset_report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Promote the partial manifest to final.
    final_manifest = reports_dir / "clean_manifest.csv"
    if final_manifest.exists():
        final_manifest.unlink()
    manifest_path.rename(final_manifest)

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Raw boxes:    {stats.total_boxes_raw}")
    print(f"  Kept boxes:   {stats.total_boxes_kept}")
    print(f"  Track-id lines fixed: {stats.lines_with_trackid}")
    print(f"  Dropped (class/degenerate/malformed): "
          f"{stats.boxes_dropped_bad_class}/{stats.boxes_dropped_degenerate}/"
          f"{stats.boxes_dropped_malformed}")
    print(f"  Splits: {{ {', '.join(f'{k}:{len(v)}' for k,v in splits.items())} }}")
    print(f"  Report: {report_json}")
    print(f"  Manifest: {final_manifest}")
    print(f"  data.yaml: {data_yaml}")


if __name__ == "__main__":
    main()
