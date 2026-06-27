#!/usr/bin/env python3
"""Build an ImageFolder leaf-disease classification dataset from the project's
YOLO **detection** datasets by cropping every labelled box.

Why: every disease dataset in this project is in YOLO bounding-box format, not
classification folders. A detection label is strictly richer than a class label
— each box is a labelled leaf region we can crop. This script turns those boxes
into a clean ``train/validation/test`` ImageFolder dataset, unifying class names
via ``class_mapping`` and preventing train/val/test leakage.

Output layout::

    datasets/leaf_clf/
        train/<canonical_class>/<dataset>__<source_id>__<box>.jpg
        validation/<canonical_class>/...
        test/<canonical_class>/...
        dataset_report.json
        class_distribution.png   (if matplotlib is available)

Leakage prevention: a source photo can appear (augmented) across datasets and
splits. We assign each *source image id* (filename with the Roboflow
``.rf.<hash>`` suffix stripped) to a single split using the priority
**test > validation > train**, scanning all datasets first (pass 1), then
cropping (pass 2). All crops of a given source image therefore land in exactly
one split, regardless of how many datasets contain it.

Usage:
    python training/disease_classification/build_classification_dataset.py
    python training/disease_classification/build_classification_dataset.py \
        --datasets tomato_diseases merged_diseases plantdoc tomato_6k \
        --padding 0.08 --min-size 24 --min-samples-per-class 25
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "training" / "disease_classification"))
from class_mapping import CANONICAL_CLASSES, build_id_to_canonical  # noqa: E402

# Per-dataset layout. For each dataset we list the (images_dir, labels_dir) for
# each canonical split that exists, plus where to read class ``names`` from.
# Paths are relative to PROJECT_ROOT.
DATASET_CONFIGS: dict[str, dict] = {
    "tomato_diseases": {
        "names_yaml": "datasets/tomato_diseases/data.yaml",
        "splits": {
            "train": ("datasets/tomato_diseases/train/images", "datasets/tomato_diseases/train/labels"),
            "validation": ("datasets/tomato_diseases/valid/images", "datasets/tomato_diseases/valid/labels"),
            "test": ("datasets/tomato_diseases/test/images", "datasets/tomato_diseases/test/labels"),
        },
    },
    "merged_diseases": {
        "names_yaml": "datasets/merged_diseases/data.yaml",
        "splits": {
            "train": ("datasets/merged_diseases/train/images", "datasets/merged_diseases/train/labels"),
            "validation": ("datasets/merged_diseases/valid/images", "datasets/merged_diseases/valid/labels"),
        },
    },
    "plantdoc": {
        "names_yaml": "PlantDoc.v1-resize-416x416.yolov11/data.yaml",
        "splits": {
            "train": ("PlantDoc.v1-resize-416x416.yolov11/train/images", "PlantDoc.v1-resize-416x416.yolov11/train/labels"),
            "test": ("PlantDoc.v1-resize-416x416.yolov11/test/images", "PlantDoc.v1-resize-416x416.yolov11/test/labels"),
        },
    },
    "tomato_6k": {
        "names_yaml": "datasets/tomato_6k/path.yaml",
        "splits": {
            "train": ("datasets/tomato_6k/sample_16_12_2023/images/train", "datasets/tomato_6k/sample_16_12_2023/labels/train"),
            "validation": ("datasets/tomato_6k/sample_16_12_2023/images/valid", "datasets/tomato_6k/sample_16_12_2023/labels/valid"),
            "test": ("datasets/tomato_6k/sample_16_12_2023/images/test", "datasets/tomato_6k/sample_16_12_2023/labels/test"),
        },
    },
}

_SPLIT_PRIORITY = {"test": 3, "validation": 2, "train": 1}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)


def _source_id(stem: str) -> str:
    """Strip the Roboflow ``.rf.<hash>`` augmentation suffix to get the source id."""
    return _RF_SUFFIX.sub("", stem)


def _load_names(names_yaml: Path) -> list[str] | dict[int, str]:
    data = yaml.safe_load(names_yaml.read_text(encoding="utf-8"))
    names = data.get("names")
    if names is None:
        raise SystemExit(f"No 'names' in {names_yaml}")
    return names


def _iter_label_files(labels_dir: Path):
    if not labels_dir.is_dir():
        return
    for txt in sorted(labels_dir.glob("*.txt")):
        yield txt


def _find_image(images_dir: Path, stem: str) -> tuple[Path | None, str | None]:
    """Locate the image for a label stem, tolerant of bad filesystem paths.

    Returns ``(path, None)`` on success, or ``(None, reason)`` where reason is
    one of ``image_not_found`` / ``image_path_too_long`` / ``invalid_image_path``.
    A single over-long or invalid filename must never crash the whole build, so
    every filesystem call here is guarded against ``OSError``.
    """
    try:
        for ext in _IMG_EXTS:
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                return p, None
        # case-insensitive / odd-extension fallback
        for p in images_dir.glob(f"{stem}.*"):
            if p.suffix.lower() in _IMG_EXTS:
                return p, None
    except OSError as exc:
        if getattr(exc, "errno", None) == errno.ENAMETOOLONG:
            return None, "image_path_too_long"
        return None, "invalid_image_path"
    return None, "image_not_found"


def _parse_boxes(label_file: Path) -> list[tuple[int, float, float, float, float]]:
    """Return list of (class_id, cx, cy, w, h) from a YOLO label file (first 5 cols)."""
    boxes = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        boxes.append((cid, cx, cy, w, h))
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS),
                    choices=list(DATASET_CONFIGS), help="Which source datasets to include")
    ap.add_argument("--out", default="datasets/leaf_clf", help="Output dataset root")
    ap.add_argument("--padding", type=float, default=0.08, help="Padding fraction added to each box side")
    ap.add_argument("--min-size", type=int, default=24, help="Discard crops with width or height < this (px)")
    ap.add_argument("--min-samples-per-class", type=int, default=25,
                    help="Drop canonical classes with fewer than this many crops total")
    ap.add_argument("--split-mode", choices=["existing", "resplit"], default="existing",
                    help="'existing' = honor each dataset's train/valid/test folders (with global "
                         "leakage priority test>val>train); 'resplit' = ignore the original folders "
                         "and build ONE fixed reproducible source-level split (val/test fractions below).")
    ap.add_argument("--val-frac", type=float, default=0.1, help="resplit only")
    ap.add_argument("--test-frac", type=float, default=0.1, help="resplit only")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = (PROJECT_ROOT / args.out).resolve()
    reports_dir = PROJECT_ROOT / "training" / "disease_classification" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Resolve per-dataset id->canonical maps and validate paths.
    id_maps: dict[str, dict[int, str | None]] = {}
    for ds in args.datasets:
        cfg = DATASET_CONFIGS[ds]
        names = _load_names(PROJECT_ROOT / cfg["names_yaml"])
        id_maps[ds] = build_id_to_canonical(names)

    # ---- Pass 1: assign each source image id to a single split ----
    # Both modes guarantee a source image lands in exactly ONE split, so the
    # test set can never contain an image used for training.
    source_split: dict[str, str] = {}
    if args.split_mode == "resplit":
        # Ignore the original folders; build one fixed reproducible 80/10/10
        # (configurable) split at the SOURCE-IMAGE level across all data.
        all_sids: set[str] = set()
        for ds in args.datasets:
            for split, (img_dir, lbl_dir) in DATASET_CONFIGS[ds]["splits"].items():
                for txt in _iter_label_files(PROJECT_ROOT / lbl_dir):
                    all_sids.add(_source_id(txt.stem))
        sids = sorted(all_sids)
        random.Random(args.seed).shuffle(sids)
        n = len(sids)
        n_test = int(round(n * args.test_frac))
        n_val = int(round(n * args.val_frac))
        test_ids = set(sids[:n_test])
        val_ids = set(sids[n_test:n_test + n_val])
        for sid in sids:
            source_split[sid] = ("test" if sid in test_ids
                                 else "validation" if sid in val_ids else "train")
    else:
        # Honor existing split folders; priority test > validation > train so a
        # source appearing in multiple datasets is pinned to its strictest split.
        source_split_prio: dict[str, int] = {}
        for ds in args.datasets:
            for split, (img_dir, lbl_dir) in DATASET_CONFIGS[ds]["splits"].items():
                for txt in _iter_label_files(PROJECT_ROOT / lbl_dir):
                    sid = _source_id(txt.stem)
                    prio = _SPLIT_PRIORITY[split]
                    if prio > source_split_prio.get(sid, 0):
                        source_split_prio[sid] = prio
                        source_split[sid] = split

    # ---- Pass 2: crop boxes into the assigned split ----
    counts: dict[str, dict[str, int]] = {s: defaultdict(int) for s in ("train", "validation", "test")}
    discarded = defaultdict(int)            # reason -> count
    unmapped_labels = defaultdict(int)      # "dataset:classid" -> count
    problem_files: list[dict] = []          # details of files we had to skip
    crop_sizes: list[tuple[int, int]] = []
    written = 0

    for ds in args.datasets:
        id_map = id_maps[ds]
        for split, (img_dir, lbl_dir) in DATASET_CONFIGS[ds]["splits"].items():
            images_dir = PROJECT_ROOT / img_dir
            labels_dir = PROJECT_ROOT / lbl_dir
            for txt in _iter_label_files(labels_dir):
                sid = _source_id(txt.stem)
                final_split = source_split.get(sid, split)
                img_path, find_reason = _find_image(images_dir, txt.stem)
                if img_path is None:
                    discarded[find_reason] += 1
                    if len(problem_files) < 100:  # cap so the report stays small
                        problem_files.append({
                            "dataset": ds,
                            "label_file": txt.name,
                            "image_stem": txt.stem[:150],
                            "reason": find_reason,
                        })
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    discarded["unreadable_image"] += 1
                    continue
                H, W = img.shape[:2]
                for bi, (cid, cx, cy, bw, bh) in enumerate(_parse_boxes(txt)):
                    canonical = id_map.get(cid, "__UNMAPPED__")
                    if canonical == "__UNMAPPED__":
                        unmapped_labels[f"{ds}:{cid}"] += 1
                        discarded["unmapped_class"] += 1
                        continue
                    if canonical is None:
                        discarded["ignored_class"] += 1
                        continue
                    # denormalize + padding
                    pw, ph = bw * (1 + 2 * args.padding), bh * (1 + 2 * args.padding)
                    x1 = int(round((cx - pw / 2) * W))
                    y1 = int(round((cy - ph / 2) * H))
                    x2 = int(round((cx + pw / 2) * W))
                    y2 = int(round((cy + ph / 2) * H))
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    if x2 - x1 < args.min_size or y2 - y1 < args.min_size:
                        discarded["too_small"] += 1
                        continue
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        discarded["empty_crop"] += 1
                        continue
                    # Safe, deterministic, length-bounded output filename so a
                    # very long source stem can't produce an over-long path.
                    safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", txt.stem)[:80]
                    digest = hashlib.md5(txt.stem.encode("utf-8")).hexdigest()[:8]
                    fname = f"{ds}__{safe_stem}__{digest}__{bi}.jpg"
                    dst_dir = out_root / final_split / canonical
                    try:
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        ok = cv2.imwrite(str(dst_dir / fname), crop,
                                         [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                    except OSError as exc:
                        reason = ("image_path_too_long"
                                  if getattr(exc, "errno", None) == errno.ENAMETOOLONG
                                  else "invalid_image_path")
                        discarded[f"write_{reason}"] += 1
                        if len(problem_files) < 100:
                            problem_files.append({"dataset": ds, "label_file": txt.name,
                                                  "image_stem": txt.stem[:150], "reason": f"write_{reason}"})
                        continue
                    if not ok:
                        discarded["write_failed"] += 1
                        continue
                    counts[final_split][canonical] += 1
                    crop_sizes.append((x2 - x1, y2 - y1))
                    written += 1

    # ---- Drop under-populated classes (count over all splits) ----
    total_per_class: dict[str, int] = defaultdict(int)
    for split in counts:
        for cls, n in counts[split].items():
            total_per_class[cls] += n
    dropped_classes = [c for c, n in total_per_class.items() if n < args.min_samples_per_class]
    for c in dropped_classes:
        for split in counts:
            counts[split].pop(c, None)
        # remove their folders
        for split in ("train", "validation", "test"):
            d = out_root / split / c
            if d.is_dir():
                for f in d.glob("*"):
                    f.unlink()
                d.rmdir()

    kept_classes = sorted(set(total_per_class) - set(dropped_classes),
                          key=lambda c: CANONICAL_CLASSES.index(c) if c in CANONICAL_CLASSES else 999)

    # ---- Report ----
    total = sum(counts[s][c] for s in counts for c in counts[s])
    avg_w = sum(w for w, _ in crop_sizes) / len(crop_sizes) if crop_sizes else 0
    avg_h = sum(h for _, h in crop_sizes) / len(crop_sizes) if crop_sizes else 0
    per_class_total = {c: sum(counts[s].get(c, 0) for s in counts) for c in kept_classes}
    max_c = max(per_class_total.values()) if per_class_total else 0
    min_c = min(per_class_total.values()) if per_class_total else 0
    report = {
        "datasets_used": args.datasets,
        "params": {"padding": args.padding, "min_size": args.min_size,
                   "min_samples_per_class": args.min_samples_per_class, "seed": args.seed,
                   "split_mode": args.split_mode,
                   "val_frac": args.val_frac, "test_frac": args.test_frac},
        "total_crops": total,
        "classes": kept_classes,
        "per_split_counts": {s: dict(counts[s]) for s in counts},
        "per_class_total": per_class_total,
        "class_imbalance_ratio": round(max_c / min_c, 2) if min_c else None,
        "dropped_classes_below_threshold": dropped_classes,
        "discarded": dict(discarded),
        "problem_files": problem_files,
        "unmapped_labels": dict(unmapped_labels),
        "avg_crop_size": {"width": round(avg_w, 1), "height": round(avg_h, 1)},
        "leakage_policy": (
            f"source image assigned to ONE split (mode={args.split_mode}); "
            + ("priority test>validation>train" if args.split_mode == "existing"
               else f"fixed seeded {1-args.val_frac-args.test_frac:.0%}/{args.val_frac:.0%}/{args.test_frac:.0%} source-level split")
            + " — test never shares an image with train"),
        "output_root": str(out_root),
    }
    report_path = reports_dir / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Emit the human-reviewable class-mapping report alongside the dataset report.
    try:
        from dump_class_mapping import write_mapping_report
        write_mapping_report(reports_dir / "CLASS_MAPPING_REPORT.md", args.datasets)
    except Exception as exc:  # non-fatal
        print(f"(class-mapping report skipped: {exc})")

    # Class-distribution plot (best-effort; skipped if matplotlib unavailable).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        splits = ["train", "validation", "test"]
        x = np.arange(len(kept_classes))
        width = 0.26
        fig, ax = plt.subplots(figsize=(max(8, len(kept_classes) * 1.1), 5))
        for i, s in enumerate(splits):
            ax.bar(x + (i - 1) * width, [counts[s].get(c, 0) for c in kept_classes],
                   width, label=s)
        ax.set_xticks(x); ax.set_xticklabels(kept_classes, rotation=45, ha="right")
        ax.set_ylabel("crops"); ax.set_title("Leaf-disease crops per class / split")
        ax.legend(); fig.tight_layout()
        fig.savefig(reports_dir / "class_distribution.png", dpi=130)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"(class_distribution plot skipped: {exc})")

    print("=" * 60)
    print(f"Wrote {written} crops to {out_root}")
    print(f"Classes kept ({len(kept_classes)}): {kept_classes}")
    if dropped_classes:
        print(f"Dropped (< {args.min_samples_per_class}): {dropped_classes}")
    if unmapped_labels:
        print(f"WARNING unmapped labels: {dict(unmapped_labels)}")
    print(f"Discarded: {dict(discarded)}")
    if problem_files:
        print(f"Skipped {len(problem_files)} problematic file(s) (e.g. long/invalid paths); "
              f"see 'problem_files' in {report_path.name}. First: {problem_files[0]}")
    print(f"Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
