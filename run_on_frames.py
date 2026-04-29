"""Run the SmartLeafDetection pipeline on individual image frames.

Since the full pipeline expects a drone video, this script adapts the
pipeline stages to work on static images directly:

  Frame → Leaf Detection (YOLOv11) → Leaf Tracking → Normalization →
  Disease Classification (ResNet50) → Temporal Aggregation →
  Plant Status → Report Export

Usage:
    # With trained weights:
    python run_on_frames.py --frames 1_original.jpeg 2_original.jpeg 3_original.jpeg

    # Demo mode (random classifier weights — predictions will be random):
    python run_on_frames.py --frames 1_original.jpeg 2_original.jpeg 3_original.jpeg --demo

    # With specific weight paths:
    python run_on_frames.py --frames 1_original.jpeg 2_original.jpeg 3_original.jpeg \
        --leaf-weights yolo11_leaves.pt \
        --classifier-weights resnet50_tomato.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def find_leaf_regions(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find leaf-like regions using multi-range HSV green segmentation.

    Uses multiple HSV ranges to capture different shades of green
    (bright green, dark green, yellowish-green) common in drone imagery.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Multiple green ranges to catch various leaf shades
    ranges = [
        (np.array([20, 20, 20]), np.array([95, 255, 255])),   # broad green
        (np.array([15, 15, 40]), np.array([100, 255, 255])),   # yellowish-green
        (np.array([25, 10, 10]), np.array([85, 255, 200])),    # dark/muted green
    ]

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Morphological cleanup
    kernel = np.ones((10, 10), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter: at least 0.1% of image, at most 60%
    min_area = (h * w) * 0.001
    max_area = (h * w) * 0.6
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, bw, bh = cv2.boundingRect(cnt)
            boxes.append((x, y, x + bw, y + bh))

    # If still nothing found, fall back to a grid-based approach
    # (split the frame into quadrants and use each as a "leaf region")
    if not boxes:
        qh, qw = h // 2, w // 2
        boxes = [
            (0, 0, qw, qh),
            (qw, 0, w, qh),
            (0, qh, qw, h),
            (qw, qh, w, h),
        ]

    return boxes






def extract_red_boxes(annotated_frame: np.ndarray, min_area: int = 800) -> list[tuple[int, int, int, int]]:
    """Extract red bounding box regions from an annotated image.

    Uses the red outlines as *walls/separators*: dilates them to form
    solid barriers, inverts the mask, then finds enclosed non-red
    regions via connected components.  Each enclosed region corresponds
    to one leaf box.  This avoids the classic problem of overlapping or
    touching red outlines merging into one contour or fragmenting into
    dozens of pieces.
    """
    h, w = annotated_frame.shape[:2]
    img_area = h * w
    hsv = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2HSV)

    # Red wraps around in HSV — two ranges (slightly wider to catch all reds)
    mask1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 100, 80]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Dilate the red lines so they form solid walls between regions.
    # Use iterations=2 (not 3) to avoid eating too far into small boxes.
    kernel = np.ones((7, 7), np.uint8)
    red_walls = cv2.dilate(red_mask, kernel, iterations=2)

    # Also add a thin border around the image edges so that boxes touching
    # the frame boundary are properly enclosed (otherwise they leak into
    # the background component).
    border = 3
    red_walls[:border, :] = 255
    red_walls[-border:, :] = 255
    red_walls[:, :border] = 255
    red_walls[:, -border:] = 255

    # Invert: non-red enclosed regions become white foreground
    inverse = cv2.bitwise_not(red_walls)

    # Connected components — each enclosed area is a separate component
    num_labels, labels = cv2.connectedComponents(inverse)

    boxes = []
    for label_id in range(1, num_labels):  # skip background (0)
        component_mask = (labels == label_id).astype(np.uint8)
        component_area = int(component_mask.sum())

        # Must be meaningful size (>0.5% of image) but not the entire
        # background (<40% of image)
        if component_area < img_area * 0.005:
            continue
        if component_area > img_area * 0.40:
            continue

        coords = cv2.findNonZero(component_mask)
        if coords is None:
            continue
        x, y, bw, bh = cv2.boundingRect(coords)

        # Filter very thin slivers
        aspect = bw / max(bh, 1)
        if aspect < 0.15 or aspect > 7.0:
            continue
        if bw < 25 or bh < 25:
            continue

        boxes.append((x, y, x + bw, y + bh))

    # Sort by area descending
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes


def run_with_annotations(
    frame_paths: list[str],
    annotated_paths: list[str],
    disease_weights: str | None,
    output_path: str,
    output_dir: str,
    confidence: float = 0.25,
) -> None:
    """Use red bounding boxes from annotated images as leaf locations.

    Extracts leaf regions from annotated frames, then classifies each
    crop using the disease YOLO model (or random weights if no model).
    """
    from smart_leaf_detection.models import DiseaseRecord
    from smart_leaf_detection.report_exporter import ReportExporter

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load disease model if available
    model = None
    if disease_weights and Path(disease_weights).exists():
        from ultralytics import YOLO
        print(f"Loading disease model: {disease_weights}")
        model = YOLO(disease_weights)
        print(f"  Classes: {model.names}")
    else:
        print("No disease model — will use bounding boxes only (no classification)")

    exporter = ReportExporter(output_format="json")
    all_records: list[DiseaseRecord] = []

    for frame_idx, (frame_path, ann_path) in enumerate(zip(frame_paths, annotated_paths)):
        print(f"\n--- Processing: {frame_path} (annotations from {ann_path}) ---")
        frame = cv2.imread(frame_path)
        annotated = cv2.imread(ann_path)
        if frame is None:
            print(f"  WARNING: Could not read {frame_path}, skipping.")
            continue
        if annotated is None:
            print(f"  WARNING: Could not read {ann_path}, skipping.")
            continue

        h, w = frame.shape[:2]
        print(f"  Image size: {w}x{h}")

        # Extract red boxes from annotated image
        leaf_boxes = extract_red_boxes(annotated)
        print(f"  Found {len(leaf_boxes)} leaf regions from annotations")

        if not leaf_boxes:
            print("  No leaf regions found in annotations, skipping.")
            continue

        plant_id = frame_idx + 1
        leaf_viz_results: list[tuple[tuple[int, int, int, int], str, float]] = []
        disease_counts: dict[str, float] = {}
        healthy_count = 0
        diseased_count = 0

        for i, (x1, y1, x2, y2) in enumerate(leaf_boxes):
            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            leaf_crop = frame[y1:y2, x1:x2]
            if leaf_crop.size == 0:
                continue

            label = "unknown"
            conf_val = 0.0

            if model is not None:
                # Classify the crop with the disease YOLO model
                results = model(leaf_crop, conf=0.05, verbose=False)
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    # Take highest confidence detection
                    best_idx = int(boxes.conf.argmax())
                    cls_id = int(boxes.cls[best_idx])
                    conf_val = float(boxes.conf[best_idx])
                    label = model.names[cls_id]
                else:
                    # No detection on crop — try classifying the whole crop as one region
                    # Use the model's class with highest probability
                    label = "undetected"
                    conf_val = 0.0
            else:
                label = "leaf"
                conf_val = 1.0

            print(f"  Leaf {i+1}: bbox=({x1},{y1},{x2},{y2}) → {label} (conf={conf_val:.3f})")

            leaf_viz_results.append(((x1, y1, x2, y2), label, conf_val))

            if label.lower() in ("healthy", "undetected", "unknown", "leaf"):
                healthy_count += 1
            else:
                diseased_count += 1
                disease_counts[label] = disease_counts.get(label, 0) + conf_val

        total = healthy_count + diseased_count
        plant_status = "diseased" if diseased_count > 0 else "healthy"
        print(f"  Plant {plant_id} status: {plant_status} "
              f"({diseased_count}/{total} leaves diseased)")

        annotated_frame = draw_results_on_frame(
            frame, leaf_viz_results, plant_status, plant_id,
        )
        stem = Path(frame_path).stem
        out_img_path = out_dir / f"{stem}_annotated.jpeg"
        cv2.imwrite(str(out_img_path), annotated_frame)
        print(f"  Annotated image saved: {out_img_path}")

        if plant_status == "diseased":
            evidence = dict(disease_counts)
            evidence["leaf_count"] = float(total)
            evidence["diseased_leaf_count"] = float(diseased_count)

            all_records.append(DiseaseRecord(
                flight_id="static_frames",
                plant_id=plant_id,
                gps=None,
                disease_labels=list(disease_counts.keys()),
                evidence_metrics=evidence,
            ))

    exporter.export(all_records, output_path)
    print(f"\n{'='*60}")
    print(f"Disease report saved to: {output_path}")
    print(f"Annotated images saved to: {output_dir}/")
    print(f"Total diseased plants: {len(all_records)}")

    with open(output_path) as f:
        data = json.load(f)
    print(json.dumps(data, indent=2))


# Disease label → color mapping for visualization
_DISEASE_COLORS: dict[str, tuple[int, int, int]] = {
    "healthy":                          (0, 200, 0),     # green
    "Bacterial_spot":                   (0, 0, 255),     # red
    "Early_blight":                     (0, 100, 255),   # orange
    "Late_blight":                      (0, 50, 200),    # dark orange
    "Leaf_Mold":                        (200, 0, 200),   # purple
    "Septoria_leaf_spot":               (255, 0, 150),   # magenta
    "Spider_mites":                     (0, 200, 255),   # yellow
    "Target_Spot":                      (255, 100, 0),   # blue-ish
    "Tomato_Yellow_Leaf_Curl_Virus":    (0, 255, 255),   # cyan
    "Tomato_mosaic_virus":              (150, 0, 255),   # pink
}


def draw_results_on_frame(
    frame: np.ndarray,
    leaf_results: list[tuple[tuple[int, int, int, int], str, float]],
    plant_status: str,
    plant_id: int,
) -> np.ndarray:
    """Draw bounding boxes and disease labels on a frame.

    Args:
        frame: BGR image.
        leaf_results: List of ((x1,y1,x2,y2), disease_label, confidence).
        plant_status: "healthy" or "diseased".
        plant_id: Plant ID for the header.

    Returns:
        Annotated copy of the frame.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw header bar
    bar_h = 40
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    status_color = (0, 200, 0) if plant_status == "healthy" else (0, 0, 255)
    status_text = f"Plant {plant_id}: {plant_status.upper()}"
    cv2.putText(annotated, status_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # Draw each leaf bbox with disease label
    for (x1, y1, x2, y2), label, conf in leaf_results:
        color = _DISEASE_COLORS.get(label, (255, 255, 255))
        thickness = 3 if label != "healthy" else 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label background
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 5, th + 5)
        cv2.rectangle(annotated, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), color, -1)
        cv2.putText(annotated, text, (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated


def run_demo_mode(frame_paths: list[str], output_path: str, output_dir: str) -> None:
    """Run with random ResNet50 weights and simulated leaf detections.

    No trained YOLO weights needed — leaf regions are extracted using
    green-color segmentation. Saves annotated output images.
    """
    from smart_leaf_detection.disease_classifier import DiseaseClassifier
    from smart_leaf_detection.leaf_normalizer import LeafNormalizer
    from smart_leaf_detection.models import (
        AggregatedLabel,
        ClassificationResult,
        DiseaseRecord,
        GPSCoordinate,
    )
    from smart_leaf_detection.plant_status_engine import PlantStatusEngine
    from smart_leaf_detection.report_exporter import ReportExporter
    from smart_leaf_detection.temporal_aggregator import TemporalAggregator

    class_names = [
        "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
        "Septoria_leaf_spot", "Spider_mites", "Target_Spot",
        "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", "healthy",
    ]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ResNet50 with random weights (demo mode)...")
    classifier = DiseaseClassifier(weights_path=None, class_names=class_names, device="cpu")
    normalizer = LeafNormalizer()
    aggregator = TemporalAggregator(window_size=15, dual_threshold_enabled=False)
    status_engine = PlantStatusEngine(top_k=3)
    exporter = ReportExporter(output_format="json")

    all_records: list[DiseaseRecord] = []
    leaf_id_counter = 0

    for frame_idx, frame_path in enumerate(frame_paths):
        print(f"\n--- Processing: {frame_path} ---")
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"  WARNING: Could not read {frame_path}, skipping.")
            continue

        h, w = frame.shape[:2]
        print(f"  Image size: {w}x{h}")

        leaf_boxes = find_leaf_regions(frame)
        print(f"  Found {len(leaf_boxes)} leaf-like regions")

        plant_id = frame_idx + 1
        leaf_labels: list[AggregatedLabel] = []
        leaf_viz_results: list[tuple[tuple[int, int, int, int], str, float]] = []

        for box_idx, (x1, y1, x2, y2) in enumerate(leaf_boxes):
            leaf_id_counter += 1
            leaf_id = leaf_id_counter

            leaf_roi = frame[y1:y2, x1:x2]
            if leaf_roi.size == 0:
                continue

            tensor = normalizer.normalize(leaf_roi)
            result = classifier.classify(tensor, leaf_id=leaf_id, plant_id=plant_id)
            conf = result.probability_vector[result.predicted_class]
            print(f"  Leaf {box_idx+1}: bbox=({x1},{y1},{x2},{y2}) → {result.predicted_class} "
                  f"(conf={conf:.3f})")

            agg_label = aggregator.update(result)
            leaf_labels.append(agg_label)
            leaf_viz_results.append(((x1, y1, x2, y2), result.predicted_class, conf))

        if not leaf_labels:
            continue

        status = status_engine.compute_status(leaf_labels)
        print(f"  Plant {plant_id} status: {status.status} "
              f"({status.diseased_leaf_count}/{status.leaf_count} leaves diseased)")

        # Draw annotated output image
        annotated_frame = draw_results_on_frame(
            frame, leaf_viz_results, status.status, plant_id,
        )
        stem = Path(frame_path).stem
        out_img_path = out_dir / f"{stem}_result.jpeg"
        cv2.imwrite(str(out_img_path), annotated_frame)
        print(f"  Annotated image saved: {out_img_path}")

        if status.status == "diseased":
            evidence: dict[str, float] = {}
            for disease_label, confidence in status.top_diseases:
                evidence[disease_label] = confidence
            evidence["leaf_count"] = float(status.leaf_count)
            evidence["diseased_leaf_count"] = float(status.diseased_leaf_count)

            all_records.append(DiseaseRecord(
                flight_id="static_frames",
                plant_id=plant_id,
                gps=None,
                disease_labels=[d for d, _ in status.top_diseases],
                evidence_metrics=evidence,
            ))

    exporter.export(all_records, output_path)
    print(f"\n{'='*60}")
    print(f"Disease report saved to: {output_path}")
    print(f"Annotated images saved to: {output_dir}/")
    print(f"Total diseased plants: {len(all_records)}")

    with open(output_path) as f:
        data = json.load(f)
    print(json.dumps(data, indent=2))


def run_with_weights(
    frame_paths: list[str],
    leaf_weights: str,
    classifier_weights: str | None,
    output_path: str,
    output_dir: str,
) -> None:
    """Run with trained YOLO leaf detector and optional ResNet50 weights."""
    from smart_leaf_detection.disease_classifier import DiseaseClassifier
    from smart_leaf_detection.leaf_detector import LeafDetector
    from smart_leaf_detection.leaf_normalizer import LeafNormalizer
    from smart_leaf_detection.leaf_tracker import LeafTracker
    from smart_leaf_detection.models import (
        AggregatedLabel,
        DiseaseRecord,
    )
    from smart_leaf_detection.plant_status_engine import PlantStatusEngine
    from smart_leaf_detection.report_exporter import ReportExporter
    from smart_leaf_detection.temporal_aggregator import TemporalAggregator

    class_names = [
        "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
        "Septoria_leaf_spot", "Spider_mites", "Target_Spot",
        "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", "healthy",
    ]

    print(f"Loading leaf detector: {leaf_weights}")
    leaf_detector = LeafDetector(weights_path=leaf_weights, confidence_threshold=0.25)

    print(f"Loading disease classifier: {classifier_weights or 'random weights'}")
    classifier = DiseaseClassifier(
        weights_path=classifier_weights, class_names=class_names, device="cpu",
    )
    normalizer = LeafNormalizer()
    aggregator = TemporalAggregator(window_size=15, dual_threshold_enabled=False)
    status_engine = PlantStatusEngine(top_k=3)
    exporter = ReportExporter(output_format="json")

    all_records: list[DiseaseRecord] = []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame_path in enumerate(frame_paths):
        print(f"\n--- Processing: {frame_path} ---")
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"  WARNING: Could not read {frame_path}, skipping.")
            continue

        h, w = frame.shape[:2]
        print(f"  Image size: {w}x{h}")

        plant_id = frame_idx + 1

        # Detect leaves using trained YOLO model
        leaf_detections = leaf_detector.detect(frame, plant_id=plant_id)
        print(f"  YOLO detected {len(leaf_detections)} leaves")

        if not leaf_detections:
            print("  No leaves detected, skipping frame.")
            continue

        # Track leaves
        leaf_tracker = LeafTracker(plant_id=plant_id)
        tracked_leaves = leaf_tracker.update(leaf_detections)

        leaf_labels: list[AggregatedLabel] = []
        leaf_viz_results: list[tuple[tuple[int, int, int, int], str, float]] = []

        for tl in tracked_leaves:
            x1, y1, x2, y2 = (int(tl.bbox[0]), int(tl.bbox[1]),
                               int(tl.bbox[2]), int(tl.bbox[3]))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            leaf_roi = frame[y1:y2, x1:x2]
            if leaf_roi.size == 0:
                continue

            tensor = normalizer.normalize(leaf_roi)
            result = classifier.classify(tensor, leaf_id=tl.leaf_id, plant_id=plant_id)
            conf = result.probability_vector[result.predicted_class]
            print(f"  Leaf {tl.leaf_id}: bbox=({x1},{y1},{x2},{y2}) → {result.predicted_class} "
                  f"(conf={conf:.3f})")

            agg_label = aggregator.update(result)
            leaf_labels.append(agg_label)
            leaf_viz_results.append(((x1, y1, x2, y2), result.predicted_class, conf))

        if not leaf_labels:
            continue

        status = status_engine.compute_status(leaf_labels)
        print(f"  Plant {plant_id} status: {status.status} "
              f"({status.diseased_leaf_count}/{status.leaf_count} leaves diseased)")

        annotated_frame = draw_results_on_frame(
            frame, leaf_viz_results, status.status, plant_id,
        )
        stem = Path(frame_path).stem
        out_img_path = out_dir / f"{stem}_result.jpeg"
        cv2.imwrite(str(out_img_path), annotated_frame)
        print(f"  Annotated image saved: {out_img_path}")

        if status.status == "diseased":
            evidence: dict[str, float] = {}
            for disease_label, confidence in status.top_diseases:
                evidence[disease_label] = confidence
            evidence["leaf_count"] = float(status.leaf_count)
            evidence["diseased_leaf_count"] = float(status.diseased_leaf_count)

            all_records.append(DiseaseRecord(
                flight_id="static_frames",
                plant_id=plant_id,
                gps=None,
                disease_labels=[d for d, _ in status.top_diseases],
                evidence_metrics=evidence,
            ))

    exporter.export(all_records, output_path)
    print(f"\n{'='*60}")
    print(f"Disease report saved to: {output_path}")
    print(f"Annotated images saved to: {output_dir}/")
    print(f"Total diseased plants: {len(all_records)}")

    with open(output_path) as f:
        data = json.load(f)
    print(json.dumps(data, indent=2))


def run_with_disease_yolo(
    frame_paths: list[str],
    weights_path: str,
    output_path: str,
    output_dir: str,
    confidence: float = 0.25,
) -> None:
    """Run with a single YOLO model that detects AND classifies diseases.

    This uses the trained disease-detection YOLO model directly (e.g. from
    the Roboflow tomato leaf diseases dataset). The model outputs bounding
    boxes with disease class labels in one pass — no separate ResNet needed.
    """
    from ultralytics import YOLO
    from smart_leaf_detection.models import DiseaseRecord
    from smart_leaf_detection.report_exporter import ReportExporter

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading disease detection model: {weights_path}")
    model = YOLO(weights_path)
    class_names = model.names  # {0: 'Bacterial Spot', 1: 'Early_Blight', ...}
    print(f"  Classes: {class_names}")

    exporter = ReportExporter(output_format="json")
    all_records: list[DiseaseRecord] = []

    for frame_idx, frame_path in enumerate(frame_paths):
        print(f"\n--- Processing: {frame_path} ---")
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"  WARNING: Could not read {frame_path}, skipping.")
            continue

        h, w = frame.shape[:2]
        print(f"  Image size: {w}x{h}")

        plant_id = frame_idx + 1

        # Run YOLO inference
        results = model(frame, conf=confidence, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            print("  No detections found.")
            continue

        leaf_viz_results: list[tuple[tuple[int, int, int, int], str, float]] = []
        disease_counts: dict[str, float] = {}
        healthy_count = 0
        diseased_count = 0

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].cpu())
            cls_id = int(boxes.cls[i].cpu())
            label = class_names[cls_id]

            # Skip oversized detections (likely whole-plant or background, not individual leaves)
            box_w, box_h = x2 - x1, y2 - y1
            max_leaf_ratio = 0.35  # leaf shouldn't be more than 35% of frame width or height
            if box_w > w * max_leaf_ratio and box_h > h * max_leaf_ratio:
                print(f"  Detection {i+1}: bbox=({x1},{y1},{x2},{y2}) → SKIPPED (too large: {box_w}x{box_h})")
                continue

            print(f"  Detection {i+1}: bbox=({x1},{y1},{x2},{y2}) → {label} (conf={conf:.3f})")

            leaf_viz_results.append(((x1, y1, x2, y2), label, conf))

            if label.lower() == "healthy":
                healthy_count += 1
            else:
                diseased_count += 1
                disease_counts[label] = disease_counts.get(label, 0) + conf

        total = healthy_count + diseased_count
        plant_status = "diseased" if diseased_count > 0 else "healthy"
        print(f"  Plant {plant_id} status: {plant_status} "
              f"({diseased_count}/{total} leaves diseased)")

        # Draw annotated output
        annotated_frame = draw_results_on_frame(
            frame, leaf_viz_results, plant_status, plant_id,
        )
        stem = Path(frame_path).stem
        out_img_path = out_dir / f"{stem}_yolo_merged.jpeg"
        cv2.imwrite(str(out_img_path), annotated_frame)
        print(f"  Annotated image saved: {out_img_path}")

        if plant_status == "diseased":
            evidence = dict(disease_counts)
            evidence["leaf_count"] = float(total)
            evidence["diseased_leaf_count"] = float(diseased_count)

            all_records.append(DiseaseRecord(
                flight_id="static_frames",
                plant_id=plant_id,
                gps=None,
                disease_labels=list(disease_counts.keys()),
                evidence_metrics=evidence,
            ))

    exporter.export(all_records, output_path)
    print(f"\n{'='*60}")
    print(f"Disease report saved to: {output_path}")
    print(f"Annotated images saved to: {output_dir}/")
    print(f"Total diseased plants: {len(all_records)}")

    with open(output_path) as f:
        data = json.load(f)
    print(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SmartLeafDetection on individual image frames"
    )
    parser.add_argument(
        "--frames", nargs="+", required=True,
        help="Image files to process (use the ORIGINAL clean frames, not annotated ones)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Demo mode: use green-color segmentation + random classifier weights "
             "(no trained models needed)",
    )
    parser.add_argument(
        "--disease-weights", default="weights/best.pt",
        help="Path to trained YOLO11 disease detection weights (default: weights/best.pt)",
    )
    parser.add_argument(
        "--annotated", nargs="+", default=None,
        help="Annotated images with red bounding boxes around leaves. "
             "Must match --frames in order (e.g. --frames 1_original.jpeg --annotated 1.jpeg)",
    )
    parser.add_argument(
        "--leaf-weights", default="yolo11_leaves.pt",
        help="Path to trained YOLOv11 leaf-only detection weights",
    )
    parser.add_argument(
        "--classifier-weights", default=None,
        help="Path to trained ResNet50 weights (None = random weights)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--output", default="disease_report.json",
        help="Output report path",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory to save annotated output images (default: output/)",
    )
    args = parser.parse_args()

    # Validate frames exist
    for fp in args.frames:
        if not Path(fp).exists():
            print(f"ERROR: Frame not found: {fp}")
            sys.exit(1)

    if args.demo:
        print("=" * 60)
        print("DEMO MODE — green segmentation + random classifier weights")
        print("Predictions will be RANDOM (no trained models loaded)")
        print("=" * 60)
        run_demo_mode(args.frames, args.output, args.output_dir)
    elif args.annotated:
        # Use red bounding boxes from annotated images as leaf locations
        if len(args.annotated) != len(args.frames):
            print(f"ERROR: --annotated count ({len(args.annotated)}) must match "
                  f"--frames count ({len(args.frames)})")
            sys.exit(1)
        for fp in args.annotated:
            if not Path(fp).exists():
                print(f"ERROR: Annotated image not found: {fp}")
                sys.exit(1)
        print("=" * 60)
        print("ANNOTATION MODE — using red boxes from annotated images")
        print("=" * 60)
        run_with_annotations(
            args.frames,
            annotated_paths=args.annotated,
            disease_weights=args.disease_weights,
            output_path=args.output,
            output_dir=args.output_dir,
            confidence=args.confidence,
        )
    elif args.disease_weights and Path(args.disease_weights).exists():
        # Single YOLO model that detects + classifies diseases
        print("=" * 60)
        print("DISEASE DETECTION MODE — trained YOLO11 model")
        print("=" * 60)
        run_with_disease_yolo(
            args.frames,
            weights_path=args.disease_weights,
            output_path=args.output,
            output_dir=args.output_dir,
            confidence=args.confidence,
        )
    else:
        # Two-stage: YOLO leaf detector + ResNet classifier
        if not Path(args.leaf_weights).exists():
            print(f"ERROR: Leaf weights not found: {args.leaf_weights}")
            print("Either train the model first (see training/ scripts) or use --demo mode.")
            sys.exit(1)

        run_with_weights(
            args.frames,
            leaf_weights=args.leaf_weights,
            classifier_weights=args.classifier_weights,
            output_path=args.output,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()

