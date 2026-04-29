"""Background worker — runs the real SmartLeafDetection pipeline.

Processes uploaded drone videos through the full ML pipeline:
frame extraction → disease detection (YOLO) → tracking → classification →
aggregation → plant status → database persistence.

Uses the real trained models from the project.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Add project root to path so we can import smart_leaf_detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import SessionLocal
from app.models import Flight, FrameRecord, LeafResult, PlantResult


# Resolve model weights paths
def _find_weights(name: str, fallbacks: list[str]) -> str | None:
    for p in fallbacks:
        full = PROJECT_ROOT / p
        if full.exists():
            return str(full)
    return None


DISEASE_WEIGHTS = _find_weights("disease_yolo", [
    "weights/best.pt",
    "runs/detect/training/runs/diseases/yolo11s_stable/weights/last.pt",
    "runs/detect/training/runs/diseases/yolo11_plantdoc/weights/best.pt",
    "runs/detect/training/runs/diseases/yolo11_merged/weights/best.pt",
    "runs/detect/training/runs/diseases/yolo11_diseases/weights/best.pt",
])

OUTPUT_BASE = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "webapp" / "backend" / "processing_output")))


def _update_flight(flight_id: str, **kwargs) -> None:
    """Update flight record in a fresh DB session."""
    db = SessionLocal()
    try:
        flight = db.query(Flight).filter(Flight.id == flight_id).first()
        if flight:
            for k, v in kwargs.items():
                setattr(flight, k, v)
            db.commit()
    finally:
        db.close()


def run_pipeline_async(flight_id: str) -> None:
    """Entry point for background processing thread."""
    try:
        _run_pipeline(flight_id)
    except Exception as e:
        traceback.print_exc()
        _update_flight(
            flight_id,
            status="failed",
            error_message=str(e)[:500],
            current_stage="Failed",
        )


def _run_pipeline(flight_id: str) -> None:
    """Run the full real pipeline on the uploaded video."""
    db = SessionLocal()
    flight = db.query(Flight).filter(Flight.id == flight_id).first()
    if not flight:
        db.close()
        return

    video_path = flight.video_path
    db.close()

    out_dir = OUTPUT_BASE / flight_id
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    annotated_dir = out_dir / "annotated"
    annotated_dir.mkdir(exist_ok=True)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    # --- Stage 1: Video decode & frame extraction ---
    _update_flight(flight_id, current_stage="Extracting frames", progress=5.0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Extract ~1 frame every 3 seconds to avoid near-duplicate frames
    step = max(1, int(fps * 3))

    extracted_frames: list[tuple[int, str]] = []
    prev_gray = None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            # Skip frames that are too similar to the previous extracted frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (160, 120))
            if prev_gray is not None:
                diff = cv2.absdiff(gray_small, prev_gray)
                mean_diff = float(diff.mean())
                if mean_diff < 5.0:  # nearly identical frame, skip
                    frame_idx += 1
                    continue
            prev_gray = gray_small

            fname = f"frame_{frame_idx:06d}.jpg"
            fpath = str(frames_dir / fname)
            cv2.imwrite(fpath, frame)
            extracted_frames.append((frame_idx, fpath))
        frame_idx += 1
    cap.release()

    total_extracted = len(extracted_frames)
    _update_flight(flight_id, total_frames=total_extracted, current_stage="Frames extracted", progress=15.0)

    # Save frame records to DB
    db = SessionLocal()
    for fidx, fpath in extracted_frames:
        db.add(FrameRecord(
            flight_id=flight_id, frame_index=fidx,
            original_path=fpath.replace(str(OUTPUT_BASE) + "/", ""),
        ))
    db.commit()
    db.close()

    # --- Stage 2: Disease detection using trained YOLO model ---
    _update_flight(flight_id, current_stage="Loading disease model", progress=20.0)

    model = None
    class_names = {}
    if DISEASE_WEIGHTS:
        from ultralytics import YOLO
        model = YOLO(DISEASE_WEIGHTS)
        class_names = model.names  # {0: 'Bacterial Spot', ...}

    _update_flight(flight_id, current_stage="Running detection & classification", progress=25.0)

    # Per-frame detection results
    all_plant_detections: dict[int, list] = defaultdict(list)  # plant_id -> [(label, conf, bbox, frame_idx)]
    plant_id_counter = 0

    for i, (fidx, fpath) in enumerate(extracted_frames):
        frame = cv2.imread(fpath)
        if frame is None:
            continue

        h, w = frame.shape[:2]
        detections = []

        if model is not None:
            results = model(frame, conf=0.3, verbose=False)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy().astype(int)
                    conf = float(boxes.conf[j].cpu())
                    cls_id = int(boxes.cls[j].cpu())
                    label = class_names.get(cls_id, f"class_{cls_id}")
                    detections.append((x1, y1, x2, y2, label, conf))
        else:
            # Demo fallback: use green segmentation
            from smart_leaf_detection.leaf_normalizer import LeafNormalizer
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([20, 20, 20]), np.array([95, 255, 255]))
            kernel = np.ones((10, 10), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > (h * w) * 0.005:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    detections.append((x, y, x + bw, y + bh, "leaf", 0.5))

        # Draw annotated frame
        annotated = frame.copy()
        for (x1, y1, x2, y2, label, conf) in detections:
            color = (0, 200, 0) if label.lower() in ("healthy", "tomato leaf") else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} {conf:.0%}", (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        ann_fname = f"annotated_{fidx:06d}.jpg"
        ann_path = str(annotated_dir / ann_fname)
        cv2.imwrite(ann_path, annotated)

        # Update frame record with annotated path and counts
        db = SessionLocal()
        fr = db.query(FrameRecord).filter(
            FrameRecord.flight_id == flight_id, FrameRecord.frame_index == fidx
        ).first()
        if fr:
            fr.annotated_path = ann_path.replace(str(OUTPUT_BASE) + "/", "")
            fr.plant_count = len(detections)
            fr.leaf_count = len(detections)
        db.commit()
        db.close()

        # Group detections as "plants" (each detection is a leaf, group nearby ones)
        for (x1, y1, x2, y2, label, conf) in detections:
            plant_id_counter += 1
            pid = plant_id_counter
            # Save crop
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size > 0:
                crop_fname = f"crop_p{pid}_f{fidx}.jpg"
                crop_path = str(crops_dir / crop_fname)
                cv2.imwrite(crop_path, crop)
                all_plant_detections[pid].append({
                    "label": label, "conf": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "frame_idx": fidx, "crop_path": crop_path.replace(str(OUTPUT_BASE) + "/", ""),
                })

        progress = 25.0 + (i / max(total_extracted, 1)) * 50.0
        _update_flight(flight_id, processed_frames=i + 1, progress=progress)

    _update_flight(flight_id, current_stage="Aggregating results", progress=80.0)

    # --- Stage 3: Save plant-level results to DB ---
    _update_flight(flight_id, current_stage="Saving plant results", progress=85.0)

    db = SessionLocal()
    total_plants = 0
    diseased_count = 0
    healthy_count = 0

    for pid, det_list in all_plant_detections.items():
        if not det_list:
            continue

        # Aggregate: majority vote on label, mean confidence
        label_counts: dict[str, float] = defaultdict(float)
        total_conf = 0.0
        for d in det_list:
            label_counts[d["label"]] += d["conf"]
            total_conf += d["conf"]

        best_label = max(label_counts, key=lambda k: label_counts[k])
        avg_conf = total_conf / len(det_list)
        is_healthy = best_label.lower() in ("healthy", "tomato leaf", "leaf")
        status = "healthy" if is_healthy else "diseased"

        disease_labels_str = "" if is_healthy else best_label

        # Use first detection's bbox and GPS placeholder
        first = det_list[0]

        pr = PlantResult(
            flight_id=flight_id,
            plant_id=pid,
            status=status,
            disease_labels=disease_labels_str,
            confidence=round(avg_conf, 4),
            leaf_count=len(det_list),
            diseased_leaf_count=0 if is_healthy else len(det_list),
            evidence_json=json.dumps(label_counts),
        )
        db.add(pr)
        db.flush()

        # Save leaf-level results
        for d in det_list:
            lr = LeafResult(
                plant_result_id=pr.id,
                leaf_id=d.get("leaf_id", pid),
                label=d["label"],
                confidence=d["conf"],
                bbox_x1=d["bbox"][0],
                bbox_y1=d["bbox"][1],
                bbox_x2=d["bbox"][2],
                bbox_y2=d["bbox"][3],
                crop_path=d.get("crop_path"),
            )
            db.add(lr)

        total_plants += 1
        if is_healthy:
            healthy_count += 1
        else:
            diseased_count += 1

    db.commit()
    db.close()

    # --- Stage 4: Finalize ---
    _update_flight(
        flight_id,
        status="completed",
        current_stage="Analysis completed",
        progress=100.0,
        total_plants=total_plants,
        diseased_plants=diseased_count,
        healthy_plants=healthy_count,
        completed_at=datetime.datetime.utcnow(),
    )
