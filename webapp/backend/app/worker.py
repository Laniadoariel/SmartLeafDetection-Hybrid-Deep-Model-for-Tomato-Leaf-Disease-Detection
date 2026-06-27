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

# Dedicated leaf-detection model (object detector that locates individual
# leaves). When present, the worker runs a two-stage flow: this model finds
# the leaves and the (unchanged) disease model classifies each leaf crop.
# If absent, the worker falls back to its previous behaviour.
LEAF_WEIGHTS = _find_weights("leaf_detector", [
    "weights/leaf_best.pt",
])
# Confidence threshold for the leaf detector. Set from the evaluation conf
# sweep (training/leaf_detection/reports/metrics_improved.json) via env var.
LEAF_CONF = float(os.getenv("LEAF_CONF", "0.3"))

# --- Leaf tracker backend ------------------------------------------------
# Which multi-object tracker follows each leaf across frames. BoT-SORT adds
# Global Motion Compensation (GMC), which compensates for the moving drone
# camera and keeps the same physical leaf on the same track ID across more
# frames than ByteTrack (the previous baseline) can.
#   LEAF_TRACKER=botsort   -> BoT-SORT + GMC (trackers/botsort_gmc.yaml)
#   LEAF_TRACKER=bytetrack -> ByteTrack baseline (trackers/bytetrack.yaml)
# ByteTrack is kept as a fallback. Default is "botsort": BoT-SORT + GMC proved
# (visually and in the A/B comparison) to track the same leaf across far more
# frames on this moving-camera drone footage. Set LEAF_TRACKER=bytetrack to
# revert to the baseline.
_DEFAULT_TRACKER = "botsort"
_TRACKER_DIR = Path(__file__).resolve().parent / "trackers"
_TRACKER_CONFIGS = {
    "botsort": _TRACKER_DIR / "botsort_gmc.yaml",
    "bytetrack": _TRACKER_DIR / "bytetrack.yaml",
}
LEAF_TRACKER = os.getenv("LEAF_TRACKER", _DEFAULT_TRACKER).strip().lower()
if LEAF_TRACKER not in _TRACKER_CONFIGS:
    print(f"[worker] unknown LEAF_TRACKER={LEAF_TRACKER!r}; "
          f"falling back to {_DEFAULT_TRACKER!r}")
    LEAF_TRACKER = _DEFAULT_TRACKER
LEAF_TRACKER_CFG = str(_TRACKER_CONFIGS[LEAF_TRACKER])

# --- Disease prediction backend -------------------------------------------
# "classifier" = dedicated image classifier (weights/leaf_classifier.pt);
# "yolo"       = legacy YOLO disease detector run on each leaf crop;
# "auto"       = classifier if its weights exist, else yolo.
# The leaf detector (YOLO) is ALWAYS used for localization + tracking; only the
# per-crop disease prediction differs between backends.
CLASSIFIER_WEIGHTS = os.getenv("CLASSIFIER_WEIGHTS")
if CLASSIFIER_WEIGHTS:
    # Allow pointing at any benchmarked checkpoint without copying it.
    _p = Path(CLASSIFIER_WEIGHTS)
    if not _p.is_absolute():
        _p = PROJECT_ROOT / _p
    CLASSIFIER_WEIGHTS = str(_p) if _p.exists() else None
else:
    CLASSIFIER_WEIGHTS = _find_weights("disease_classifier", ["weights/leaf_classifier.pt"])
_BACKEND_PREF = os.getenv("DISEASE_BACKEND", "auto").lower()
DISEASE_BACKEND = (
    "classifier" if (_BACKEND_PREF == "classifier" or (_BACKEND_PREF == "auto" and CLASSIFIER_WEIGHTS))
    else "yolo"
)
_CLF_MODEL = None  # cached LeafDiseaseClassifier


def _get_classifier():
    """Lazily load (once) the dedicated disease classifier, or None on failure."""
    global _CLF_MODEL
    if _CLF_MODEL is None and CLASSIFIER_WEIGHTS:
        try:
            from smart_leaf_detection.leaf_disease_classifier import LeafDiseaseClassifier
            _CLF_MODEL = LeafDiseaseClassifier(CLASSIFIER_WEIGHTS, device="auto")
            print(f"[worker] disease classifier loaded: arch={_CLF_MODEL.arch} "
                  f"from {CLASSIFIER_WEIGHTS} ({len(_CLF_MODEL.classes)} classes)")
        except Exception as exc:  # pragma: no cover - fall back to YOLO
            print(f"[worker] classifier load failed ({exc}); falling back to YOLO backend")
            _CLF_MODEL = None
    return _CLF_MODEL


# Labels treated as "not diseased" across both backends.
_HEALTHY_LABELS = {"healthy", "tomato leaf", "leaf"}


def _classify_crop(crop, yolo_model, class_names) -> tuple[str, float] | None:
    """Predict a disease label for one leaf crop using the active backend.

    Returns ``(label, confidence)`` or ``None`` to keep the caller's default
    (the generic "leaf" label). The classifier backend is preferred when
    available; otherwise the legacy YOLO-on-crop path is used.
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    if DISEASE_BACKEND == "classifier":
        clf = _get_classifier()
        if clf is not None:
            label, conf, _ = clf.classify(crop)
            return label, conf
    # Legacy YOLO-on-crop fallback.
    if yolo_model is not None:
        dres = yolo_model(crop, conf=0.25, verbose=False)
        dboxes = dres[0].boxes
        if dboxes is not None and len(dboxes) > 0:
            bi = int(dboxes.conf.argmax())
            cls_id = int(dboxes.cls[bi].cpu())
            return class_names.get(cls_id, f"class_{cls_id}"), float(dboxes.conf[bi].cpu())
    return None



OUTPUT_BASE = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "webapp" / "backend" / "processing_output")))


def _rel_to_output(path: str | Path) -> str:
    """Path relative to OUTPUT_BASE as a forward-slash (URL/OS-agnostic) string.

    Stored paths are served as URL fragments by the API and must use '/'
    regardless of OS. A naive ``str(path).replace(base + "/", "")`` breaks on
    Windows (backslash separators), so we relativise properly and emit POSIX
    separators.
    """
    p = Path(path).resolve()
    try:
        rel = p.relative_to(OUTPUT_BASE.resolve())
    except ValueError:
        # Not under OUTPUT_BASE — fall back to the path's own name.
        rel = Path(p.name)
    return rel.as_posix()

# --- Best-frame selection tuning (override via env vars) ---
# Minimum fraction of green/vegetation pixels for a frame to be a leaf candidate.
MIN_GREEN_RATIO = float(os.getenv("MIN_GREEN_RATIO", "0.05"))
# Keep at most this many of the best leaf frames for prediction.
MAX_LEAF_FRAMES = int(os.getenv("MAX_LEAF_FRAMES", "30"))
# Weight balance between vegetation content and sharpness in the combined score.
GREEN_WEIGHT = float(os.getenv("GREEN_WEIGHT", "0.6"))
SHARPNESS_WEIGHT = float(os.getenv("SHARPNESS_WEIGHT", "0.4"))
# A frame must have at least this many confident leaf detections to be kept
# when ranking with the trained leaf detector.
MIN_LEAVES = int(os.getenv("MIN_LEAVES", "1"))

# Cached leaf-detector model so frame selection and detection share one load.
_LEAF_MODEL = None


def _get_leaf_model():
    """Lazily load (once) and return the trained leaf detector, or None."""
    global _LEAF_MODEL
    if _LEAF_MODEL is None and LEAF_WEIGHTS:
        from ultralytics import YOLO
        _LEAF_MODEL = YOLO(LEAF_WEIGHTS)
    return _LEAF_MODEL


def _green_ratio(bgr: np.ndarray) -> float:
    """Fraction of pixels that fall in the green/vegetation hue range."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([90, 255, 255]))
    return float(np.count_nonzero(mask)) / mask.size


def _sharpness(bgr: np.ndarray) -> float:
    """Variance of the Laplacian — higher means a sharper, less blurry frame."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _count_leaf_detections(model, bgr: np.ndarray) -> tuple[int, float]:
    """Run the leaf detector on a frame and return (count, mean_confidence)."""
    try:
        results = model(bgr, conf=LEAF_CONF, verbose=False)
    except Exception:
        return 0, 0.0
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return 0, 0.0
    confs = boxes.conf.cpu().numpy()
    return int(len(confs)), float(confs.mean())


def _select_best_leaf_frames(
    candidates: list[tuple[int, str]],
    leaf_model=None,
) -> list[tuple[int, str]]:
    """Rank candidate frames and return the best ones for analysis.

    Preferred strategy (when a trained leaf detector is available): score each
    frame by the number of leaves the detector actually finds (with mean
    confidence as a tiebreaker), drop frames with fewer than ``MIN_LEAVES``
    detections, and keep the top ``MAX_LEAF_FRAMES``. This directly selects the
    frames where the model genuinely detects the most leaves — and because the
    kept frames tend to fall in contiguous runs, they are also the easiest to
    track across time.

    Fallback (no detector): the previous vegetation-content + sharpness
    heuristic.
    """
    if leaf_model is not None:
        scored_det: list[tuple[int, str, int, float]] = []
        for fidx, fpath in candidates:
            img = cv2.imread(fpath)
            if img is None:
                continue
            n, mean_conf = _count_leaf_detections(leaf_model, img)
            scored_det.append((fidx, fpath, n, mean_conf))

        kept = [s for s in scored_det if s[2] >= MIN_LEAVES]
        # If the detector found nothing anywhere, fall back to the heuristic.
        if kept:
            # Rank by leaf count, then by mean confidence.
            kept.sort(key=lambda s: (s[2], s[3]), reverse=True)
            best = kept[:MAX_LEAF_FRAMES]
            best.sort(key=lambda s: s[0])  # restore temporal order for tracking
            return [(fidx, fpath) for fidx, fpath, _n, _c in best]

    # --- Fallback: vegetation content + sharpness heuristic ---
    scored: list[tuple[int, str, float, float]] = []
    for fidx, fpath in candidates:
        img = cv2.imread(fpath)
        if img is None:
            continue
        scored.append((fidx, fpath, _green_ratio(img), _sharpness(img)))

    if not scored:
        return []

    # Normalize sharpness across the set so it is comparable to green ratio (0-1).
    max_sharp = max(s[3] for s in scored) or 1.0

    ranked = []
    for fidx, fpath, green, sharp in scored:
        if green < MIN_GREEN_RATIO:
            continue
        combined = GREEN_WEIGHT * green + SHARPNESS_WEIGHT * (sharp / max_sharp)
        ranked.append((fidx, fpath, combined))

    # Fallback: if nothing passed the green filter, keep the sharpest frames.
    if not ranked:
        ranked = [
            (fidx, fpath, sharp / max_sharp) for fidx, fpath, _g, sharp in scored
        ]

    ranked.sort(key=lambda x: x[2], reverse=True)
    best = ranked[:MAX_LEAF_FRAMES]
    # Restore temporal order for nicer downstream display.
    best.sort(key=lambda x: x[0])
    return [(fidx, fpath) for fidx, fpath, _score in best]


def _select_leaf_window(
    candidates: list[tuple[int, str]],
    leaf_model,
    window: int,
) -> list[tuple[int, str]]:
    """Pick the sharpest, most leaf-dense CONTIGUOUS run of frames.

    For each (temporally ordered) candidate frame we measure both how many
    leaves the trained detector finds AND how in-focus the frame is (variance
    of Laplacian). Blurry frames detect far fewer leaves, so each frame's score
    is its leaf count boosted by its normalized sharpness. A sliding window then
    returns the contiguous block with the highest total score — giving frames
    that are both sharp and leaf-rich, while staying contiguous so the same
    leaves can be tracked across them.
    """
    ordered = sorted(candidates, key=lambda c: c[0])
    measured: list[tuple[int, str, int, float]] = []  # fidx, fpath, count, sharp
    for fidx, fpath in ordered:
        img = cv2.imread(fpath)
        if img is None:
            measured.append((fidx, fpath, 0, 0.0))
            continue
        n, _conf = _count_leaf_detections(leaf_model, img)
        measured.append((fidx, fpath, n, _sharpness(img)))

    if not measured or all(m[2] == 0 for m in measured):
        # Detector found nothing — fall back to the heuristic selection.
        return _select_best_leaf_frames(candidates, leaf_model=None)

    # Focus-weighted score: leaves * (0.5 .. 1.5) depending on relative sharpness.
    max_sharp = max(m[3] for m in measured) or 1.0
    scores = [n * (0.5 + (sharp / max_sharp)) for _f, _p, n, sharp in measured]

    w = min(window, len(measured))
    best_start, best_sum, cur = 0, sum(scores[:w]), sum(scores[:w])
    for start in range(1, len(measured) - w + 1):
        cur += scores[start + w - 1] - scores[start - 1]
        if cur > best_sum:
            best_sum, best_start = cur, start

    return [(f, p) for f, p, _n, _s in measured[best_start:best_start + w]]


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
    # Sampling stride. Denser sampling keeps the same leaf visible across
    # consecutive sampled frames so the tracker can follow it. Configurable via
    # FRAME_STRIDE_SEC (default 1s; lower = better tracking, more compute).
    stride_sec = float(os.getenv("FRAME_STRIDE_SEC", "1.0"))
    step = max(1, int(fps * stride_sec))

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

    # --- Stage 1b: Select the best leaf frames ---
    _update_flight(flight_id, current_stage="Selecting best leaf frames", progress=12.0)
    all_extracted = len(extracted_frames)
    # Use the trained leaf detector to focus on the most leaf-dense CONTIGUOUS
    # segment (so the same leaves can be tracked across frames); falls back to a
    # heuristic if the detector is unavailable.
    leaf_model = _get_leaf_model()
    if leaf_model is not None:
        best_frames = _select_leaf_window(extracted_frames, leaf_model, MAX_LEAF_FRAMES)
    else:
        best_frames = _select_best_leaf_frames(extracted_frames, leaf_model=None)
    if best_frames:
        extracted_frames = best_frames

    total_extracted = len(extracted_frames)
    select_mode = "leaf-detector window" if leaf_model is not None else "green+sharpness"
    _update_flight(
        flight_id,
        total_frames=total_extracted,
        current_stage=f"Selected {total_extracted} best leaf frames of "
                      f"{all_extracted} (by {select_mode})",
        progress=15.0,
    )

    # Frame records are created lazily INSIDE the detection loop below, and
    # ONLY for frames where at least one leaf is actually detected. Frames with
    # no detected leaves are skipped entirely (not saved, not shown).

    # --- Stage 2: Load the disease-prediction backend ---
    _update_flight(flight_id, current_stage=f"Loading disease backend ({DISEASE_BACKEND})", progress=20.0)

    # The YOLO disease model is needed for the legacy "yolo" backend AND as a
    # fallback when the classifier backend is selected but its weights are not
    # present yet (so we never silently label every leaf "healthy").
    model = None
    class_names = {}
    if DISEASE_WEIGHTS and (DISEASE_BACKEND == "yolo" or not CLASSIFIER_WEIGHTS):
        from ultralytics import YOLO
        model = YOLO(DISEASE_WEIGHTS)
        class_names = model.names  # {0: 'Bacterial Spot', ...}

    # Fresh leaf-model instance for TRACKING, so the tracker's state (leaf IDs)
    # starts clean for each video instead of leaking across uploads.
    track_model = None
    if LEAF_WEIGHTS:
        from ultralytics import YOLO
        track_model = YOLO(LEAF_WEIGHTS)
        _update_flight(
            flight_id,
            current_stage=f"Tracking leaves with {Path(LEAF_WEIGHTS).name} "
                          f"@ conf={LEAF_CONF} (tracker={LEAF_TRACKER})",
        )

    _update_flight(flight_id,
                   current_stage=f"Running leaf detection + disease prediction "
                                 f"(backend={DISEASE_BACKEND})", progress=25.0)

    # Per-leaf detections, keyed by stable tracking ID (leaf-centric).
    all_plant_detections: dict[int, list] = defaultdict(list)  # leaf_id -> [obs...]
    untracked_counter = 0
    frames_with_leaves = 0  # how many kept frames actually contained leaves

    for i, (fidx, fpath) in enumerate(extracted_frames):
        frame = cv2.imread(fpath)
        if frame is None:
            continue

        h, w = frame.shape[:2]
        detections = []  # each: {"bbox", "label", "conf", "leaf_id"}

        if track_model is not None:
            # --- Leaf-centric tracking: follow each leaf across frames with a
            #     stable leaf ID; disease model classifies each leaf crop ---
            tres = track_model.track(frame, conf=LEAF_CONF, persist=True,
                                     tracker=LEAF_TRACKER_CFG, verbose=False)
            lboxes = tres[0].boxes
            if lboxes is not None and len(lboxes) > 0:
                if lboxes.id is not None:
                    ids = lboxes.id.int().cpu().tolist()
                else:
                    ids = [None] * len(lboxes)
                for j in range(len(lboxes)):
                    lx1, ly1, lx2, ly2 = lboxes.xyxy[j].cpu().numpy().astype(int)
                    lx1, ly1 = max(0, int(lx1)), max(0, int(ly1))
                    lx2, ly2 = min(w, int(lx2)), min(h, int(ly2))
                    if lx2 <= lx1 or ly2 <= ly1:
                        continue
                    tid = ids[j]
                    if tid is None:
                        untracked_counter += 1
                        tid = 1_000_000 + untracked_counter  # synthetic, untracked
                    label = "leaf"
                    conf = float(lboxes.conf[j].cpu())
                    # Predict disease for the leaf crop via the active backend
                    # (dedicated classifier when available, else YOLO-on-crop).
                    crop = frame[ly1:ly2, lx1:lx2]
                    pred = _classify_crop(crop, model, class_names)
                    if pred is not None:
                        label, conf = pred
                    detections.append({
                        "bbox": (lx1, ly1, lx2, ly2),
                        "label": label, "conf": conf, "leaf_id": int(tid),
                    })
        elif model is not None:
            results = model(frame, conf=0.3, verbose=False)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy().astype(int)
                    conf = float(boxes.conf[j].cpu())
                    cls_id = int(boxes.cls[j].cpu())
                    label = class_names.get(cls_id, f"class_{cls_id}")
                    untracked_counter += 1
                    detections.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "label": label, "conf": conf, "leaf_id": untracked_counter,
                    })
        else:
            # Demo fallback: use green segmentation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([20, 20, 20]), np.array([95, 255, 255]))
            kernel = np.ones((10, 10), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > (h * w) * 0.005:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    untracked_counter += 1
                    detections.append({
                        "bbox": (x, y, x + bw, y + bh),
                        "label": "leaf", "conf": 0.5, "leaf_id": untracked_counter,
                    })

        # Drop frames with no detected leaves — they add no value to a
        # leaf-centric view, so we don't save or display them at all.
        if not detections:
            progress = 25.0 + (i / max(total_extracted, 1)) * 50.0
            _update_flight(flight_id, processed_frames=i + 1, progress=progress)
            continue

        frames_with_leaves += 1

        # Draw annotated frame (stable leaf id + disease label)
        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label, conf, lid = d["label"], d["conf"], d["leaf_id"]
            is_healthy_lbl = label.lower() in _HEALTHY_LABELS
            color = (0, 200, 0) if is_healthy_lbl else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"leaf#{lid} {label} {conf:.0%}",
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        ann_fname = f"annotated_{fidx:06d}.jpg"
        ann_path = str(annotated_dir / ann_fname)
        cv2.imwrite(ann_path, annotated)

        # Create the frame record now that we know this frame has leaves.
        db = SessionLocal()
        db.add(FrameRecord(
            flight_id=flight_id, frame_index=fidx,
            original_path=_rel_to_output(fpath),
            annotated_path=_rel_to_output(ann_path),
            plant_count=len({d["leaf_id"] for d in detections}),
            leaf_count=len(detections),
        ))
        db.commit()
        db.close()

        # Group observations by stable LEAF id (leaf-centric, not plant-centric)
        for d in detections:
            lid = d["leaf_id"]
            x1, y1, x2, y2 = d["bbox"]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            crop_path_rel = None
            if crop.size > 0:
                crop_fname = f"leaf{lid}_f{fidx}.jpg"
                crop_path = str(crops_dir / crop_fname)
                cv2.imwrite(crop_path, crop)
                crop_path_rel = _rel_to_output(crop_path)
            all_plant_detections[lid].append({
                "label": d["label"], "conf": d["conf"],
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "frame_idx": fidx, "crop_path": crop_path_rel,
                "leaf_id": lid,
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

    # Keep leaves that were actually tracked across several frames (drops
    # one-frame flukes / untracked detections). Falls back to keeping all if
    # this filter would remove everything.
    min_track_len = int(os.getenv("MIN_TRACK_LEN", "2"))
    leaves = {lid: dl for lid, dl in all_plant_detections.items() if dl}
    tracked = {lid: dl for lid, dl in leaves.items() if len(dl) >= min_track_len}
    if not tracked:
        tracked = leaves

    for pid, det_list in tracked.items():
        if not det_list:
            continue

        # Aggregate over the frames this leaf was seen in. The production
        # decision is a CONFIDENCE-WEIGHTED vote: each observation adds its
        # confidence to its predicted label; the label with the highest summed
        # confidence wins. We also track the plain view-count agreement so the
        # UI can present both honestly and never contradict the weighted logic.
        label_conf: dict[str, float] = defaultdict(float)   # summed confidence per label
        label_votes: dict[str, int] = defaultdict(int)      # view count per label
        total_conf = 0.0
        for d in det_list:
            label_conf[d["label"]] += d["conf"]
            label_votes[d["label"]] += 1
            total_conf += d["conf"]

        best_label = max(label_conf, key=lambda k: label_conf[k])     # weighted winner
        count_winner = max(label_votes, key=lambda k: label_votes[k])  # plain plurality
        views_total = len(det_list)
        views_agreeing = label_votes[best_label]
        weighted_decision = best_label != count_winner  # winner came from weighting, not plurality

        # Headline confidence = mean confidence of the observations that voted
        # for the winning class (i.e. "how sure is the model when it sees this
        # disease"), NOT the mean over all views (which a dissenting view would
        # unfairly drag down).
        winning_confs = [d["conf"] for d in det_list if d["label"] == best_label]
        winning_conf = sum(winning_confs) / len(winning_confs) if winning_confs else (
            total_conf / max(views_total, 1))

        is_healthy = best_label.lower() in _HEALTHY_LABELS
        status = "healthy" if is_healthy else "diseased"

        disease_labels_str = "" if is_healthy else best_label

        evidence = dict(label_conf)
        evidence["frames_seen"] = views_total
        evidence["views_agreeing"] = views_agreeing
        evidence["weighted_decision"] = weighted_decision

        # Each row is ONE tracked leaf (leaf-centric view).
        pr = PlantResult(
            flight_id=flight_id,
            plant_id=pid,
            status=status,
            disease_labels=disease_labels_str,
            confidence=round(winning_conf, 4),
            leaf_count=1,
            diseased_leaf_count=0 if is_healthy else 1,
            views_total=views_total,
            views_agreeing=views_agreeing,
            weighted_decision=1 if weighted_decision else 0,
            evidence_json=json.dumps(evidence),
        )
        db.add(pr)
        db.flush()

        # Save leaf-level results (one row per observation / frame)
        for d in det_list:
            lr = LeafResult(
                plant_result_id=pr.id,
                leaf_id=d.get("leaf_id", pid),
                frame_index=int(d.get("frame_idx", 0)),
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

    total_detections = sum(len(dl) for dl in all_plant_detections.values())

    # --- Stage 4: Finalize ---
    _update_flight(
        flight_id,
        status="completed",
        current_stage="Analysis completed",
        progress=100.0,
        total_frames=frames_with_leaves,
        total_video_frames=total_video_frames,
        relevant_frames=frames_with_leaves,
        total_detections=total_detections,
        total_plants=total_plants,
        diseased_plants=diseased_count,
        healthy_plants=healthy_count,
        completed_at=datetime.datetime.utcnow(),
    )
