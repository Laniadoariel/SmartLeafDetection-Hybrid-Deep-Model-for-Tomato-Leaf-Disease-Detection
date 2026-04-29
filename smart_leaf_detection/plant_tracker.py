"""Plant tracking using a lightweight ByteTrack-style IOU tracker.

Assigns stable PlantIDs to plant detections across frames using
IOU-based association with support for both high-confidence and
low-confidence detections (ByteTrack's key feature).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_leaf_detection.models import Detection, TrackedDetection


# ------------------------------------------------------------------ #
# Internal track representation
# ------------------------------------------------------------------ #

@dataclass
class _Track:
    """Internal state for a single tracked object."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_label: str
    frames_since_seen: int = 0
    is_active: bool = True


# ------------------------------------------------------------------ #
# IOU helpers
# ------------------------------------------------------------------ #

def _iou(box_a: tuple[float, float, float, float],
         box_b: tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union between two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _compute_iou_matrix(
    tracks: list[_Track],
    detections: list[Detection],
) -> np.ndarray:
    """Return an (N_tracks, N_detections) IOU cost matrix."""
    n_tracks = len(tracks)
    n_dets = len(detections)
    matrix = np.zeros((n_tracks, n_dets), dtype=np.float64)
    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            matrix[i, j] = _iou(trk.bbox, det.bbox)
    return matrix


def _greedy_match(
    iou_matrix: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy IOU matching (simple but effective for moderate track counts).

    Returns:
        matched: list of (track_idx, det_idx) pairs
        unmatched_tracks: track indices with no match
        unmatched_dets: detection indices with no match
    """
    n_tracks, n_dets = iou_matrix.shape
    matched: list[tuple[int, int]] = []
    used_tracks: set[int] = set()
    used_dets: set[int] = set()

    if n_tracks == 0 or n_dets == 0:
        return (
            [],
            list(range(n_tracks)),
            list(range(n_dets)),
        )

    # Flatten and sort by descending IOU for greedy assignment
    flat_indices = np.argsort(-iou_matrix.ravel())
    for flat_idx in flat_indices:
        t_idx = int(flat_idx // n_dets)
        d_idx = int(flat_idx % n_dets)
        if t_idx in used_tracks or d_idx in used_dets:
            continue
        if iou_matrix[t_idx, d_idx] < threshold:
            break  # remaining are all below threshold
        matched.append((t_idx, d_idx))
        used_tracks.add(t_idx)
        used_dets.add(d_idx)

    unmatched_tracks = [i for i in range(n_tracks) if i not in used_tracks]
    unmatched_dets = [j for j in range(n_dets) if j not in used_dets]
    return matched, unmatched_tracks, unmatched_dets


# ------------------------------------------------------------------ #
# Default tracker configuration
# ------------------------------------------------------------------ #

_DEFAULTS: dict[str, Any] = {
    "high_confidence_threshold": 0.5,
    "match_iou_threshold": 0.3,
    "second_match_iou_threshold": 0.2,
    "max_frames_lost": 30,
}


def _load_config(
    config_path: str | None,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge defaults ← YAML file ← keyword overrides."""
    cfg: dict[str, Any] = dict(_DEFAULTS)

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            suffix = path.suffix.lower()
            if suffix in {".yaml", ".yml"}:
                file_cfg = yaml.safe_load(raw) or {}
            elif suffix == ".json":
                file_cfg = json.loads(raw)
            else:
                file_cfg = {}
            if isinstance(file_cfg, dict):
                cfg.update(file_cfg)

    cfg.update(overrides)
    return cfg


# ------------------------------------------------------------------ #
# PlantTracker
# ------------------------------------------------------------------ #

class PlantTracker:
    """ByteTrack-style IOU tracker for assigning stable PlantIDs.

    Implements the two-stage association strategy from ByteTrack:
    1. First associate high-confidence detections with active tracks.
    2. Then associate remaining low-confidence detections with
       unmatched tracks (reduces fragmentation under occlusion).

    Tracks that are not matched for ``max_frames_lost`` consecutive
    frames are removed.  Tracks that reappear within that window are
    re-associated with their original PlantID.
    """

    def __init__(self, config_path: str | None = None, **kwargs: Any) -> None:
        cfg = _load_config(config_path, kwargs)

        self._high_conf_thresh: float = float(cfg["high_confidence_threshold"])
        self._match_iou_thresh: float = float(cfg["match_iou_threshold"])
        self._second_match_iou_thresh: float = float(cfg["second_match_iou_threshold"])
        self._max_frames_lost: int = int(cfg["max_frames_lost"])

        self._tracks: list[_Track] = []
        self._next_id: int = 1

    # -------------------------------------------------------------- #
    # Public API
    # -------------------------------------------------------------- #

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
    ) -> list[TrackedDetection]:
        """Update tracker state and return tracked detections with PlantIDs.

        Args:
            detections: New detections for the current frame.
            frame: The current BGR video frame (unused by IOU tracker
                   but kept for API compatibility with richer trackers).

        Returns:
            A list of :class:`TrackedDetection` objects, one per
            successfully associated or newly created track.
        """
        if not detections:
            self._age_tracks()
            return []

        # ----- Split detections into high / low confidence ---------- #
        high_dets: list[Detection] = []
        low_dets: list[Detection] = []
        high_indices: list[int] = []
        low_indices: list[int] = []

        for idx, det in enumerate(detections):
            if det.confidence >= self._high_conf_thresh:
                high_dets.append(det)
                high_indices.append(idx)
            else:
                low_dets.append(det)
                low_indices.append(idx)

        # Collect active tracks for matching
        active_tracks = [t for t in self._tracks if t.is_active]

        # ----- First association: high-confidence detections -------- #
        matched_track_idxs: set[int] = set()
        matched_det_global: set[int] = set()
        results: list[TrackedDetection] = []

        if active_tracks and high_dets:
            iou_mat = _compute_iou_matrix(active_tracks, high_dets)
            matches, unmatched_t, _ = _greedy_match(
                iou_mat, self._match_iou_thresh,
            )
            for t_idx, d_idx in matches:
                trk = active_tracks[t_idx]
                det = high_dets[d_idx]
                trk.bbox = det.bbox
                trk.confidence = det.confidence
                trk.class_label = det.class_label
                trk.frames_since_seen = 0
                matched_track_idxs.add(id(trk))
                matched_det_global.add(high_indices[d_idx])
                results.append(self._to_tracked(trk))

            remaining_tracks = [active_tracks[i] for i in unmatched_t]
        else:
            remaining_tracks = list(active_tracks)

        # ----- Second association: low-confidence detections -------- #
        if remaining_tracks and low_dets:
            iou_mat2 = _compute_iou_matrix(remaining_tracks, low_dets)
            matches2, _, _ = _greedy_match(
                iou_mat2, self._second_match_iou_thresh,
            )
            for t_idx, d_idx in matches2:
                trk = remaining_tracks[t_idx]
                det = low_dets[d_idx]
                trk.bbox = det.bbox
                trk.confidence = det.confidence
                trk.class_label = det.class_label
                trk.frames_since_seen = 0
                matched_track_idxs.add(id(trk))
                matched_det_global.add(low_indices[d_idx])
                results.append(self._to_tracked(trk))

        # ----- Create new tracks for unmatched detections ----------- #
        for idx, det in enumerate(detections):
            if idx not in matched_det_global:
                trk = _Track(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_label=det.class_label,
                )
                self._next_id += 1
                self._tracks.append(trk)
                results.append(self._to_tracked(trk))

        # ----- Age unmatched tracks --------------------------------- #
        for trk in self._tracks:
            if trk.is_active and id(trk) not in matched_track_idxs:
                trk.frames_since_seen += 1
                if trk.frames_since_seen > self._max_frames_lost:
                    trk.is_active = False

        return results

    # -------------------------------------------------------------- #
    # Internals
    # -------------------------------------------------------------- #

    def _age_tracks(self) -> None:
        """Increment missed-frame counter for all active tracks."""
        for trk in self._tracks:
            if trk.is_active:
                trk.frames_since_seen += 1
                if trk.frames_since_seen > self._max_frames_lost:
                    trk.is_active = False

    @staticmethod
    def _to_tracked(trk: _Track) -> TrackedDetection:
        return TrackedDetection(
            track_id=trk.track_id,
            bbox=trk.bbox,
            confidence=trk.confidence,
            class_label=trk.class_label,
        )
