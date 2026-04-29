"""Leaf tracking using a lightweight ByteTrack-style IOU tracker.

Assigns stable LeafIDs to leaf detections within a single PlantID scope
using IOU-based association with support for both high-confidence and
low-confidence detections (ByteTrack's key feature).

Each LeafTracker instance is bound to one PlantID. Every
TrackedLeafDetection it produces carries that PlantID.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from smart_leaf_detection.models import Detection, TrackedLeafDetection


# ------------------------------------------------------------------ #
# Internal track representation
# ------------------------------------------------------------------ #

@dataclass
class _LeafTrack:
    """Internal state for a single tracked leaf."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    frames_since_seen: int = 0
    is_active: bool = True


# ------------------------------------------------------------------ #
# IOU helpers (same logic as plant_tracker)
# ------------------------------------------------------------------ #

def _iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
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
    tracks: list[_LeafTrack],
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
    """Greedy IOU matching.

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
        return [], list(range(n_tracks)), list(range(n_dets))

    flat_indices = np.argsort(-iou_matrix.ravel())
    for flat_idx in flat_indices:
        t_idx = int(flat_idx // n_dets)
        d_idx = int(flat_idx % n_dets)
        if t_idx in used_tracks or d_idx in used_dets:
            continue
        if iou_matrix[t_idx, d_idx] < threshold:
            break
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


# ------------------------------------------------------------------ #
# LeafTracker
# ------------------------------------------------------------------ #

class LeafTracker:
    """ByteTrack-style IOU tracker for assigning stable LeafIDs within a PlantID.

    One instance per PlantID. Implements the same two-stage association
    strategy as :class:`PlantTracker`:

    1. First associate high-confidence detections with active tracks.
    2. Then associate remaining low-confidence detections with
       unmatched tracks (reduces fragmentation under occlusion).

    Every :class:`TrackedLeafDetection` returned carries the ``plant_id``
    this tracker was initialised with.
    """

    def __init__(self, plant_id: int, **kwargs: Any) -> None:
        self._plant_id = plant_id

        cfg: dict[str, Any] = dict(_DEFAULTS)
        cfg.update(kwargs)

        self._high_conf_thresh: float = float(cfg["high_confidence_threshold"])
        self._match_iou_thresh: float = float(cfg["match_iou_threshold"])
        self._second_match_iou_thresh: float = float(cfg["second_match_iou_threshold"])
        self._max_frames_lost: int = int(cfg["max_frames_lost"])

        self._tracks: list[_LeafTrack] = []
        self._next_id: int = 1

    @property
    def plant_id(self) -> int:
        """The PlantID this tracker is scoped to."""
        return self._plant_id

    # -------------------------------------------------------------- #
    # Public API
    # -------------------------------------------------------------- #

    def update(self, detections: list[Detection]) -> list[TrackedLeafDetection]:
        """Update tracker state and return tracked leaf detections.

        Args:
            detections: Leaf detections within this plant's ROI.

        Returns:
            A list of :class:`TrackedLeafDetection` objects, each with
            a stable ``leaf_id`` and the parent ``plant_id``.
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

        active_tracks = [t for t in self._tracks if t.is_active]

        # ----- First association: high-confidence detections -------- #
        matched_track_ids: set[int] = set()
        matched_det_global: set[int] = set()
        results: list[TrackedLeafDetection] = []

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
                trk.frames_since_seen = 0
                matched_track_ids.add(id(trk))
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
                trk.frames_since_seen = 0
                matched_track_ids.add(id(trk))
                matched_det_global.add(low_indices[d_idx])
                results.append(self._to_tracked(trk))

        # ----- Create new tracks for unmatched detections ----------- #
        for idx, det in enumerate(detections):
            if idx not in matched_det_global:
                trk = _LeafTrack(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                )
                self._next_id += 1
                self._tracks.append(trk)
                results.append(self._to_tracked(trk))

        # ----- Age unmatched tracks --------------------------------- #
        for trk in self._tracks:
            if trk.is_active and id(trk) not in matched_track_ids:
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

    def _to_tracked(self, trk: _LeafTrack) -> TrackedLeafDetection:
        return TrackedLeafDetection(
            leaf_id=trk.track_id,
            plant_id=self._plant_id,
            bbox=trk.bbox,
            confidence=trk.confidence,
        )
