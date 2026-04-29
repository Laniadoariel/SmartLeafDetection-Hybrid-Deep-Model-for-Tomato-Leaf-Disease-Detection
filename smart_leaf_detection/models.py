"""Core data models for the SmartLeafDetection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Detection:
    """A single object detection result (plant or leaf)."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_label: str


@dataclass
class TrackedDetection:
    """A detection with a stable track ID assigned by the tracker."""

    track_id: int  # Stable PlantID
    bbox: tuple[float, float, float, float]
    confidence: float
    class_label: str


@dataclass
class TrackedLeafDetection:
    """A leaf detection with stable LeafID and parent PlantID."""

    leaf_id: int  # Stable LeafID
    plant_id: int  # Parent PlantID
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class CroppedROI:
    """A cropped region of interest from a frame."""

    image: np.ndarray
    source_bbox: tuple[float, float, float, float]  # Original bbox before padding
    padded_bbox: tuple[int, int, int, int]  # Clamped padded bbox (x1, y1, x2, y2)
    owner_id: int  # PlantID or LeafID


@dataclass
class ClassificationResult:
    """Result of disease classification for a single leaf."""

    leaf_id: int
    plant_id: int
    predicted_class: str
    probability_vector: dict[str, float]  # class_name -> probability


@dataclass
class AggregatedLabel:
    """Temporally aggregated classification label for a leaf."""

    leaf_id: int
    plant_id: int
    label: str
    mean_probability: float
    frame_count: int
    is_confident: bool  # True if dual-threshold criteria met


@dataclass
class PlantStatus:
    """Plant-level health status derived from leaf-level results."""

    plant_id: int
    status: str  # "healthy" or "diseased"
    top_diseases: list[tuple[str, float]]  # [(disease_label, confidence), ...] top-K
    leaf_count: int
    diseased_leaf_count: int


@dataclass
class GPSCoordinate:
    """GPS coordinate from drone SRT telemetry."""

    latitude: float
    longitude: float
    altitude: float | None = None


@dataclass
class DiseaseRecord:
    """A single record in the exported disease report."""

    flight_id: str
    plant_id: int
    gps: GPSCoordinate | None
    disease_labels: list[str]
    evidence_metrics: dict[str, float]
    severity: float | None = None  # Only when severity estimation is enabled
