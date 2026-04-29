"""Pipeline orchestrator for the SmartLeafDetection system.

Wires all components together and manages the end-to-end processing flow:
frame extraction → plant detection → plant tracking → ROI cropping →
leaf detection → leaf tracking → leaf normalization → disease classification →
temporal aggregation → plant status → GPS association → report export.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from smart_leaf_detection.config import PipelineConfig
from smart_leaf_detection.disease_classifier import DiseaseClassifier
from smart_leaf_detection.frame_extractor import FrameExtractor
from smart_leaf_detection.gps_associator import GPSAssociator
from smart_leaf_detection.leaf_detector import LeafDetector
from smart_leaf_detection.leaf_normalizer import LeafNormalizer
from smart_leaf_detection.leaf_tracker import LeafTracker
from smart_leaf_detection.models import DiseaseRecord
from smart_leaf_detection.plant_detector import PlantDetector
from smart_leaf_detection.plant_status_engine import PlantStatusEngine
from smart_leaf_detection.plant_tracker import PlantTracker
from smart_leaf_detection.report_exporter import ReportExporter
from smart_leaf_detection.roi_cropper import ROICropper
from smart_leaf_detection.temporal_aggregator import TemporalAggregator

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the full SmartLeafDetection processing pipeline.

    Initialises every component from a :class:`PipelineConfig` and exposes
    a single :meth:`run` method that drives the end-to-end flow from video
    ingestion to disease-report export.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

        # --- Initialise components from config --- #
        self._plant_detector = PlantDetector(
            weights_path=config.plant_weights_path,
            confidence_threshold=config.plant_confidence_threshold,
        )
        self._plant_tracker = PlantTracker(
            config_path=config.bytetrack_config_path,
        )
        self._roi_cropper = ROICropper(padding=config.roi_padding)
        self._leaf_detector = LeafDetector(
            weights_path=config.leaf_weights_path,
            confidence_threshold=config.leaf_confidence_threshold,
        )
        self._leaf_normalizer = LeafNormalizer()
        self._disease_classifier = DiseaseClassifier(
            weights_path=config.classifier_weights_path,
            class_names=config.class_names,
            device=config.device,
        )
        self._temporal_aggregator = TemporalAggregator(
            window_size=config.aggregation_window,
            confidence_threshold=config.aggregation_confidence_threshold,
            majority_ratio=config.aggregation_majority_ratio,
            dual_threshold_enabled=config.dual_threshold_enabled,
        )
        self._plant_status_engine = PlantStatusEngine(
            top_k=config.top_k_diseases,
        )
        self._report_exporter = ReportExporter(
            output_format=config.output_format,
            severity_enabled=config.severity_enabled,
        )

    def run(self, video_path: str, srt_path: str | None = None) -> list[DiseaseRecord]:
        """Run the full pipeline on a video file.

        Args:
            video_path: Path to the drone video file.
            srt_path: Optional path to the SRT telemetry file for GPS data.

        Returns:
            A list of :class:`DiseaseRecord` — one per diseased plant.
        """
        # --- Frame extraction setup --- #
        frame_extractor = FrameExtractor(
            video_path=video_path,
            extraction_rate=self._config.extraction_rate,
            target_fps=self._config.target_fps,
        )

        # --- GPS setup --- #
        gps_associator = GPSAssociator(srt_path=srt_path)

        # Per-PlantID leaf trackers
        leaf_trackers: dict[int, LeafTracker] = {}

        # Collect all aggregated labels across all frames, keyed by plant_id
        all_aggregated_labels: dict[int, list] = defaultdict(list)

        # Track which frame_index each plant was last seen in (for GPS)
        plant_last_frame: dict[int, int] = {}

        target_fps = self._config.target_processing_fps

        # --- Process frames --- #
        for frame_index, frame in frame_extractor.extract():
            frame_start = time.monotonic()

            # Step 2: Plant detection
            plant_detections = self._plant_detector.detect(frame)

            # Step 3: Plant tracking
            tracked_plants = self._plant_tracker.update(plant_detections, frame)

            # Step 4-8: For each tracked plant, crop ROI and process leaves
            for tracked_plant in tracked_plants:
                plant_id = tracked_plant.track_id
                plant_last_frame[plant_id] = frame_index

                # Step 4: Crop plant ROI
                plant_roi = self._roi_cropper.crop(
                    frame=frame,
                    bbox=tracked_plant.bbox,
                    owner_id=plant_id,
                )

                # Step 5: Leaf detection within plant ROI
                leaf_detections = self._leaf_detector.detect(
                    plant_roi.image, plant_id,
                )

                # Step 6: Leaf tracking (one tracker per PlantID)
                if plant_id not in leaf_trackers:
                    leaf_trackers[plant_id] = LeafTracker(plant_id=plant_id)
                leaf_tracker = leaf_trackers[plant_id]
                tracked_leaves = leaf_tracker.update(leaf_detections)

                # Steps 7-9: For each tracked leaf, normalize, classify, aggregate
                for tracked_leaf in tracked_leaves:
                    # Step 7: Crop leaf ROI and normalize
                    leaf_roi = self._roi_cropper.crop(
                        frame=plant_roi.image,
                        bbox=tracked_leaf.bbox,
                        owner_id=tracked_leaf.leaf_id,
                    )
                    normalized = self._leaf_normalizer.normalize(leaf_roi.image)

                    # Step 8: Disease classification
                    classification = self._disease_classifier.classify(
                        normalized_leaf=normalized,
                        leaf_id=tracked_leaf.leaf_id,
                        plant_id=tracked_leaf.plant_id,
                    )

                    # Step 9: Temporal aggregation
                    aggregated = self._temporal_aggregator.update(classification)
                    all_aggregated_labels[plant_id].append(aggregated)

            # --- FPS monitoring --- #
            elapsed = time.monotonic() - frame_start
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                if current_fps < target_fps:
                    logger.warning(
                        "Processing rate %.1f FPS is below target %.1f FPS "
                        "(frame %d)",
                        current_fps,
                        target_fps,
                        frame_index,
                    )

        # --- Step 10: Finalize remaining leaf buffers --- #
        for plant_id, leaf_tracker in leaf_trackers.items():
            for track in leaf_tracker._tracks:
                finalized = self._temporal_aggregator.finalize(track.track_id)
                if finalized is not None:
                    all_aggregated_labels[plant_id].append(finalized)

        # --- Step 11: Plant status computation --- #
        disease_records: list[DiseaseRecord] = []

        for plant_id, labels in all_aggregated_labels.items():
            if not labels:
                continue

            plant_status = self._plant_status_engine.compute_status(labels)

            # Skip healthy plants — no record needed
            if plant_status.status == "healthy":
                continue

            # Step 12: GPS association
            last_frame = plant_last_frame.get(plant_id, 0)
            gps = gps_associator.get_gps_for_frame(last_frame)

            # Build evidence metrics from top diseases
            evidence_metrics: dict[str, float] = {}
            for disease_label, confidence in plant_status.top_diseases:
                evidence_metrics[disease_label] = confidence
            evidence_metrics["leaf_count"] = float(plant_status.leaf_count)
            evidence_metrics["diseased_leaf_count"] = float(
                plant_status.diseased_leaf_count
            )

            disease_records.append(
                DiseaseRecord(
                    flight_id=video_path,
                    plant_id=plant_id,
                    gps=gps,
                    disease_labels=[d for d, _ in plant_status.top_diseases],
                    evidence_metrics=evidence_metrics,
                    severity=None,
                )
            )

        # --- Step 13: Report export --- #
        self._report_exporter.export(disease_records, self._config.output_path)

        return disease_records
