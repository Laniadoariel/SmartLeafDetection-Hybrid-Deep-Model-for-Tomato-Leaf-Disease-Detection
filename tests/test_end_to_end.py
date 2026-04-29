"""End-to-end integration test for the SmartLeafDetection pipeline.

Tests the full pipeline flow with mocked ML models to verify that all
components wire together correctly without requiring actual model weights
or GPU resources.

Also includes unit-level tests for individual components to ensure
each stage works in isolation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from smart_leaf_detection.config import PipelineConfig
from smart_leaf_detection.disease_classifier import DiseaseClassifier
from smart_leaf_detection.frame_extractor import FrameExtractor
from smart_leaf_detection.gps_associator import GPSAssociator
from smart_leaf_detection.leaf_normalizer import LeafNormalizer
from smart_leaf_detection.leaf_tracker import LeafTracker
from smart_leaf_detection.models import (
    AggregatedLabel,
    ClassificationResult,
    CroppedROI,
    Detection,
    DiseaseRecord,
    GPSCoordinate,
    PlantStatus,
    TrackedDetection,
    TrackedLeafDetection,
)
from smart_leaf_detection.plant_status_engine import PlantStatusEngine
from smart_leaf_detection.plant_tracker import PlantTracker
from smart_leaf_detection.report_exporter import ReportExporter
from smart_leaf_detection.roi_cropper import ROICropper
from smart_leaf_detection.temporal_aggregator import TemporalAggregator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_video(path: str, num_frames: int = 10, width: int = 640, height: int = 480) -> None:
    """Create a minimal .avi video file with solid-color frames."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=(i * 25) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def make_fake_srt(path: str, num_frames: int = 10) -> None:
    """Create a minimal DJI-style SRT file with GPS data."""
    with open(path, "w") as f:
        for i in range(num_frames):
            f.write(f"{i + 1}\n")
            f.write(f"00:00:{i:02d},000 --> 00:00:{i + 1:02d},000\n")
            lat = 32.0 + i * 0.0001
            lon = 34.0 + i * 0.0001
            alt = 50.0 + i
            f.write(f"[latitude: {lat:.6f}] [longitude: {lon:.6f}] [altitude: {alt:.1f}]\n\n")


# ---------------------------------------------------------------------------
# Unit tests for individual components
# ---------------------------------------------------------------------------

class TestFrameExtractor:
    """Tests for FrameExtractor in isolation."""

    def test_extracts_all_frames_by_default(self, tmp_path: Path) -> None:
        video_path = str(tmp_path / "test.avi")
        make_fake_video(video_path, num_frames=5)
        extractor = FrameExtractor(video_path)
        frames = list(extractor.extract())
        assert len(frames) == 5
        for idx, frame in frames:
            assert frame.shape == (480, 640, 3)
            assert frame.dtype == np.uint8

    def test_extraction_rate(self, tmp_path: Path) -> None:
        video_path = str(tmp_path / "test.avi")
        make_fake_video(video_path, num_frames=10)
        extractor = FrameExtractor(video_path, extraction_rate=3)
        frames = list(extractor.extract())
        indices = [idx for idx, _ in frames]
        # Every 3rd frame: 0, 3, 6, 9
        assert indices == [0, 3, 6, 9]

    def test_invalid_video_raises(self, tmp_path: Path) -> None:
        bad_path = str(tmp_path / "nonexistent.avi")
        from smart_leaf_detection.errors import FrameExtractionError
        with pytest.raises(FrameExtractionError):
            FrameExtractor(bad_path)

    def test_mutually_exclusive_params(self, tmp_path: Path) -> None:
        video_path = str(tmp_path / "test.avi")
        make_fake_video(video_path, num_frames=5)
        with pytest.raises(ValueError, match="mutually exclusive"):
            FrameExtractor(video_path, extraction_rate=2, target_fps=10.0)


class TestROICropper:
    """Tests for ROICropper."""

    def test_basic_crop(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        cropper = ROICropper(padding=0.0)
        roi = cropper.crop(frame, bbox=(10.0, 20.0, 50.0, 60.0), owner_id=1)
        assert roi.image.shape == (40, 40, 3)
        assert roi.owner_id == 1
        assert roi.source_bbox == (10.0, 20.0, 50.0, 60.0)

    def test_padding_expands_crop(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        cropper = ROICropper(padding=0.5)
        roi = cropper.crop(frame, bbox=(50.0, 30.0, 90.0, 70.0), owner_id=2)
        # Padding adds 50% of box dims on each side, clamped to frame
        assert roi.image.shape[0] > 40  # height should be > original 40px
        assert roi.image.shape[1] > 40  # width should be > original 40px

    def test_boundary_clamping(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cropper = ROICropper(padding=1.0)
        # Box near edge — padding would go out of bounds
        roi = cropper.crop(frame, bbox=(80.0, 80.0, 95.0, 95.0), owner_id=3)
        px1, py1, px2, py2 = roi.padded_bbox
        assert px1 >= 0 and py1 >= 0
        assert px2 <= 100 and py2 <= 100


class TestLeafNormalizer:
    """Tests for LeafNormalizer."""

    def test_output_shape_and_dtype(self) -> None:
        normalizer = LeafNormalizer()
        leaf_roi = np.random.randint(0, 256, (50, 80, 3), dtype=np.uint8)
        tensor = normalizer.normalize(leaf_roi)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_different_input_sizes(self) -> None:
        normalizer = LeafNormalizer()
        for h, w in [(10, 10), (300, 400), (1, 1), (224, 224)]:
            leaf_roi = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            tensor = normalizer.normalize(leaf_roi)
            assert tensor.shape == (1, 3, 224, 224)


class TestDiseaseClassifier:
    """Tests for DiseaseClassifier with random weights (no file needed)."""

    def test_classify_returns_valid_result(self) -> None:
        class_names = ["healthy", "Early_blight", "Late_blight"]
        classifier = DiseaseClassifier(
            weights_path=None, class_names=class_names, device="cpu",
        )
        tensor = torch.randn(1, 3, 224, 224)
        result = classifier.classify(tensor, leaf_id=1, plant_id=10)
        assert result.leaf_id == 1
        assert result.plant_id == 10
        assert result.predicted_class in class_names
        assert set(result.probability_vector.keys()) == set(class_names)
        # Probabilities should sum to ~1.0 (softmax)
        total = sum(result.probability_vector.values())
        assert abs(total - 1.0) < 1e-5

    def test_missing_weights_raises(self) -> None:
        from smart_leaf_detection.errors import ModelLoadError
        with pytest.raises(ModelLoadError):
            DiseaseClassifier(
                weights_path="/nonexistent/path.pt",
                class_names=["a", "b"],
                device="cpu",
            )


class TestPlantTracker:
    """Tests for PlantTracker."""

    def test_assigns_stable_ids(self) -> None:
        tracker = PlantTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [
            Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_label="plant"),
            Detection(bbox=(300, 300, 400, 400), confidence=0.8, class_label="plant"),
        ]
        tracked = tracker.update(dets, frame)
        assert len(tracked) == 2
        ids = {t.track_id for t in tracked}
        assert len(ids) == 2  # unique IDs

    def test_maintains_ids_across_frames(self) -> None:
        tracker = PlantTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = [Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_label="plant")]
        t1 = tracker.update(det, frame)
        # Same position next frame — should keep same ID
        t2 = tracker.update(det, frame)
        assert t1[0].track_id == t2[0].track_id

    def test_empty_detections(self) -> None:
        tracker = PlantTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracked = tracker.update([], frame)
        assert tracked == []


class TestLeafTracker:
    """Tests for LeafTracker."""

    def test_preserves_plant_id(self) -> None:
        tracker = LeafTracker(plant_id=42)
        dets = [Detection(bbox=(10, 10, 50, 50), confidence=0.9, class_label="leaf")]
        tracked = tracker.update(dets)
        assert len(tracked) == 1
        assert tracked[0].plant_id == 42

    def test_assigns_unique_leaf_ids(self) -> None:
        tracker = LeafTracker(plant_id=1)
        dets = [
            Detection(bbox=(10, 10, 50, 50), confidence=0.9, class_label="leaf"),
            Detection(bbox=(100, 100, 150, 150), confidence=0.8, class_label="leaf"),
        ]
        tracked = tracker.update(dets)
        leaf_ids = {t.leaf_id for t in tracked}
        assert len(leaf_ids) == 2


class TestTemporalAggregator:
    """Tests for TemporalAggregator."""

    def test_single_update(self) -> None:
        agg = TemporalAggregator(window_size=15, dual_threshold_enabled=False)
        result = ClassificationResult(
            leaf_id=1, plant_id=10,
            predicted_class="Early_blight",
            probability_vector={"Early_blight": 0.8, "healthy": 0.2},
        )
        label = agg.update(result)
        assert label.leaf_id == 1
        assert label.plant_id == 10
        assert label.label == "Early_blight"
        assert label.frame_count == 1

    def test_majority_vote_with_dual_threshold(self) -> None:
        agg = TemporalAggregator(
            window_size=15, confidence_threshold=0.5,
            majority_ratio=0.6, dual_threshold_enabled=True,
        )
        # Feed 15 frames all predicting the same disease with high confidence
        for _ in range(15):
            result = ClassificationResult(
                leaf_id=1, plant_id=10,
                predicted_class="Late_blight",
                probability_vector={"Late_blight": 0.9, "healthy": 0.1},
            )
            label = agg.update(result)

        assert label.label == "Late_blight"
        assert label.is_confident is True
        assert label.frame_count == 15

    def test_finalize_returns_last_label(self) -> None:
        agg = TemporalAggregator(window_size=15, dual_threshold_enabled=False)
        result = ClassificationResult(
            leaf_id=5, plant_id=10,
            predicted_class="healthy",
            probability_vector={"healthy": 0.95, "Early_blight": 0.05},
        )
        agg.update(result)
        finalized = agg.finalize(5)
        assert finalized is not None
        assert finalized.leaf_id == 5
        # After finalize, buffer should be gone
        assert agg.finalize(5) is None


class TestPlantStatusEngine:
    """Tests for PlantStatusEngine."""

    def test_all_healthy(self) -> None:
        engine = PlantStatusEngine(top_k=3)
        labels = [
            AggregatedLabel(leaf_id=1, plant_id=10, label="healthy",
                            mean_probability=0.95, frame_count=15, is_confident=True),
            AggregatedLabel(leaf_id=2, plant_id=10, label="healthy",
                            mean_probability=0.90, frame_count=15, is_confident=True),
        ]
        status = engine.compute_status(labels)
        assert status.status == "healthy"
        assert status.top_diseases == []
        assert status.leaf_count == 2
        assert status.diseased_leaf_count == 0

    def test_one_diseased_leaf(self) -> None:
        engine = PlantStatusEngine(top_k=3)
        labels = [
            AggregatedLabel(leaf_id=1, plant_id=10, label="healthy",
                            mean_probability=0.9, frame_count=15, is_confident=True),
            AggregatedLabel(leaf_id=2, plant_id=10, label="Early_blight",
                            mean_probability=0.85, frame_count=15, is_confident=True),
        ]
        status = engine.compute_status(labels)
        assert status.status == "diseased"
        assert status.diseased_leaf_count == 1
        assert len(status.top_diseases) == 1
        assert status.top_diseases[0][0] == "Early_blight"

    def test_top_k_ranking(self) -> None:
        engine = PlantStatusEngine(top_k=2)
        labels = [
            AggregatedLabel(leaf_id=1, plant_id=10, label="Early_blight",
                            mean_probability=0.9, frame_count=15, is_confident=True),
            AggregatedLabel(leaf_id=2, plant_id=10, label="Late_blight",
                            mean_probability=0.7, frame_count=15, is_confident=True),
            AggregatedLabel(leaf_id=3, plant_id=10, label="Leaf_Mold",
                            mean_probability=0.5, frame_count=15, is_confident=True),
        ]
        status = engine.compute_status(labels)
        assert len(status.top_diseases) == 2
        # Highest confidence first
        assert status.top_diseases[0][0] == "Early_blight"
        assert status.top_diseases[1][0] == "Late_blight"


class TestGPSAssociator:
    """Tests for GPSAssociator."""

    def test_no_srt_returns_none(self) -> None:
        gps = GPSAssociator(srt_path=None)
        assert gps.get_gps_for_frame(0) is None

    def test_parse_srt_file(self, tmp_path: Path) -> None:
        srt_path = str(tmp_path / "test.srt")
        make_fake_srt(srt_path, num_frames=5)
        gps = GPSAssociator(srt_path=srt_path)
        coord = gps.get_gps_for_frame(0)
        assert coord is not None
        assert abs(coord.latitude - 32.0) < 0.001
        assert abs(coord.longitude - 34.0) < 0.001
        assert coord.altitude is not None

    def test_missing_frame_returns_none(self, tmp_path: Path) -> None:
        srt_path = str(tmp_path / "test.srt")
        make_fake_srt(srt_path, num_frames=3)
        gps = GPSAssociator(srt_path=srt_path)
        assert gps.get_gps_for_frame(999) is None

    def test_invalid_srt_raises(self, tmp_path: Path) -> None:
        from smart_leaf_detection.errors import SRTParseError
        bad_path = str(tmp_path / "bad.srt")
        Path(bad_path).write_text("not a valid srt file at all")
        with pytest.raises(SRTParseError):
            GPSAssociator(srt_path=bad_path)


class TestReportExporter:
    """Tests for ReportExporter."""

    def _make_records(self) -> list[DiseaseRecord]:
        return [
            DiseaseRecord(
                flight_id="flight_001",
                plant_id=1,
                gps=GPSCoordinate(latitude=32.0, longitude=34.0, altitude=50.0),
                disease_labels=["Early_blight"],
                evidence_metrics={"Early_blight": 0.85, "leaf_count": 3.0},
            ),
            DiseaseRecord(
                flight_id="flight_001",
                plant_id=2,
                gps=None,
                disease_labels=["Late_blight", "Leaf_Mold"],
                evidence_metrics={"Late_blight": 0.7, "Leaf_Mold": 0.3},
            ),
        ]

    def test_json_export(self, tmp_path: Path) -> None:
        exporter = ReportExporter(output_format="json")
        records = self._make_records()
        out_path = str(tmp_path / "report.json")
        exporter.export(records, out_path)

        with open(out_path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["plant_id"] == 1
        assert data[0]["disease_labels"] == ["Early_blight"]
        assert data[0]["gps"]["latitude"] is not None
        assert data[1]["gps"] is None

    def test_csv_export(self, tmp_path: Path) -> None:
        exporter = ReportExporter(output_format="csv")
        records = self._make_records()
        out_path = str(tmp_path / "report.csv")
        exporter.export(records, out_path)

        content = Path(out_path).read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3  # header + 2 records
        assert "flight_id" in lines[0]

    def test_empty_records(self, tmp_path: Path) -> None:
        exporter = ReportExporter(output_format="json")
        out_path = str(tmp_path / "empty.json")
        exporter.export([], out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert data == []

    def test_severity_field_included_when_enabled(self, tmp_path: Path) -> None:
        exporter = ReportExporter(output_format="json", severity_enabled=True)
        records = [
            DiseaseRecord(
                flight_id="f1", plant_id=1, gps=None,
                disease_labels=["x"], evidence_metrics={}, severity=0.75,
            )
        ]
        out_path = str(tmp_path / "sev.json")
        exporter.export(records, out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert "severity" in data[0]
        assert data[0]["severity"] == 0.75


# ---------------------------------------------------------------------------
# Full pipeline integration test (mocked ML models)
# ---------------------------------------------------------------------------

class TestPipelineEndToEnd:
    """End-to-end pipeline test with mocked detector components.

    Mocks PlantDetector and LeafDetector at the pipeline module level so
    their constructors (which load real YOLO weights) never run. This
    verifies the full wiring: video → plant detection → tracking → ROI crop →
    leaf detection → leaf tracking → normalization → classification →
    temporal aggregation → plant status → GPS → report export.
    """

    @pytest.fixture
    def pipeline_env(self, tmp_path: Path):
        """Set up a temporary environment with fake video, SRT, and config."""
        video_path = str(tmp_path / "drone_video.avi")
        srt_path = str(tmp_path / "drone_video.srt")
        output_path = str(tmp_path / "disease_report.json")

        make_fake_video(video_path, num_frames=5, width=640, height=480)
        make_fake_srt(srt_path, num_frames=5)

        return {
            "video_path": video_path,
            "srt_path": srt_path,
            "output_path": output_path,
            "tmp_path": tmp_path,
        }

    @patch("smart_leaf_detection.pipeline.LeafDetector")
    @patch("smart_leaf_detection.pipeline.PlantDetector")
    def test_full_pipeline_with_diseased_plant(
        self, MockPlantDetector, MockLeafDetector, pipeline_env,
    ) -> None:
        """Test full pipeline produces a disease report for a diseased plant."""
        env = pipeline_env

        # Configure mock PlantDetector.detect() to return one plant per frame
        mock_plant_instance = MockPlantDetector.return_value
        mock_plant_instance.detect.return_value = [
            Detection(bbox=(100.0, 100.0, 300.0, 300.0), confidence=0.92, class_label="plant"),
        ]

        # Configure mock LeafDetector.detect() to return two leaves per plant ROI
        mock_leaf_instance = MockLeafDetector.return_value
        mock_leaf_instance.detect.return_value = [
            Detection(bbox=(10.0, 10.0, 80.0, 80.0), confidence=0.88, class_label="leaf"),
            Detection(bbox=(90.0, 20.0, 160.0, 90.0), confidence=0.85, class_label="leaf"),
        ]

        config = PipelineConfig(
            video_path=env["video_path"],
            plant_weights_path="mocked_plants.pt",
            leaf_weights_path="mocked_leaves.pt",
            classifier_weights_path=None,  # random weights
            class_names=["healthy", "Early_blight", "Late_blight"],
            device="cpu",
            aggregation_window=15,
            aggregation_confidence_threshold=0.0,
            aggregation_majority_ratio=0.0,
            dual_threshold_enabled=False,
            output_format="json",
            output_path=env["output_path"],
        )

        from smart_leaf_detection.pipeline import Pipeline
        pipeline = Pipeline(config)
        records = pipeline.run(env["video_path"], srt_path=env["srt_path"])

        # Verify output structure
        assert isinstance(records, list)
        for record in records:
            assert isinstance(record, DiseaseRecord)
            assert record.flight_id == env["video_path"]
            assert isinstance(record.plant_id, int)
            assert isinstance(record.disease_labels, list)
            assert isinstance(record.evidence_metrics, dict)

        # Verify report file was written
        report_path = Path(env["output_path"])
        assert report_path.exists()
        with open(report_path) as f:
            report_data = json.load(f)
        assert isinstance(report_data, list)

    @patch("smart_leaf_detection.pipeline.LeafDetector")
    @patch("smart_leaf_detection.pipeline.PlantDetector")
    def test_pipeline_no_plants_detected(
        self, MockPlantDetector, MockLeafDetector, pipeline_env,
    ) -> None:
        """Pipeline with no plant detections should produce empty report."""
        env = pipeline_env

        # Plant detector returns no detections
        mock_plant_instance = MockPlantDetector.return_value
        mock_plant_instance.detect.return_value = []

        config = PipelineConfig(
            video_path=env["video_path"],
            plant_weights_path="mocked_plants.pt",
            leaf_weights_path="mocked_leaves.pt",
            classifier_weights_path=None,
            class_names=["healthy", "Early_blight"],
            device="cpu",
            aggregation_window=15,
            dual_threshold_enabled=False,
            output_format="json",
            output_path=env["output_path"],
        )

        from smart_leaf_detection.pipeline import Pipeline
        pipeline = Pipeline(config)
        records = pipeline.run(env["video_path"])

        assert records == []
        with open(env["output_path"]) as f:
            assert json.load(f) == []

    @patch("smart_leaf_detection.pipeline.LeafDetector")
    @patch("smart_leaf_detection.pipeline.PlantDetector")
    def test_pipeline_without_gps(
        self, MockPlantDetector, MockLeafDetector, pipeline_env,
    ) -> None:
        """Pipeline should work without SRT file (GPS = None)."""
        env = pipeline_env

        # No plants → empty report, but pipeline should not crash
        mock_plant_instance = MockPlantDetector.return_value
        mock_plant_instance.detect.return_value = []

        config = PipelineConfig(
            video_path=env["video_path"],
            plant_weights_path="mocked_plants.pt",
            leaf_weights_path="mocked_leaves.pt",
            classifier_weights_path=None,
            class_names=["healthy", "Early_blight"],
            device="cpu",
            aggregation_window=15,
            dual_threshold_enabled=False,
            output_format="json",
            output_path=env["output_path"],
        )

        from smart_leaf_detection.pipeline import Pipeline
        pipeline = Pipeline(config)
        records = pipeline.run(env["video_path"], srt_path=None)

        assert records == []
