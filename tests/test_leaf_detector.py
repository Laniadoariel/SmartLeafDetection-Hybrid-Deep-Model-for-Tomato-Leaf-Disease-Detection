"""Tests for LeafDetector: model loading, detection, and error handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smart_leaf_detection.errors import ModelLoadError
from smart_leaf_detection.leaf_detector import LeafDetector
from smart_leaf_detection.models import Detection


# ------------------------------------------------------------------
# Model loading errors
# ------------------------------------------------------------------


class TestModelLoading:
    """Verify ModelLoadError is raised for missing or corrupt weights."""

    def test_missing_weights_raises_model_load_error(self) -> None:
        with pytest.raises(ModelLoadError, match="weights file not found"):
            LeafDetector(weights_path="nonexistent_weights.pt")

    def test_missing_weights_stores_path(self) -> None:
        with pytest.raises(ModelLoadError) as exc_info:
            LeafDetector(weights_path="no_such_file.pt")
        assert exc_info.value.weights_path == "no_such_file.pt"

    def test_corrupt_weights_raises_model_load_error(self, tmp_path) -> None:
        bad_file = tmp_path / "corrupt.pt"
        bad_file.write_bytes(b"not a valid model file")
        with pytest.raises(ModelLoadError, match="failed to load YOLO model"):
            LeafDetector(weights_path=str(bad_file))


# ------------------------------------------------------------------
# Detection behaviour (with mocked YOLO model)
# ------------------------------------------------------------------


class TestDetect:
    """Verify detect() returns correct Detection records."""

    def _make_detector(self) -> LeafDetector:
        """Create a LeafDetector with a mocked YOLO model."""
        with patch("smart_leaf_detection.leaf_detector.Path.exists", return_value=True):
            with patch("smart_leaf_detection.leaf_detector.YOLO", create=True):
                # Patch the import inside __init__
                mock_yolo_cls = MagicMock()
                mock_model = MagicMock()
                mock_yolo_cls.return_value = mock_model
                with patch.dict(
                    "sys.modules",
                    {"ultralytics": MagicMock(YOLO=mock_yolo_cls)},
                ):
                    detector = LeafDetector(weights_path="yolo11_leaves.pt")
                    detector._model = mock_model
                    return detector

    def test_returns_empty_list_when_no_detections(self) -> None:
        detector = self._make_detector()
        mock_result = MagicMock()
        mock_result.boxes = None
        detector._model.return_value = [mock_result]

        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detector.detect(roi, plant_id=1)

        assert detections == []

    def test_returns_detections_with_leaf_label(self) -> None:
        detector = self._make_detector()

        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock(tolist=MagicMock(return_value=[10.0, 20.0, 50.0, 60.0]))]
        mock_box.conf = [MagicMock(__float__=lambda self: 0.85)]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        detector._model.return_value = [mock_result]

        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detector.detect(roi, plant_id=42)

        assert len(detections) == 1
        assert isinstance(detections[0], Detection)
        assert detections[0].class_label == "leaf"
        assert detections[0].bbox == (10.0, 20.0, 50.0, 60.0)
        assert detections[0].confidence == 0.85

    def test_multiple_detections(self) -> None:
        detector = self._make_detector()

        boxes = []
        for coords, conf_val in [
            ([5.0, 10.0, 30.0, 40.0], 0.9),
            ([50.0, 55.0, 80.0, 90.0], 0.7),
        ]:
            mock_box = MagicMock()
            mock_box.xyxy = [MagicMock(tolist=MagicMock(return_value=coords))]
            mock_box.conf = [MagicMock(__float__=lambda self, v=conf_val: v)]
            boxes.append(mock_box)

        mock_result = MagicMock()
        mock_result.boxes = boxes
        detector._model.return_value = [mock_result]

        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detector.detect(roi, plant_id=7)

        assert len(detections) == 2
        assert all(d.class_label == "leaf" for d in detections)

    def test_confidence_threshold_passed_to_model(self) -> None:
        detector = self._make_detector()
        detector.confidence_threshold = 0.5

        mock_result = MagicMock()
        mock_result.boxes = None
        detector._model.return_value = [mock_result]

        roi = np.zeros((50, 50, 3), dtype=np.uint8)
        detector.detect(roi, plant_id=1)

        detector._model.assert_called_once_with(roi, conf=0.5, verbose=False)

    def test_empty_boxes_list_returns_empty(self) -> None:
        detector = self._make_detector()
        mock_result = MagicMock()
        mock_result.boxes = []
        detector._model.return_value = [mock_result]

        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detector.detect(roi, plant_id=1)

        assert detections == []
