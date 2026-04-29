"""Plant detection using YOLOv11 via the Ultralytics API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from smart_leaf_detection.errors import ModelLoadError
from smart_leaf_detection.models import Detection


class PlantDetector:
    """Detects tomato plants in full video frames using YOLOv11.

    Loads a YOLOv11 model from the given weights file and runs inference
    on each frame, returning a list of :class:`Detection` records with
    ``class_label="plant"``.
    """

    def __init__(
        self,
        weights_path: str = "yolo11_plants.pt",
        confidence_threshold: float = 0.25,
    ) -> None:
        """Initialise the detector by loading the YOLOv11 model.

        Args:
            weights_path: Path to the YOLO weights file.
            confidence_threshold: Minimum confidence for a detection to be
                included in the results.

        Raises:
            ModelLoadError: If the weights file is missing or cannot be loaded.
        """
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold

        # Validate that the weights file exists before attempting to load.
        if not Path(weights_path).exists():
            raise ModelLoadError(
                weights_path,
                "weights file not found",
            )

        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]

            self._model = YOLO(weights_path)
        except Exception as exc:
            raise ModelLoadError(
                weights_path,
                f"failed to load YOLO model: {exc}",
            ) from exc

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run plant detection on a single BGR frame.

        Args:
            frame: A NumPy array of shape ``(H, W, 3)`` in BGR colour order.

        Returns:
            A list of :class:`Detection` objects with ``class_label="plant"``.
            Returns an empty list when no plants are detected.
        """
        results = self._model(frame, conf=self.confidence_threshold, verbose=False)

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append(
                    Detection(
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        confidence=conf,
                        class_label="plant",
                    )
                )

        return detections
