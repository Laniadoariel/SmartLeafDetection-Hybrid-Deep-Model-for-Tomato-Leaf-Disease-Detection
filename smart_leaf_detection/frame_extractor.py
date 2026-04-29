"""Frame extraction from drone video files."""

from __future__ import annotations

import math
from collections.abc import Iterator

import cv2
import numpy as np

from smart_leaf_detection.errors import FrameExtractionError


class FrameExtractor:
    """Reads a video file and yields frames as NumPy arrays.

    Supports two mutually exclusive extraction modes:
    - every-N-frames: extract every Nth frame (extraction_rate)
    - target-FPS: extract frames to approximate a target FPS (target_fps)

    If neither is specified, every frame is extracted.
    """

    def __init__(
        self,
        video_path: str,
        extraction_rate: int | None = None,
        target_fps: float | None = None,
    ) -> None:
        """
        Args:
            video_path: Path to the drone video file.
            extraction_rate: Extract every N-th frame. Mutually exclusive with target_fps.
            target_fps: Target frames per second to extract. Mutually exclusive with extraction_rate.

        Raises:
            ValueError: If both extraction_rate and target_fps are provided.
            FrameExtractionError: If the video file is unreadable or corrupted.
        """
        if extraction_rate is not None and target_fps is not None:
            raise ValueError(
                "extraction_rate and target_fps are mutually exclusive; provide only one."
            )

        self._video_path = video_path
        self._extraction_rate = extraction_rate
        self._target_fps = target_fps

        # Validate file readability eagerly
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise FrameExtractionError(video_path, "Unable to open video file")

        self._native_fps: float = cap.get(cv2.CAP_PROP_FPS)
        self._frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if self._native_fps <= 0:
            raise FrameExtractionError(video_path, "Video reports invalid FPS (<= 0)")

        # Compute the step (every-N-frames) from the chosen mode
        if extraction_rate is not None:
            if extraction_rate < 1:
                raise ValueError("extraction_rate must be >= 1")
            self._step = extraction_rate
        elif target_fps is not None:
            if target_fps <= 0:
                raise ValueError("target_fps must be > 0")
            self._step = max(1, math.floor(self._native_fps / target_fps))
        else:
            self._step = 1

    @property
    def native_fps(self) -> float:
        """The native FPS of the source video."""
        return self._native_fps

    @property
    def frame_count(self) -> int:
        """Total number of frames in the source video."""
        return self._frame_count

    @property
    def step(self) -> int:
        """The computed frame step interval."""
        return self._step

    def extract(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yields ``(frame_index, bgr_frame)`` tuples.

        Streaming — frames are yielded one at a time without writing to disk.

        Raises:
            FrameExtractionError: If the video cannot be opened for reading.
        """
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            cap.release()
            raise FrameExtractionError(
                self._video_path, "Unable to open video file for extraction"
            )

        try:
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_index % self._step == 0:
                    yield (frame_index, frame)
                frame_index += 1
        finally:
            cap.release()
