"""Temporal aggregation of per-frame classification results per LeafID."""

from __future__ import annotations

from collections import deque

from smart_leaf_detection.models import AggregatedLabel, ClassificationResult


class TemporalAggregator:
    """Buffers per-frame classifications per LeafID and produces stable labels.

    Uses a sliding window of recent classification outputs and applies
    dual-thresholding (mean probability + majority vote) to declare
    confident disease labels.
    """

    def __init__(
        self,
        window_size: int = 30,
        confidence_threshold: float = 0.6,
        majority_ratio: float = 0.6,
        dual_threshold_enabled: bool = True,
    ):
        """
        Args:
            window_size: Sliding buffer size (15-60 frames).
            confidence_threshold: Minimum mean probability for disease declaration.
            majority_ratio: Minimum fraction of frames with majority class.
            dual_threshold_enabled: Whether to apply dual-thresholding.
        """
        if not (15 <= window_size <= 60):
            raise ValueError(
                f"window_size must be between 15 and 60, got {window_size}"
            )
        self._window_size = window_size
        self._confidence_threshold = confidence_threshold
        self._majority_ratio = majority_ratio
        self._dual_threshold_enabled = dual_threshold_enabled

        # leaf_id -> deque of ClassificationResult
        self._buffers: dict[int, deque[ClassificationResult]] = {}
        # leaf_id -> last AggregatedLabel (for finalize)
        self._last_labels: dict[int, AggregatedLabel] = {}

    def update(self, result: ClassificationResult) -> AggregatedLabel:
        """Adds a classification result to the buffer and returns the current aggregated label."""
        leaf_id = result.leaf_id

        if leaf_id not in self._buffers:
            self._buffers[leaf_id] = deque(maxlen=self._window_size)

        self._buffers[leaf_id].append(result)

        label = self._aggregate(leaf_id)
        self._last_labels[leaf_id] = label
        return label

    def finalize(self, leaf_id: int) -> AggregatedLabel | None:
        """Returns the last aggregated result for a lost track. Discards buffer."""
        label = self._last_labels.pop(leaf_id, None)
        self._buffers.pop(leaf_id, None)
        return label

    def _aggregate(self, leaf_id: int) -> AggregatedLabel:
        """Compute the aggregated label from the current buffer for a leaf."""
        buf = self._buffers[leaf_id]

        # Compute element-wise mean of probability vectors
        mean_probs: dict[str, float] = {}
        for result in buf:
            for cls, prob in result.probability_vector.items():
                mean_probs[cls] = mean_probs.get(cls, 0.0) + prob
        n = len(buf)
        for cls in mean_probs:
            mean_probs[cls] /= n

        # Argmax of mean probability vector
        best_class = max(mean_probs, key=lambda c: mean_probs[c])
        best_prob = mean_probs[best_class]

        # Majority vote: count how many buffered frames have predicted_class == best_class
        majority_count = sum(1 for r in buf if r.predicted_class == best_class)
        majority_fraction = majority_count / n

        # Dual-threshold gating
        if self._dual_threshold_enabled:
            is_confident = (
                best_prob >= self._confidence_threshold
                and majority_fraction >= self._majority_ratio
            )
        else:
            is_confident = best_prob >= self._confidence_threshold

        # Use plant_id from the most recent result
        plant_id = buf[-1].plant_id

        return AggregatedLabel(
            leaf_id=leaf_id,
            plant_id=plant_id,
            label=best_class,
            mean_probability=best_prob,
            frame_count=n,
            is_confident=is_confident,
        )
