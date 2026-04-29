"""Plant-level status engine merging leaf-level aggregated labels."""

from __future__ import annotations

from collections import defaultdict

from smart_leaf_detection.models import AggregatedLabel, PlantStatus


class PlantStatusEngine:
    """Merges leaf-level AggregatedLabel results into a PlantStatus.

    A plant is "healthy" iff every leaf label is "healthy".
    If any leaf is diseased, the plant is "diseased" and the top-K
    disease labels (ranked by accumulated confidence) are reported.
    """

    def __init__(self, top_k: int = 3):
        """
        Args:
            top_k: Maximum number of disease labels to report.
        """
        self._top_k = top_k

    def compute_status(self, leaf_labels: list[AggregatedLabel]) -> PlantStatus:
        """Merges leaf labels for a single PlantID into a PlantStatus."""
        if not leaf_labels:
            raise ValueError("leaf_labels must not be empty")

        plant_id = leaf_labels[0].plant_id

        diseased_leaves = [ll for ll in leaf_labels if ll.label != "healthy"]
        diseased_leaf_count = len(diseased_leaves)

        if diseased_leaf_count == 0:
            return PlantStatus(
                plant_id=plant_id,
                status="healthy",
                top_diseases=[],
                leaf_count=len(leaf_labels),
                diseased_leaf_count=0,
            )

        # Accumulate confidence per disease class across diseased leaves
        disease_confidence: dict[str, float] = defaultdict(float)
        for ll in diseased_leaves:
            disease_confidence[ll.label] += ll.mean_probability

        # Sort descending by accumulated confidence, take top K
        sorted_diseases = sorted(
            disease_confidence.items(), key=lambda x: x[1], reverse=True
        )
        top_diseases = sorted_diseases[: self._top_k]

        return PlantStatus(
            plant_id=plant_id,
            status="diseased",
            top_diseases=top_diseases,
            leaf_count=len(leaf_labels),
            diseased_leaf_count=diseased_leaf_count,
        )
