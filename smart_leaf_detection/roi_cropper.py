"""ROI cropping with configurable padding and boundary clamping."""

from __future__ import annotations

import numpy as np

from smart_leaf_detection.models import CroppedROI


class ROICropper:
    """Crops regions of interest from frames with configurable padding.

    Padding is applied as a fraction of the bounding box dimensions on each side.
    The padded bounding box is clamped to frame boundaries so the crop never
    exceeds the frame edge.
    """

    def __init__(self, padding: float = 0.1) -> None:
        """
        Args:
            padding: Fractional padding to add around the bbox
                     (e.g., 0.1 = 10% of bbox width/height added on each side).
        """
        self.padding = padding

    def crop(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        owner_id: int,
        frame_shape: tuple[int, int] | None = None,
    ) -> CroppedROI:
        """Crop an ROI from *frame* with padding, clamped to frame boundaries.

        Args:
            frame: Source image as an ``(H, W, 3)`` NumPy array.
            bbox: ``(x1, y1, x2, y2)`` bounding box in pixel coordinates.
            owner_id: PlantID or LeafID that owns this ROI.
            frame_shape: Optional ``(H, W)`` override for clamping bounds.
                         Defaults to ``frame.shape[:2]`` when *None*.

        Returns:
            A :class:`CroppedROI` containing the cropped image, the original
            source bbox, the clamped padded bbox, and the owner id.
        """
        h, w = frame_shape if frame_shape is not None else frame.shape[:2]

        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1

        pad_x = box_w * self.padding
        pad_y = box_h * self.padding

        # Expand by padding and clamp to frame boundaries
        px1 = int(max(0, x1 - pad_x))
        py1 = int(max(0, y1 - pad_y))
        px2 = int(min(w, x2 + pad_x))
        py2 = int(min(h, y2 + pad_y))

        # Guarantee non-degenerate crop (x1 < x2, y1 < y2)
        if px1 >= px2:
            px1 = max(0, px2 - 1)
            if px1 >= px2:
                px2 = min(w, px1 + 1)
        if py1 >= py2:
            py1 = max(0, py2 - 1)
            if py1 >= py2:
                py2 = min(h, py1 + 1)

        padded_bbox = (px1, py1, px2, py2)
        image = frame[py1:py2, px1:px2]

        return CroppedROI(
            image=image,
            source_bbox=bbox,
            padded_bbox=padded_bbox,
            owner_id=owner_id,
        )
