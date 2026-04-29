"""Leaf ROI normalization for classifier input.

Resizes and normalizes cropped leaf images to produce tensors
ready for ResNet50 inference with ImageNet-pretrained weights.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


class LeafNormalizer:
    """Resizes and normalizes leaf ROIs for classifier input.

    Applies the standard ImageNet preprocessing pipeline:
    resize → BGR-to-RGB → scale to [0,1] → channel-wise mean/std normalization.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.target_size = target_size
        self.mean = mean
        self.std = std

    def normalize(self, leaf_roi: np.ndarray) -> torch.Tensor:
        """Resize, convert BGR→RGB, scale to [0,1], apply ImageNet mean/std normalization.

        Args:
            leaf_roi: BGR uint8 NumPy array of arbitrary spatial size, shape (H, W, 3).

        Returns:
            A ``(1, 3, 224, 224)`` float32 torch.Tensor ready for model inference.
        """
        # Resize to target dimensions (width, height order for cv2)
        resized = cv2.resize(leaf_roi, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)

        # BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Scale uint8 [0, 255] → float32 [0, 1]
        scaled = rgb.astype(np.float32) / 255.0

        # Channel-wise ImageNet normalization: (pixel - mean) / std
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        normalized = (scaled - mean) / std

        # HWC → CHW and add batch dimension → (1, 3, H, W)
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)

        return tensor
