"""Reusable leaf-disease image classifier (inference).

Loads a checkpoint produced by
``training/disease_classification/train_classifier.py`` and classifies a single
leaf crop into a canonical disease class. This is the one place the webapp (and
the evaluation/comparison scripts) call for classification, so the logic is not
duplicated.

The checkpoint stores everything needed to rebuild the network and preprocess
input (architecture, class list, image size, normalization), so inference is
self-contained and works on macOS / Windows / Linux via
``device_utils.resolve_torch_device``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from smart_leaf_detection.device_utils import resolve_torch_device
from smart_leaf_detection.errors import ModelLoadError


class LeafDiseaseClassifier:
    """Classify a leaf crop into a canonical disease class.

    Args:
        weights_path: path to ``leaf_classifier.pt`` (the training checkpoint).
        device: ``"auto"`` (default) resolves CUDA -> MPS -> CPU.
    """

    def __init__(self, weights_path: str, device: str = "auto") -> None:
        if not Path(weights_path).is_file():
            raise ModelLoadError(weights_path, "classifier weights not found")
        self.device = torch.device(resolve_torch_device(device))
        try:
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        except Exception as exc:  # pragma: no cover
            raise ModelLoadError(weights_path, f"failed to load checkpoint: {exc}") from exc

        # Import here so torchvision is only required when the classifier is used.
        import sys
        proj = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(proj / "training" / "disease_classification"))
        from model_factory import build_model

        self.classes: list[str] = list(ckpt["classes"])
        self.img_size: int = int(ckpt.get("img_size", 224))
        self.mean = np.array(ckpt.get("mean", (0.485, 0.456, 0.406)), dtype=np.float32)
        self.std = np.array(ckpt.get("std", (0.229, 0.224, 0.225)), dtype=np.float32)
        self.arch: str = ckpt.get("arch", "efficientnet_v2_s")

        model, _ = build_model(self.arch, len(self.classes), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(self.device).eval()
        self._model = model

    def _preprocess(self, bgr_crop: np.ndarray) -> torch.Tensor:
        """Resize -> BGR->RGB -> [0,1] -> ImageNet normalize -> (1,3,H,W) tensor."""
        resized = cv2.resize(bgr_crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def classify(self, bgr_crop: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """Classify a BGR leaf crop.

        Returns ``(predicted_class, confidence, probability_vector)``.
        """
        logits = self._model(self._preprocess(bgr_crop))
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        idx = int(probs.argmax())
        prob_vector = {c: float(p) for c, p in zip(self.classes, probs)}
        return self.classes[idx], float(probs[idx]), prob_vector
