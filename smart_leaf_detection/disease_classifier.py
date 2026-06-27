"""Disease classification using a fine-tuned ResNet50.

Loads a ResNet50 with a custom final layer matching the number of
disease classes and runs inference on normalized 224×224 leaf tensors.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torchvision import models

from smart_leaf_detection.device_utils import resolve_torch_device
from smart_leaf_detection.errors import ModelLoadError
from smart_leaf_detection.models import ClassificationResult


class DiseaseClassifier:
    """Classifies normalized leaf ROIs into disease classes using ResNet50.

    The final fully-connected layer of the stock ResNet50 is replaced with
    one whose output dimension equals ``len(class_names)``.  If
    *weights_path* points to an existing file the corresponding state dict
    is loaded; if it is ``None`` or an empty string the randomly-initialised
    model is used as-is (useful for testing / demo).  Any other missing or
    corrupt path raises :class:`ModelLoadError`.
    """

    def __init__(
        self,
        weights_path: str | None,
        class_names: list[str],
        device: str = "auto",
    ) -> None:
        self.class_names = class_names

        # Resolve the compute device for the current machine. "auto" picks
        # CUDA -> MPS (Apple Silicon) -> CPU; an explicit backend that is not
        # available degrades to CPU so the same call works on Mac and Windows.
        self.device = torch.device(resolve_torch_device(device))

        # Build ResNet50 with custom head
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

        # Optionally load custom weights
        if weights_path is not None and weights_path != "":
            if not os.path.isfile(weights_path):
                raise ModelLoadError(weights_path, "Weights file not found")
            try:
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
            except Exception as exc:
                raise ModelLoadError(weights_path, f"Failed to load weights: {exc}") from exc

        model.to(self.device)
        model.eval()
        self._model = model

    def classify(
        self,
        normalized_leaf: torch.Tensor,
        leaf_id: int,
        plant_id: int,
    ) -> ClassificationResult:
        """Run inference on a single normalised leaf tensor.

        Args:
            normalized_leaf: A ``(1, 3, 224, 224)`` float tensor produced by
                :class:`LeafNormalizer`.
            leaf_id: Stable leaf track identifier.
            plant_id: Parent plant track identifier.

        Returns:
            A :class:`ClassificationResult` with the predicted class (argmax)
            and the full probability vector over all class names.
        """
        input_tensor = normalized_leaf.to(self.device)

        with torch.no_grad():
            logits = self._model(input_tensor)  # (1, num_classes)
            probabilities = F.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

        prob_list = probabilities.cpu().tolist()
        probability_vector = {name: prob for name, prob in zip(self.class_names, prob_list)}

        predicted_class = self.class_names[int(probabilities.argmax().item())]

        return ClassificationResult(
            leaf_id=leaf_id,
            plant_id=plant_id,
            predicted_class=predicted_class,
            probability_vector=probability_vector,
        )
