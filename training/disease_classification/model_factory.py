"""torchvision classifier factory shared by training, evaluation, and inference.

Keeping the architecture construction in one place means the trainer, the
evaluator, and the webapp inference module all build the network identically —
no duplicated head-surgery logic.

Supported architectures (all ImageNet-pretrained via torchvision):
    efficientnet_v2_s  (default, best accuracy/compute balance)
    efficientnet_b0    (lightest EfficientNet)
    resnet50           (baseline)
    convnext_tiny      (highest capacity here)
    mobilenet_v3_large (fastest on CPU)
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm

SUPPORTED_ARCHS = (
    "efficientnet_v2_s",
    "efficientnet_b0",
    "resnet50",
    "convnext_tiny",
    "mobilenet_v3_large",
)


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> tuple[nn.Module, list[str]]:
    """Build a classifier and return (model, head_module_names).

    ``head_module_names`` lists the top-level module attributes that make up the
    classification head, so the trainer can freeze everything else and only
    train the head during the warm-up phase.
    """
    arch = arch.lower()
    if arch == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, ["fc"]
    if arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m, ["classifier"]
    if arch == "efficientnet_v2_s":
        m = tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m, ["classifier"]
    if arch == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m, ["classifier"]
    if arch == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m, ["classifier"]
    raise ValueError(f"Unsupported arch {arch!r}; choose from {SUPPORTED_ARCHS}")
