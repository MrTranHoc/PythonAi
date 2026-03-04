"""Model builder for DeepLabV3+ using segmentation-models-pytorch."""

from __future__ import annotations

import segmentation_models_pytorch as smp



def build_model(
    num_classes: int,
    architecture: str = "deeplabv3plus",
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
):
    architecture = architecture.lower()
    if architecture != "deeplabv3plus":
        raise ValueError("Hiện chỉ hỗ trợ architecture='deeplabv3plus'.")

    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=None,
    )
