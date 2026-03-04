"""Segmentation metrics (mIoU)."""

from __future__ import annotations

import torch


@torch.no_grad()
def mean_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    preds = preds.argmax(dim=1)

    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls

        valid = targets != ignore_index
        pred_mask = pred_mask & valid
        target_mask = target_mask & valid

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union > 0:
            ious.append((intersection / union).item())

    if not ious:
        return 0.0
    return float(sum(ious) / len(ious))
