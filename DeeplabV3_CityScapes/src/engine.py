"""Training and evaluation loops."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from tqdm import tqdm

from .metrics import mean_iou


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    num_classes: int,
    ignore_index: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_miou = 0.0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_miou = mean_iou(logits.detach(), masks, num_classes=num_classes, ignore_index=ignore_index)

        running_loss += loss.item()
        running_miou += batch_miou
        pbar.set_postfix(loss=f"{loss.item():.4f}", miou=f"{batch_miou:.4f}")

    return running_loss / len(dataloader), running_miou / len(dataloader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_miou = 0.0

    pbar = tqdm(dataloader, desc="Val", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        batch_miou = mean_iou(logits, masks, num_classes=num_classes, ignore_index=ignore_index)

        running_loss += loss.item()
        running_miou += batch_miou

    return running_loss / len(dataloader), running_miou / len(dataloader)
