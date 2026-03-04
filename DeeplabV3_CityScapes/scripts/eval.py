#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_dataloaders
from src.engine import evaluate
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeepLabV3+ on Cityscapes")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--encoder-name", type=str, default="resnet50")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    model = build_model(
        num_classes=args.num_classes,
        architecture="deeplabv3plus",
        encoder_name=args.encoder_name,
        encoder_weights=None,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    val_loss, val_miou = evaluate(
        model,
        val_loader,
        criterion,
        device,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
    )

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation mIoU: {val_miou:.4f}")


if __name__ == "__main__":
    main()
