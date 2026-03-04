#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_dataloaders
from src.engine import evaluate, train_one_epoch
from src.model import build_model
from src.train_utils import load_yaml, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on Cityscapes")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--data-root", type=str, required=True, help="Path to Cityscapes root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        image_size=train_cfg["image_size"],
    )

    model = build_model(
        num_classes=cfg["num_classes"],
        architecture=model_cfg["architecture"],
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    optimizer = AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("amp", True) and device.type == "cuda")
    scaler = scaler if scaler.is_enabled() else None

    best_miou = -1.0
    save_dir = Path(train_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, train_miou = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            num_classes=cfg["num_classes"],
            ignore_index=cfg["ignore_index"],
        )
        val_loss, val_miou = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=cfg["num_classes"],
            ignore_index=cfg["ignore_index"],
        )

        print(
            f"Epoch [{epoch}/{train_cfg['epochs']}] "
            f"train_loss={train_loss:.4f} train_mIoU={train_miou:.4f} "
            f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f}"
        )

        latest_path = save_dir / "latest.pt"
        save_checkpoint({"model": model.state_dict(), "epoch": epoch, "miou": val_miou}, latest_path)

        if val_miou > best_miou:
            best_miou = val_miou
            best_path = save_dir / "best.pt"
            save_checkpoint({"model": model.state_dict(), "epoch": epoch, "miou": val_miou}, best_path)


if __name__ == "__main__":
    main()
