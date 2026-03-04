"""Dataset and dataloader utilities for Cityscapes."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .cityscapes_labels import encode_target


class CityscapesDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", image_size: int = 512):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        self.image_paths = sorted((self.root / "leftImg8bit" / split).glob("*/*_leftImg8bit.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"Không tìm thấy ảnh cho split={split} tại {self.root}")

        self.mask_paths = [self._image_to_mask_path(p) for p in self.image_paths]
        self.transform = self._build_transforms(split, image_size)

    def _image_to_mask_path(self, image_path: Path) -> Path:
        city = image_path.parent.name
        file_stem = image_path.name.replace("_leftImg8bit.png", "")
        return self.root / "gtFine" / self.split / city / f"{file_stem}_gtFine_labelIds.png"

    @staticmethod
    def _build_transforms(split: str, image_size: int) -> A.Compose:
        if split == "train":
            return A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.Normalize(),
                ]
            )

        return A.Compose([A.Resize(image_size, image_size), A.Normalize()])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Không đọc được mask: {mask_path}")
        mask = encode_target(mask.astype(np.uint8))

        transformed = self.transform(image=image, mask=mask)
        image_t = torch.from_numpy(transformed["image"].transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(transformed["mask"]).long()

        return image_t, mask_t


def create_dataloaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = CityscapesDataset(data_root, split="train", image_size=image_size)
    val_dataset = CityscapesDataset(data_root, split="val", image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
