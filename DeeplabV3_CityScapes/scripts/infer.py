#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cityscapes_labels import decode_target
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with trained DeepLabV3+ model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Path to output mask image")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--encoder-name", type=str, default="resnet50")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=args.num_classes,
        architecture="deeplabv3plus",
        encoder_name=args.encoder_name,
        encoder_weights=None,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh đầu vào: {args.input}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    transform = A.Compose([A.Resize(args.image_size, args.image_size), A.Normalize()])
    image_t = transform(image=image_rgb)["image"]
    image_t = torch.from_numpy(image_t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_t)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_color = decode_target(pred)
    pred_color = cv2.resize(pred_color, (w, h), interpolation=cv2.INTER_NEAREST)
    pred_bgr = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), pred_bgr)
    print(f"Saved prediction to: {output_path}")


if __name__ == "__main__":
    main()
