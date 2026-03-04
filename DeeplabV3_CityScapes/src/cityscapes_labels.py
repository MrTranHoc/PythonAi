"""Utilities for Cityscapes label conversion and color mapping."""

from __future__ import annotations

import numpy as np

# Map raw Cityscapes label IDs -> train IDs (19 classes), others -> 255
ID_TO_TRAIN_ID = np.full(256, 255, dtype=np.uint8)

_VALID_IDS = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}

for raw_id, train_id in _VALID_IDS.items():
    ID_TO_TRAIN_ID[raw_id] = train_id

TRAIN_ID_TO_COLOR = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)


def encode_target(mask: np.ndarray) -> np.ndarray:
    """Convert Cityscapes raw mask IDs to train IDs."""
    return ID_TO_TRAIN_ID[mask]


def decode_target(mask: np.ndarray) -> np.ndarray:
    """Convert train-ID mask to RGB visualization."""
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(TRAIN_ID_TO_COLOR))
    color_mask[valid] = TRAIN_ID_TO_COLOR[mask[valid]]
    return color_mask
