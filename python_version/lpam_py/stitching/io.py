from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def write_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255)


def resize_max_width(image: np.ndarray, max_width: int | None) -> np.ndarray:
    if max_width is None or image.shape[1] <= max_width:
        return image
    scale = float(max_width) / float(image.shape[1])
    new_size = (max_width, max(1, int(round(image.shape[0] * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
