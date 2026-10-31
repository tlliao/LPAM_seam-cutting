from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(np.clip(image, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2GRAY)


def image_support_mask(image: np.ndarray) -> np.ndarray:
    gray = rgb_to_gray(image)
    return ndimage.binary_fill_holes(gray > 0)


def find_border(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    right = np.zeros_like(mask)
    right[:, :-1] = mask[:, 1:]
    left = np.zeros_like(mask)
    left[:, 1:] = mask[:, :-1]
    down = np.zeros_like(mask)
    down[:-1, :] = mask[1:, :]
    up = np.zeros_like(mask)
    up[1:, :] = mask[:-1, :]
    return (mask & ~right) | (mask & ~left) | (mask & ~down) | (mask & ~up)


def broadcast_mask(mask: np.ndarray, channels: int = 3) -> np.ndarray:
    return np.repeat(np.asarray(mask, dtype=np.float32)[..., None], channels, axis=2)


def hist_otsu(edge_values: np.ndarray, interval_rgb: float = 0.06) -> float:
    values = np.asarray(edge_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0

    bins = np.arange(0.0, 1.0 + 1e-12, interval_rgb)
    centers = np.arange(interval_rgb / 2.0, 1.0 - interval_rgb / 2.0 + 1e-12, interval_rgb)
    counts, _ = np.histogram(values, bins=bins)
    centers = centers[: counts.size]
    total = counts.sum()
    if total == 0:
        return 1.0

    probabilities = counts.astype(np.float64) / float(total)
    total_mean = float(np.sum(probabilities * centers))
    best_energy = -np.inf
    best_index = 0
    for idx in range(centers.size):
        weight = float(np.sum(probabilities[: idx + 1]))
        if weight <= 0.0 or weight >= 1.0:
            continue
        class_mean = float(np.sum(probabilities[: idx + 1] * centers[: idx + 1]))
        sigma = (class_mean - total_mean * weight) ** 2 / (weight * (1.0 - weight))
        if sigma > best_energy:
            best_energy = sigma
            best_index = idx
    return float(centers[best_index] + interval_rgb / 2.0)


def calc_sigmoid_diff(image_diff: np.ndarray, overlap_mask: np.ndarray) -> np.ndarray:
    a_rgb = 0.06
    beta = 4.0 / a_rgb
    alpha = hist_otsu(image_diff[np.asarray(overlap_mask, dtype=bool)], a_rgb)
    logits = np.clip(beta * (alpha - image_diff), -80.0, 80.0)
    diff = 1.0 / (1.0 + np.exp(logits))
    return diff * np.asarray(overlap_mask, dtype=np.float32)
