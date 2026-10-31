from __future__ import annotations

import cv2
import numpy as np


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0 + 0.5).astype(np.uint8)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def estimate_dense_flow(source_grid_image: np.ndarray, target_image: np.ndarray, backend: str = "dis") -> np.ndarray:
    """Estimate dense flow from `source_grid_image` coordinates to `target_image`.

    The returned flow has shape `(H, W, 2)` and can be used as
    `target[y + flow_y, x + flow_x]` on the source image grid.
    """

    backend = backend.lower()
    if backend in {"none", "zero", "zeros"}:
        h, w = source_grid_image.shape[:2]
        return np.zeros((h, w, 2), dtype=np.float32)

    prev = _to_gray_u8(source_grid_image)
    nxt = _to_gray_u8(target_image)
    if prev.shape != nxt.shape:
        nxt = cv2.resize(nxt, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)

    if backend.startswith("dis"):
        preset = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
        if "ultra" in backend:
            preset = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
        elif "fast" in backend:
            preset = cv2.DISOPTICAL_FLOW_PRESET_FAST
        estimator = cv2.DISOpticalFlow_create(preset)
        return estimator.calc(prev, nxt, None).astype(np.float32)

    if backend == "farneback":
        return cv2.calcOpticalFlowFarneback(
            prev,
            nxt,
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=21,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        ).astype(np.float32)

    raise ValueError(f"Unsupported flow backend: {backend}")


def _smooth_weight(mask: np.ndarray, width: int, height: int, beta: float) -> np.ndarray:
    orth = np.array([1.0, 0.0], dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if mask.size:
        if np.all(mask[:, -1]):
            orth = np.array([-1.0, 0.0], dtype=np.float32)
        if np.all(mask[0, :]):
            orth = np.array([0.0, 1.0], dtype=np.float32)
        if np.all(mask[-1, :]):
            orth = np.array([0.0, -1.0], dtype=np.float32)

    corners = np.array([[0.0, 0.0], [0.0, height - 1.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0]])
    corner_proj = corners @ orth
    min_proj = float(corner_proj.min())
    max_proj = float(corner_proj.max())
    if max_proj <= min_proj:
        return np.ones((height, width), dtype=np.float32)

    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    proj = (orth[0] * xs + orth[1] * ys - min_proj) / (max_proj - min_proj)
    return (1.0 / (1.0 + np.exp(-beta * (proj - 0.5)))).astype(np.float32)


def realignment_via_flow(
    image1: np.ndarray,
    image2: np.ndarray,
    mask_p: np.ndarray,
    backend: str = "dis",
    smooth_beta: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Locally warp `image1` toward `image2`, mirroring realignmentviaSIFTflow.m."""

    h, w = image2.shape[:2]
    if image1.shape[:2] != image2.shape[:2]:
        image1 = cv2.resize(image1, (w, h), interpolation=cv2.INTER_AREA)

    flow = estimate_dense_flow(image2, image1, backend=backend)
    smooth = _smooth_weight(mask_p, w, h, smooth_beta)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(xs + flow[..., 0] * smooth, 0.0, float(w - 1)).astype(np.float32)
    map_y = np.clip(ys + flow[..., 1] * smooth, 0.0, float(h - 1)).astype(np.float32)

    warped1 = cv2.remap(image1.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return np.clip(warped1, 0.0, 1.0), image2.copy()
