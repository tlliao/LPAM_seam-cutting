from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity

from lpam_py.flow import realignment_via_flow
from lpam_py.graphcut import GraphCutResult, graph_cut

from .utils import broadcast_mask, calc_sigmoid_diff, find_border, image_support_mask


def _right_down_edges(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    horizontal_left = np.zeros_like(mask)
    horizontal_left[:, :-1] = mask[:, :-1] & mask[:, 1:]
    horizontal_right = np.zeros_like(mask)
    horizontal_right[:, 1:] = horizontal_left[:, :-1]

    vertical_up = np.zeros_like(mask)
    vertical_up[:-1, :] = mask[:-1, :] & mask[1:, :]
    vertical_down = np.zeros_like(mask)
    vertical_down[1:, :] = vertical_up[:-1, :]
    return horizontal_left, horizontal_right, vertical_up, vertical_down


def _diff_map(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    diff = image1 * broadcast_mask(mask) - image2 * broadcast_mask(mask)
    return np.sqrt(np.sum(diff * diff, axis=2) / 3.0)


def _build_edges(mask: np.ndarray, revind: np.ndarray, weight_map: np.ndarray) -> np.ndarray:
    h_left, h_right, v_up, v_down = _right_down_edges(mask)
    weights_h = (weight_map[h_left] + weight_map[h_right]) / 2.0 + 1e-8
    weights_v = (weight_map[v_up] + weight_map[v_down]) / 2.0 + 1e-8

    edges_h = np.column_stack(
        [
            revind[h_left.ravel()],
            revind[h_right.ravel()],
            weights_h,
            weights_h,
        ]
    )
    edges_v = np.column_stack(
        [
            revind[v_up.ravel()],
            revind[v_down.ravel()],
            weights_v,
            weights_v,
        ]
    )
    if edges_h.size == 0:
        return edges_v.astype(np.float64)
    if edges_v.size == 0:
        return edges_h.astype(np.float64)
    return np.vstack([edges_h, edges_v]).astype(np.float64)


def seam_estimation_in_diff(image1: np.ndarray, image2: np.ndarray) -> tuple[np.ndarray, np.ndarray, GraphCutResult]:
    support_a = image_support_mask(image1)
    support_b = image_support_mask(image2)
    overlap = support_a & support_b
    if not np.any(overlap):
        return support_a, support_b, GraphCutResult(0.0, np.zeros(0, dtype=np.uint8), "none")

    revind = np.full(overlap.size, -1, dtype=np.int64)
    overlap_flat = overlap.ravel()
    revind[overlap_flat] = np.arange(int(overlap_flat.sum()), dtype=np.int64)

    border_b = find_border(support_b)
    border_c = find_border(overlap)
    seed_a = border_b & border_c
    seed_b = (~seed_a) & border_c

    terminal_weights = np.zeros((int(overlap_flat.sum()), 2), dtype=np.float64)
    terminal_weights[revind[seed_a.ravel()], 0] = np.inf
    terminal_weights[revind[seed_b.ravel()], 1] = np.inf

    image_diff = _diff_map(image1, image2, overlap)
    edge_weights = _build_edges(overlap, revind, image_diff)
    result = graph_cut(terminal_weights, edge_weights)

    seam_a = support_a.copy()
    seam_b = support_b.copy()
    overlap_indices = np.flatnonzero(overlap_flat)
    seam_a.ravel()[overlap_indices[result.labels == 1]] = False
    seam_b.ravel()[overlap_indices[result.labels == 0]] = False
    return seam_a, seam_b, result


def patch_seam_estimation(
    image1: np.ndarray,
    image2: np.ndarray,
    seam_img1: np.ndarray,
    seam_img2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, GraphCutResult]:
    h, w = seam_img1.shape
    crop_mask = np.ones((h, w), dtype=bool)
    revind = np.arange(h * w, dtype=np.int64)

    border_a = find_border(seam_img1)
    border_b = find_border(seam_img2)
    border_c = find_border(crop_mask)
    seed_a = border_a & border_c
    seed_b = border_b & border_c

    terminal_weights = np.zeros((h * w, 2), dtype=np.float64)
    terminal_weights[revind[seed_a.ravel()], 0] = np.inf
    terminal_weights[revind[seed_b.ravel()], 1] = np.inf

    image_diff = _diff_map(image1, image2, crop_mask)
    sigmoid_diff = calc_sigmoid_diff(image_diff, crop_mask)
    edge_weights = _build_edges(crop_mask, revind, sigmoid_diff)
    result = graph_cut(terminal_weights, edge_weights)

    seam_a = crop_mask.copy()
    seam_b = crop_mask.copy()
    seam_a.ravel()[revind[result.labels == 1]] = False
    seam_b.ravel()[revind[result.labels == 0]] = False
    return seam_a, seam_b, result


def eval_quality_of_seam(
    image1: np.ndarray,
    image2: np.ndarray,
    overlap: np.ndarray,
    seam_points: np.ndarray,
    patch_size: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    errors = np.zeros(seam_points.shape[0], dtype=np.float64)
    patch_coords = np.zeros((seam_points.shape[0], 4), dtype=np.int64)
    radius = (patch_size - 1) // 2

    for idx, (row, col) in enumerate(seam_points):
        y1 = max(int(row) - radius, 0)
        y2 = min(int(row) + radius + 1, image1.shape[0])
        x1 = max(int(col) - radius, 0)
        x2 = min(int(col) + radius + 1, image1.shape[1])
        patch_coords[idx] = [y1, y2, x1, x2]
        patch_mask = overlap[y1:y2, x1:x2]
        crop1 = image1[y1:y2, x1:x2, :] * broadcast_mask(patch_mask)
        crop2 = image2[y1:y2, x1:x2, :] * broadcast_mask(patch_mask)

        if min(crop1.shape[:2]) < 7:
            errors[idx] = float(np.mean(np.abs(crop1 - crop2)))
            continue

        channel_scores = [
            structural_similarity(crop1[..., ch], crop2[..., ch], data_range=1.0)
            for ch in range(crop1.shape[2])
        ]
        errors[idx] = 1.0 - float(np.mean(channel_scores))
    return errors, patch_coords


def seam_improving(
    image1: np.ndarray,
    image2: np.ndarray,
    seam_a: np.ndarray,
    seam_b: np.ndarray,
    patch_size: int = 21,
    flow_backend: str = "dis",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[GraphCutResult]]:
    support_a = image_support_mask(image1)
    support_b = image_support_mask(image2)
    overlap = support_a & support_b
    seam_band = ndimage.binary_dilation(seam_a, structure=ndimage.generate_binary_structure(2, 1)) & support_a
    seam_pixels = seam_band & seam_b
    seam_points = np.column_stack(np.nonzero(seam_pixels))
    if seam_points.size == 0:
        return seam_a, seam_b, image1, image2, []

    errors, patch_coords = eval_quality_of_seam(image1, image2, overlap, seam_points, patch_size)
    if errors.size == 0 or float(errors.max()) <= 1.5 * float(errors.mean()):
        return seam_a, seam_b, image1, image2, []

    threshold = _safe_otsu(errors)
    artifact_patches = patch_coords[errors >= threshold]
    artifact_mask = np.zeros(overlap.shape, dtype=bool)
    for y1, y2, x1, x2 in artifact_patches:
        artifact_mask[y1:y2, x1:x2] = True
    artifact_mask = ndimage.binary_closing(artifact_mask, structure=np.ones((10, 10), dtype=bool))

    labels, component_count = ndimage.label(artifact_mask)
    improved_a = seam_a.copy()
    improved_b = seam_b.copy()
    improved_image1 = image1.copy()
    improved_image2 = image2.copy()
    graphcut_results: list[GraphCutResult] = []

    for component_id in range(1, component_count + 1):
        rows, cols = np.nonzero(labels == component_id)
        if rows.size == 0:
            continue
        y1, y2 = int(rows.min()), int(rows.max()) + 1
        x1, x2 = int(cols.min()), int(cols.max()) + 1
        if (y2 - y1) < 2 or (x2 - x1) < 2:
            continue

        crop1 = improved_image1[y1:y2, x1:x2, :]
        crop2 = improved_image2[y1:y2, x1:x2, :]
        crop_seam_a = improved_a[y1:y2, x1:x2]
        crop_seam_b = improved_b[y1:y2, x1:x2]

        warped_crop1, warped_crop2 = realignment_via_flow(crop1, crop2, crop_seam_a, backend=flow_backend)
        local_a, local_b, result = patch_seam_estimation(warped_crop1, warped_crop2, crop_seam_a, crop_seam_b)
        improved_a[y1:y2, x1:x2] = local_a
        improved_b[y1:y2, x1:x2] = local_b
        improved_image1[y1:y2, x1:x2, :] = warped_crop1
        improved_image2[y1:y2, x1:x2, :] = warped_crop2
        graphcut_results.append(result)

    return improved_a, improved_b, improved_image1, improved_image2, graphcut_results


def _safe_otsu(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if np.allclose(values, values[0]):
        return float(values[0])
    from skimage.filters import threshold_otsu

    return float(threshold_otsu(values))


def seam_blending(image1: np.ndarray, image2: np.ndarray, seam_a: np.ndarray, seam_b: np.ndarray) -> np.ndarray:
    return np.clip(image1 * broadcast_mask(seam_a) + image2 * broadcast_mask(seam_b), 0.0, 1.0)
