from __future__ import annotations

import cv2
import numpy as np

from .features import sift_match


def register_texture(
    image1: np.ndarray,
    image2: np.ndarray,
    ransac_reproj_threshold: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    pts1, pts2 = sift_match(image1, image2)
    homography, inlier_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_reproj_threshold)
    if homography is None:
        raise RuntimeError("cv2.findHomography failed.")
    inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
    warped1, warped2 = homography_align(image1, image2, homography)
    return warped1, warped2, homography, inlier_count


def homography_align(image1: np.ndarray, image2: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    corners1 = np.array([[[0.0, 0.0], [w1 - 1.0, 0.0], [w1 - 1.0, h1 - 1.0], [0.0, h1 - 1.0]]], dtype=np.float32)
    corners2 = np.array([[0.0, 0.0], [w2 - 1.0, 0.0], [w2 - 1.0, h2 - 1.0], [0.0, h2 - 1.0]], dtype=np.float32)
    warped_corners1 = cv2.perspectiveTransform(corners1, homography)[0]
    all_corners = np.vstack([warped_corners1, corners2])

    min_xy = np.floor(all_corners.min(axis=0)).astype(np.int64)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(np.int64)
    width = int(max_xy[0] - min_xy[0] + 1)
    height = int(max_xy[1] - min_xy[1] + 1)
    translation = np.array([[1.0, 0.0, -float(min_xy[0])], [0.0, 1.0, -float(min_xy[1])], [0.0, 0.0, 1.0]])

    warp_matrix = translation @ homography
    warped1 = cv2.warpPerspective(
        image1.astype(np.float32),
        warp_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask1 = cv2.warpPerspective(
        np.ones((h1, w1), dtype=np.float32),
        warp_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped1 *= mask1[..., None]

    warped2 = np.zeros((height, width, image2.shape[2]), dtype=np.float32)
    offset_x = int(-min_xy[0])
    offset_y = int(-min_xy[1])
    warped2[offset_y : offset_y + h2, offset_x : offset_x + w2, :] = image2.astype(np.float32)
    return np.clip(warped1, 0.0, 1.0), np.clip(warped2, 0.0, 1.0)
