from __future__ import annotations

import cv2
import numpy as np


def _gray_u8(image: np.ndarray) -> np.ndarray:
    image_u8 = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)


def sift_match(image1: np.ndarray, image2: np.ndarray, ratio: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    gray1 = _gray_u8(image1)
    gray2 = _gray_u8(image2)

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=6000)
        norm_type = cv2.NORM_HAMMING

    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    if descriptors1 is None or descriptors2 is None:
        raise RuntimeError("No local descriptors were found in one of the images.")

    matcher = cv2.BFMatcher(norm_type)
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            good_matches.append(first)

    if len(good_matches) < 4:
        raise RuntimeError(f"Not enough matches for homography: {len(good_matches)}")

    pts1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches], dtype=np.float64)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches], dtype=np.float64)
    return _unique_match_pairs(pts1, pts2)


def _unique_match_pairs(pts1: np.ndarray, pts2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pair_array = np.concatenate([pts1, pts2], axis=1)
    _, unique_indices = np.unique(np.round(pair_array, decimals=4), axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return pts1[unique_indices], pts2[unique_indices]
