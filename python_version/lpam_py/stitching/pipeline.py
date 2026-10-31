from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lpam_py.graphcut import GraphCutResult

from .homography import register_texture
from .seam import seam_blending, seam_estimation_in_diff, seam_improving


@dataclass
class PipelineResult:
    warped_img1: np.ndarray
    warped_img2: np.ndarray
    seam_a_initial: np.ndarray
    seam_b_initial: np.ndarray
    seam_a: np.ndarray
    seam_b: np.ndarray
    composition: np.ndarray
    homography: np.ndarray
    inlier_count: int
    initial_graphcut: GraphCutResult
    patch_graphcuts: list[GraphCutResult]


def run_pipeline(
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    flow_backend: str = "dis",
    improve_seam: bool = True,
    ransac_reproj_threshold: float = 4.0,
) -> PipelineResult:
    warped1, warped2, homography, inlier_count = register_texture(
        image1,
        image2,
        ransac_reproj_threshold=ransac_reproj_threshold,
    )
    seam_a_initial, seam_b_initial, initial_graphcut = seam_estimation_in_diff(warped1, warped2)

    if improve_seam:
        seam_a, seam_b, warped1_final, warped2_final, patch_graphcuts = seam_improving(
            warped1,
            warped2,
            seam_a_initial,
            seam_b_initial,
            flow_backend=flow_backend,
        )
    else:
        seam_a, seam_b = seam_a_initial, seam_b_initial
        warped1_final, warped2_final = warped1, warped2
        patch_graphcuts = []

    composition = seam_blending(warped1_final, warped2_final, seam_a, seam_b)
    return PipelineResult(
        warped_img1=warped1_final,
        warped_img2=warped2_final,
        seam_a_initial=seam_a_initial,
        seam_b_initial=seam_b_initial,
        seam_a=seam_a,
        seam_b=seam_b,
        composition=composition,
        homography=homography,
        inlier_count=inlier_count,
        initial_graphcut=initial_graphcut,
        patch_graphcuts=patch_graphcuts,
    )
