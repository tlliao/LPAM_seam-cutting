from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lpam_py import run_pipeline
from lpam_py.stitching.io import read_rgb, resize_max_width, write_mask, write_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LPAM seam-cutting Python pipeline on one image pair.")
    parser.add_argument("--img1", required=True, help="Target image path, equivalent to MATLAB path1.")
    parser.add_argument("--img2", required=True, help="Reference image path, equivalent to MATLAB path2.")
    parser.add_argument("--output", default="python_version/outputs/run_pair", help="Output directory.")
    parser.add_argument("--flow-backend", default="dis", choices=["dis", "dis_fast", "dis_ultra", "farneback", "none"])
    parser.add_argument("--max-width", type=int, default=None, help="Optional downscale width for quick tests.")
    parser.add_argument("--skip-improvement", action="store_true", help="Run only initial graph-cut seam estimation.")
    parser.add_argument("--ransac-threshold", type=float, default=4.0, help="OpenCV findHomography RANSAC threshold in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image1 = resize_max_width(read_rgb(args.img1), args.max_width)
    image2 = resize_max_width(read_rgb(args.img2), args.max_width)

    start = time.perf_counter()
    result = run_pipeline(
        image1,
        image2,
        flow_backend=args.flow_backend,
        improve_seam=not args.skip_improvement,
        ransac_reproj_threshold=args.ransac_threshold,
    )
    elapsed = time.perf_counter() - start

    write_rgb(output_dir / "warped_img1.png", result.warped_img1)
    write_rgb(output_dir / "warped_img2.png", result.warped_img2)
    write_rgb(output_dir / "composition.png", result.composition)
    write_mask(output_dir / "seam_a_initial.png", result.seam_a_initial)
    write_mask(output_dir / "seam_b_initial.png", result.seam_b_initial)
    write_mask(output_dir / "seam_a.png", result.seam_a)
    write_mask(output_dir / "seam_b.png", result.seam_b)

    metadata = {
        "img1": args.img1,
        "img2": args.img2,
        "elapsed_seconds": elapsed,
        "input_shape_img1": list(image1.shape),
        "input_shape_img2": list(image2.shape),
        "output_shape": list(result.composition.shape),
        "inlier_count": result.inlier_count,
        "initial_graphcut_backend": result.initial_graphcut.backend,
        "initial_graphcut_cut_value": result.initial_graphcut.cut_value,
        "patch_graphcut_count": len(result.patch_graphcuts),
        "patch_graphcut_backends": [item.backend for item in result.patch_graphcuts],
        "flow_backend": args.flow_backend,
        "skip_improvement": args.skip_improvement,
        "homography": result.homography.tolist(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
