# LPAM seam cutting Python version

This folder contains a Python port of the MATLAB pipeline in the repository root.
The original `Imgs/` directory is reused by default.

## Layout

- `lpam_py/graphcut`: min-cut/max-flow wrapper compatible with the MATLAB `graphCutMex` inputs.
- `lpam_py/flow`: local dense correspondence backends used in the seam-improvement stage.
- `lpam_py/stitching`: feature matching, homography alignment, seam estimation, local realignment, and blending.
- `scripts/run_pair.py`: command-line entry point for one image pair.

## Install

```powershell
python -m pip install -r python_version\requirements.txt
```

`PyMaxflow` is strongly recommended for full-size images. If it is missing, the code falls back to `networkx` so small smoke tests can still run, but that fallback is much slower.

## Run one pair

```powershell
python python_version\scripts\run_pair.py --img1 Imgs\1_l.jpg --img2 Imgs\1_r.jpg --output python_version\outputs\pair_1
```

For a quick environment check without `PyMaxflow`, downscale the images:

```powershell
python python_version\scripts\run_pair.py --img1 Imgs\1_l.jpg --img2 Imgs\1_r.jpg --output python_version\outputs\smoke_1 --max-width 160
```

## Notes on MATLAB parity

The graph-cut energy and mask logic follow the MATLAB files closely. The local re-alignment step intentionally uses OpenCV optical flow instead of the MATLAB SIFT Flow MEX implementation, and OpenCV SIFT/RANSAC is used for the initial homography. For that reason, the final composition is not expected to be pixel-identical to MATLAB.