"""Python port of the LPAM seam-cutting pipeline."""

from .stitching.pipeline import PipelineResult, run_pipeline

__all__ = ["PipelineResult", "run_pipeline"]
