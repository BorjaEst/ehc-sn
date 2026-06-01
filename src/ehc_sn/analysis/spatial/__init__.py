"""Spatial analysis primitives: rate-map geometry, gridness, spacing.

Public symbols
--------------
SpatialBinGeometry
    Explicit spatial bin geometry for converting between pixel and world coords.
GridnessResult
    Gridness score with per-angle correlations and mask metadata.
SpacingOrientationResult
    Grid spacing and orientation estimate with peak details.
compute_gridness
    Compute gridness from a full uncropped spatial autocorrelogram.
estimate_grid_spacing_orientation
    Estimate grid spacing and orientation from peak analysis.
"""

from __future__ import annotations

from ehc_sn.analysis.spatial.geometry import SpatialBinGeometry
from ehc_sn.analysis.spatial.gridness import (
    GridnessResult,
    SpacingOrientationResult,
    compute_gridness,
    estimate_grid_spacing_orientation,
)

__all__ = [
    "SpatialBinGeometry",
    "GridnessResult",
    "SpacingOrientationResult",
    "compute_gridness",
    "estimate_grid_spacing_orientation",
]
