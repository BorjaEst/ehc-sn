"""Spatial analysis primitives: rate-map geometry, gridness, spacing, place-field metrics.

Public symbols
--------------
SpatialBinGeometry
    Explicit spatial bin geometry for converting between pixel and world coords.
GridnessResult
    Gridness score with per-angle correlations and mask metadata.
SpacingOrientationResult
    Grid spacing and orientation estimate with peak details.
RateMapStats
    Per-cell place-field metrics (peak/mean rate, spatial information, field location).
compute_gridness
    Compute gridness from a full uncropped spatial autocorrelogram.
estimate_grid_spacing_orientation
    Estimate grid spacing and orientation from peak analysis.
compute_rate_map_stats
    Compute place-field metrics for one cell's occupancy-normalised rate map.
"""

from __future__ import annotations

from ehc_sn.analysis.spatial.geometry import SpatialBinGeometry
from ehc_sn.analysis.spatial.gridness import (
    GridnessResult,
    SpacingOrientationResult,
    compute_gridness,
    estimate_grid_spacing_orientation,
)
from ehc_sn.analysis.spatial.ratemap_stats import (
    RateMapStats,
    compute_rate_map_stats,
)

__all__ = [
    "SpatialBinGeometry",
    "GridnessResult",
    "SpacingOrientationResult",
    "RateMapStats",
    "compute_gridness",
    "estimate_grid_spacing_orientation",
    "compute_rate_map_stats",
]
