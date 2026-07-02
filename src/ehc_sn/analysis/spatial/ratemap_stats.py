"""Rate-map statistics for place-like spatial firing analysis.

Implements per-cell scalar metrics derived from occupancy-normalized rate
maps: peak / mean firing rate, Skaggs-style spatial information, sparsity,
peak field location, and field area.

These primitives are used by the HPC place-metrics selector and figure
templates.  They are pure functions consuming ``numpy`` arrays; no traces,
no figure objects, no registry dependencies.

References
----------
Skaggs, McNaughton, Gothard & Markus (1993). "An Information-Theoretic
    Approach to Deciphering the Hippocampal Code".  NeurIPS 5.
Markus et al. (1994). "Spatial information content and reliability of
    hippocampal CA1 neurons".  J. Neurosci. 14(5):2774–2786.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ehc_sn.analysis.spatial.geometry import SpatialBinGeometry

# ── Defaults ─────────────────────────────────────────────────────────────────

_DEFAULT_FIELD_THRESHOLD_FRACTION: float = 0.2
"""Bins above this fraction of peak rate are counted as part of the field."""

_DEFAULT_MIN_SI_FOR_FIELD: float = 0.1
"""Minimum spatial information (bits) for a cell to receive a valid field
center.  Below this threshold ``field_x`` / ``field_y`` / ``field_area`` are
returned as ``NaN``."""


# =============================================================================
# Result dataclass
# =============================================================================


@dataclass(frozen=True)
class RateMapStats:
    """Per-cell scalar metrics derived from an occupancy-normalized rate map.

    ``NaN`` values indicate invalid / silent / all-NaN rate maps rather than
    meaningful zero measurements.
    """

    peak_rate: float
    """Maximum firing rate in any visited bin."""

    mean_rate: float
    """Occupancy-weighted mean firing rate over visited bins."""

    spatial_information: float
    """Skaggs-style spatial information in bits.  ``NaN`` for silent cells."""

    sparsity: float
    """Sparsity (Böhm et al.).  ``NaN`` for silent cells."""

    field_x: float
    """World x-coordinate of the peak-rate bin.  ``NaN`` when the cell is not
    sufficiently spatially informative (see *min_spatial_information_for_field*)."""

    field_y: float
    """World y-coordinate of the peak-rate bin.  ``NaN`` when the cell is not
    sufficiently spatially informative."""

    field_area: float
    """Area (world units²) of bins exceeding ``threshold_fraction × peak_rate``.
    ``NaN`` when the cell is not sufficiently spatially informative."""

    field_mask: NDArray[np.bool_]
    """Boolean mask of bins above field threshold, shape ``(height, width)``.
    ``True`` where ``rate_map >= field_threshold_fraction × peak_rate`` AND
    the bin was visited.  All-``False`` for silent or non-informative cells."""

    coverage: float
    """Fraction of visited bins belonging to the field (``[0, 1]``).
    ``NaN`` for silent or non-informative cells."""


# =============================================================================
# Public API
# =============================================================================


def compute_rate_map_stats(
    rate_map: NDArray[np.floating],
    *,
    occupancy: NDArray[np.floating],
    extent: tuple[float, float, float, float],
    geometry: SpatialBinGeometry,
    field_threshold_fraction: float = _DEFAULT_FIELD_THRESHOLD_FRACTION,
    min_spatial_information_for_field: float = _DEFAULT_MIN_SI_FOR_FIELD,
) -> RateMapStats:
    """Compute place-field metrics for one cell's occupancy-normalized rate map.

    Parameters
    ----------
    rate_map : NDArray
        Occupancy-normalized firing-rate map, shape ``(height, width)``.
        Values are nonnegative firing rates.  ``NaN`` marks unvisited bins.
    occupancy : NDArray
        Per-bin occupancy count, shape ``(height, width)``.  Must be
        nonnegative; zero marks unvisited bins.
    extent : tuple[float, float, float, float]
        ``(xmin, xmax, ymin, ymax)`` world extent of the rate-map grid
        (same convention as ``PreparedRateMap.extent``).
    geometry : SpatialBinGeometry
        Bin geometry for world / pixel coordinate conversion.
    field_threshold_fraction : float
        Fraction of peak rate used as the field-defining threshold
        (default 0.2).
    min_spatial_information_for_field : float
        Minimum spatial information (bits) required for ``field_x`` /
        ``field_y`` / ``field_area`` to be returned as valid numbers rather
        than ``NaN`` (default 0.1).

    Returns
    -------
    RateMapStats
        Per-cell scalar metrics.
    """
    # ── Guard: empty or all-NaN rate map ────────────────────────────────────
    if rate_map.size == 0 or not np.isfinite(rate_map).any():
        empty_mask = np.zeros(rate_map.shape, dtype=bool)
        return RateMapStats(
            peak_rate=float("nan"),
            mean_rate=float("nan"),
            spatial_information=float("nan"),
            sparsity=float("nan"),
            field_x=float("nan"),
            field_y=float("nan"),
            field_area=float("nan"),
            field_mask=empty_mask,
            coverage=float("nan"),
        )

    rate_map = np.asarray(rate_map, dtype=float)
    occupancy = np.asarray(occupancy, dtype=float)

    # ── Visited bins ────────────────────────────────────────────────────────
    visited = np.isfinite(rate_map) & (occupancy > 0)
    if not visited.any():
        empty_mask = np.zeros(rate_map.shape, dtype=bool)
        return RateMapStats(
            peak_rate=float("nan"),
            mean_rate=float("nan"),
            spatial_information=float("nan"),
            sparsity=float("nan"),
            field_x=float("nan"),
            field_y=float("nan"),
            field_area=float("nan"),
            field_mask=empty_mask,
            coverage=float("nan"),
        )

    r = rate_map[visited]
    occ = occupancy[visited]

    # ── Occupancy probabilities ─────────────────────────────────────────────
    p = occ / np.sum(occ)

    # ── Peak rate ───────────────────────────────────────────────────────────
    peak_rate = float(np.nanmax(rate_map))
    if peak_rate <= 0:
        empty_mask = np.zeros(rate_map.shape, dtype=bool)
        return RateMapStats(
            peak_rate=0.0,
            mean_rate=0.0,
            spatial_information=float("nan"),
            sparsity=float("nan"),
            field_x=float("nan"),
            field_y=float("nan"),
            field_area=float("nan"),
            field_mask=empty_mask,
            coverage=float("nan"),
        )

    # ── Mean rate (occupancy-weighted) ──────────────────────────────────────
    mean_rate = float(np.sum(p * r))

    # ── Spatial information (Skaggs) ────────────────────────────────────────
    firing = r > 0
    if mean_rate > 0 and firing.any():
        ratio = r[firing] / mean_rate
        spatial_information = float(np.sum(p[firing] * ratio * np.log2(ratio)))
    else:
        spatial_information = float("nan")

    # ── Sparsity ────────────────────────────────────────────────────────────
    sum_p_r = float(np.sum(p * r))
    sum_p_r2 = float(np.sum(p * r * r))
    if sum_p_r2 > 0:
        sparsity = float(sum_p_r * sum_p_r / sum_p_r2)
    else:
        sparsity = float("nan")

    # ── Field center and area (only if spatially informative) ───────────────
    if (
        not np.isfinite(spatial_information)
        or spatial_information < min_spatial_information_for_field
    ):
        visited_bin_count = int(np.sum(visited))
        coverage_val = float("nan")
        empty_mask = np.zeros(rate_map.shape, dtype=bool)
        return RateMapStats(
            peak_rate=peak_rate,
            mean_rate=mean_rate,
            spatial_information=spatial_information,
            sparsity=sparsity,
            field_x=float("nan"),
            field_y=float("nan"),
            field_area=float("nan"),
            field_mask=empty_mask,
            coverage=coverage_val,
        )

    # Peak location in world coordinates.
    y_idx, x_idx = np.unravel_index(np.nanargmax(rate_map), rate_map.shape)
    n_rows, n_cols = rate_map.shape
    xmin, xmax, ymin, ymax = extent
    xs = np.linspace(xmin, xmax, n_cols)
    ys = np.linspace(ymin, ymax, n_rows)
    field_x = float(xs[x_idx])
    field_y = float(ys[y_idx])

    # Field mask and coverage.
    threshold = field_threshold_fraction * peak_rate
    field_mask_bool = (rate_map >= threshold) & visited
    field_mask = field_mask_bool & visited
    active_bin_count = int(np.sum(field_mask))
    visited_bin_count = int(np.sum(visited))
    bin_area = geometry.bin_size_x * geometry.bin_size_y
    field_area = float(active_bin_count * bin_area)
    coverage_val = (
        float(active_bin_count / visited_bin_count)
        if visited_bin_count > 0
        else float("nan")
    )

    return RateMapStats(
        peak_rate=peak_rate,
        mean_rate=mean_rate,
        spatial_information=spatial_information,
        sparsity=sparsity,
        field_x=field_x,
        field_y=field_y,
        field_area=field_area,
        field_mask=field_mask,
        coverage=coverage_val,
    )
