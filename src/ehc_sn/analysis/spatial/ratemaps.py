"""Population-level rate-map materialization from sufficient statistics.

This module owns the conversion from ``SpatialPopulationStatisticsPayload``
to ``SpatialRateMapPayload`` via ``rate = activation_sum / occupancy``.

The function is a pure NumPy operation: no device transfer, no persistence,
no figure rendering.  Downstream consumers (``ratemap_stats.py``, figures)
apply per-cell metrics and spatial analysis on the output.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from ehc_sn.types import (
    RateMapConfig,
    ScaleMetadata,
    SpatialPopulationStatisticsPayload,
    SpatialRateMapPayload,
)


# =============================================================================
def compute_rate_maps(
    statistics: SpatialPopulationStatisticsPayload,
    config: RateMapConfig = RateMapConfig(),
) -> SpatialRateMapPayload:
    """Produce occupancy-normalized rate maps from sufficient statistics.

    Computes ``rate = activation_sum / occupancy`` for every band, with
    unvisited bins (occupancy < minimum_occupancy) set to ``empty_bin_value``.

    Args:
        statistics: Sufficient statistics from a
            ``SpatialPopulationAccumulator``.
        config: Normalisation and empty-bin policy.

    Returns:
        Rate-map payload with per-band arrays, provenance, and geometry.

    Raises:
        ValueError: If input shapes are incompatible or values invalid.
        ValueError: If smoothing is requested with invalid geometry.
        NotImplementedError: If ``smoothing_mode == "direct"``.
    """
    _validate(statistics, config)

    occupancy = np.asarray(statistics.occupancy, dtype=np.int64)
    visited_mask = occupancy >= config.minimum_occupancy

    rate_maps: dict[str, np.ndarray] = {}

    if config.smoothing_sigma is not None:
        # Occupancy-weighted smoothing path.
        from scipy.ndimage import gaussian_filter

        from ehc_sn.analysis.spatial.geometry import SpatialBinGeometry

        geom = cast("SpatialBinGeometry", statistics.geometry)
        sigma = config.smoothing_sigma

        # Reshape occupancy to spatial grid.
        occ_grid = geom.flat_to_grid(occupancy)  # (E, rows, cols)
        smooth_occ = gaussian_filter(
            occ_grid.astype(np.float64), sigma=sigma, mode="constant", cval=0.0
        )

        for band in statistics.activation_sum:
            act_sum = statistics.activation_sum[band]  # (E, L, U)
            # Reshape activation sum to spatial grid.
            act_grid = geom.flat_to_grid(act_sum)  # (E, rows, cols, U)

            # Smooth with Gaussian filter (spatial axes only).
            smooth_act = np.empty_like(act_grid, dtype=np.float64)
            for u in range(act_grid.shape[-1]):
                smooth_act[..., u] = gaussian_filter(
                    act_grid[..., u].astype(np.float64),
                    sigma=sigma,
                    mode="constant",
                    cval=0.0,
                )

            # Divide: smooth_act / smooth_occ[..., None].
            denominator_grid = smooth_occ[..., None]
            rate_grid = np.full(
                act_grid.shape,
                config.empty_bin_value,
                dtype=np.dtype(config.output_dtype),
            )
            mask = (smooth_occ > 0)[..., None]
            np.divide(
                smooth_act,
                denominator_grid,
                out=rate_grid,
                where=mask,
            )

            # Flatten back to (E, L, U).
            rates = geom.grid_to_flat(rate_grid)

            # Re-apply empty_bin_value for unvisited locations.
            rates[~visited_mask] = config.empty_bin_value

            rate_maps[band] = rates

    else:
        # Unsmoothed path (unchanged).
        denominator = occupancy[..., None]
        for band in statistics.activation_sum:
            act_sum = statistics.activation_sum[band]

            values = np.full(
                act_sum.shape,
                config.empty_bin_value,
                dtype=np.dtype(config.output_dtype),
            )

            np.divide(
                act_sum,
                denominator,
                out=values,
                where=visited_mask[..., None],
            )

            rate_maps[band] = values

    return SpatialRateMapPayload(
        rate_maps=rate_maps,
        occupancy=occupancy,
        visited_mask=visited_mask,
        geometry=statistics.geometry,
        environment_ids=statistics.environment_ids,
        scale_metadata=dict(statistics.scale_metadata),
        minimum_occupancy=config.minimum_occupancy,
        empty_bin_value=config.empty_bin_value,
        smoothing_sigma=config.smoothing_sigma,
        source_valid_step_count=statistics.valid_step_count,
    )


# =============================================================================
def _validate(
    statistics: SpatialPopulationStatisticsPayload,
    config: RateMapConfig,
) -> None:
    """Validate input statistics and config before materialisation."""
    if config.smoothing_sigma is not None:
        from ehc_sn.analysis.spatial.geometry import SpatialBinGeometry

        if config.smoothing_mode == "direct":
            raise NotImplementedError(
                "direct smoothing (smooth(rate)) is not implemented. "
                "Use smoothing_mode='occupancy_weighted' for "
                "smooth(sum) / smooth(occ)."
            )
        if config.smoothing_mode != "occupancy_weighted":
            raise ValueError(
                f"Unknown smoothing_mode: {config.smoothing_mode!r}"
            )
        if not isinstance(statistics.geometry, SpatialBinGeometry):
            raise ValueError(
                "Occupancy-weighted smoothing requires geometry to be a "
                "SpatialBinGeometry instance.  "
                f"Got {type(statistics.geometry).__name__}."
            )

    occupancy = statistics.occupancy
    if occupancy.ndim != 2:
        raise ValueError(
            f"occupancy must be 2-D [E, L], got shape {occupancy.shape}"
        )

    if (occupancy < 0).any():
        raise ValueError("occupancy must be non-negative.")

    n_env, n_loc = occupancy.shape

    if len(statistics.environment_ids) != n_env:
        raise ValueError(
            f"environment_ids length ({len(statistics.environment_ids)}) "
            f"does not match occupancy dim 0 ({n_env})"
        )

    if statistics.activation_sum.keys() != statistics.scale_metadata.keys():
        raise ValueError(
            "activation_sum keys must match scale_metadata keys: "
            f"activation_sum={set(statistics.activation_sum.keys())} != "
            f"scale_metadata={set(statistics.scale_metadata.keys())}"
        )

    for band in statistics.activation_sum:
        act = statistics.activation_sum[band]
        if act.ndim != 3:
            raise ValueError(
                f"activation_sum[{band!r}] must be 3-D [E, L, U], "
                f"got shape {act.shape}"
            )
        if act.shape[:2] != (n_env, n_loc):
            raise ValueError(
                f"activation_sum[{band!r}] spatial shape {act.shape[:2]} "
                f"does not match occupancy shape ({n_env}, {n_loc})"
            )

        sm = statistics.scale_metadata[band]
        if act.shape[2] != sm.feature_dim:
            raise ValueError(
                f"activation_sum[{band!r}] feature dim {act.shape[2]} "
                f"does not match scale_metadata[{band!r}].feature_dim "
                f"({sm.feature_dim})"
            )

        if np.isfinite(act).all():
            continue

        # Only reject non-finite values at locations that have finite
        # occupancy.  A non-finite activation at a location that was never
        # visited (occupancy = 0) is expected — it will be masked anyway.
        visited = occupancy > 0
        visited_non_finite = ~np.isfinite(act[visited])
        if visited_non_finite.any():
            raise ValueError(
                f"activation_sum[{band!r}] contains {int(visited_non_finite.sum())} "
                f"non-finite values at visited locations."
            )


# =============================================================================
__all__ = [
    "compute_rate_maps",
]
