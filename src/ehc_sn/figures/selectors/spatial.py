"""Spatial rate-map preparation helpers and geometry utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from ehc_sn.figures._contracts import AnyWorld, PreparedRateMap
from ehc_sn.figures.utils.axes import _environment_n_locations
from ehc_sn.figures.utils.rasterize import rasterize_locations_additive
from ehc_sn.figures.utils.spatial import aggregate_rate_map

# ── Geometry helpers (from modules/_spatial.py) ──────────────────────────────

OPEN_FIELD_SPATIAL_GEOMETRY = "open_field"
OPEN_FIELD_RATE_SMOOTH_SIGMA = 1.0
DEFAULT_RATE_SMOOTH_SIGMA = 0.0


def world_spatial_geometry(world: AnyWorld) -> str:
    """Return the declared spatial geometry for a world-like object."""
    if isinstance(world, Mapping):
        value = world.get("spatial_geometry", "unknown")
    else:
        value = getattr(world, "spatial_geometry", "unknown")
    return value if isinstance(value, str) and value else "unknown"


def spatial_rate_smooth_sigma(world: AnyWorld) -> float:
    """Return the rate-map smoothing sigma for the world's geometry."""
    if world_spatial_geometry(world) == OPEN_FIELD_SPATIAL_GEOMETRY:
        return OPEN_FIELD_RATE_SMOOTH_SIGMA
    return DEFAULT_RATE_SMOOTH_SIGMA


# ── Prep helpers ─────────────────────────────────────────────────────────────

DEFAULT_RATE_MAP_MIN_BIN_OCCUPANCY = 1.0


def rectify_response(values: NDArray) -> NDArray:
    """Return nonnegative spatial response (rectify negatives to zero)."""
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def compute_location_responses(
    cells_trace: NDArray,
    location_ids: Sequence[int] | NDArray,
    n_locations: int,
) -> tuple[NDArray, NDArray]:
    """Aggregate rectified cell responses into per-location means."""
    return aggregate_rate_map(rectify_response(cells_trace), location_ids, n_locations)


def compute_cell_location_responses(
    world: AnyWorld,
    cells_trace: NDArray,
    location_ids: Sequence[int] | NDArray,
    cell_idx: int,
) -> tuple[NDArray, NDArray]:
    """Return per-location mean responses and counts for one cell."""
    n_locations = _environment_n_locations(world)
    location_response_matrix, location_counts = compute_location_responses(cells_trace, location_ids, n_locations)
    if location_response_matrix.size == 0 or cell_idx < 0 or cell_idx >= location_response_matrix.shape[0]:
        empty = np.zeros((0,), dtype=float)
        return empty, np.zeros((0,), dtype=int)
    return location_response_matrix[cell_idx], location_counts


def _empty_prepared_rate_map(*, smooth_sigma: float, min_bin_occupancy: float) -> PreparedRateMap:
    empty_grid = np.zeros((0, 0), dtype=float)
    empty_values = np.zeros((0,), dtype=float)
    return PreparedRateMap(
        location_responses=empty_values,
        location_counts=empty_values,
        response_mass_grid=empty_grid,
        occupancy_grid=empty_grid,
        smoothed_response_mass_grid=empty_grid,
        smoothed_occupancy_grid=empty_grid,
        rate_map=empty_grid,
        valid_mask=empty_grid.astype(bool),
        extent=(0.0, 0.0, 0.0, 0.0),
        smooth_sigma=float(smooth_sigma),
        min_bin_occupancy=float(min_bin_occupancy),
    )


def prepare_rate_map(
    world: AnyWorld,
    cells_trace: NDArray,
    location_ids: Sequence[int] | NDArray,
    cell_idx: int,
    *,
    grid_res: float | None = None,
    smooth_sigma: float,
    min_bin_occupancy: float = DEFAULT_RATE_MAP_MIN_BIN_OCCUPANCY,
) -> PreparedRateMap:
    """Build a prepared rate map for a selected cell."""
    location_responses, location_counts = compute_cell_location_responses(world, cells_trace, location_ids, cell_idx)
    if location_responses.size == 0:
        return _empty_prepared_rate_map(smooth_sigma=smooth_sigma, min_bin_occupancy=min_bin_occupancy)
    return prepare_rate_map_from_location_responses(
        world,
        location_responses,
        location_counts,
        grid_res=grid_res,
        smooth_sigma=smooth_sigma,
        min_bin_occupancy=min_bin_occupancy,
    )


def prepare_rate_maps(
    world: AnyWorld,
    cells_trace: NDArray,
    location_ids: Sequence[int] | NDArray,
    *,
    cell_indices: Optional[Sequence[int]] = None,
    grid_res: float | None = None,
    smooth_sigma: float,
    min_bin_occupancy: float = DEFAULT_RATE_MAP_MIN_BIN_OCCUPANCY,
) -> tuple[PreparedRateMap, ...]:
    """Build prepared rate maps for one or more cells."""
    cell_array = np.asarray(cells_trace)
    if cell_array.ndim < 2 or cell_array.shape[-1] == 0:
        return ()

    n_locations = _environment_n_locations(world)
    location_response_matrix, location_counts = compute_location_responses(cells_trace, location_ids, n_locations)
    if location_response_matrix.size == 0:
        return ()

    if cell_indices is None:
        indices = list(range(int(location_response_matrix.shape[0])))
    else:
        indices = [int(idx) for idx in cell_indices if 0 <= int(idx) < int(location_response_matrix.shape[0])]

    return tuple(
        prepare_rate_map_from_location_responses(
            world,
            location_response_matrix[idx],
            location_counts,
            grid_res=grid_res,
            smooth_sigma=smooth_sigma,
            min_bin_occupancy=min_bin_occupancy,
        )
        for idx in indices
    )


def prepare_rate_map_from_location_responses(
    world: AnyWorld,
    location_responses: NDArray,
    location_counts: NDArray,
    *,
    grid_res: float | None = None,
    smooth_sigma: float,
    min_bin_occupancy: float = DEFAULT_RATE_MAP_MIN_BIN_OCCUPANCY,
) -> PreparedRateMap:
    """Build a prepared rate map from location-level responses."""
    location_responses = np.asarray(location_responses, dtype=float)
    location_counts = np.asarray(location_counts, dtype=float)
    if location_responses.size == 0 or location_counts.size == 0:
        return _empty_prepared_rate_map(smooth_sigma=smooth_sigma, min_bin_occupancy=min_bin_occupancy)

    occupied_locations = np.isfinite(location_counts) & (location_counts > 0)
    valid_response_locations = np.isfinite(location_responses) & occupied_locations

    response_mass_by_location = np.zeros_like(location_responses, dtype=float)
    response_mass_by_location[valid_response_locations] = (
        location_responses[valid_response_locations] * location_counts[valid_response_locations]
    )

    response_mass_grid, _, extent = rasterize_locations_additive(
        world, response_mass_by_location, grid_res=grid_res, include_mask=occupied_locations,
    )
    occupancy_grid, _, _ = rasterize_locations_additive(
        world, location_counts, grid_res=grid_res, include_mask=occupied_locations,
    )
    if response_mass_grid.size == 0 or occupancy_grid.size == 0:
        return _empty_prepared_rate_map(smooth_sigma=smooth_sigma, min_bin_occupancy=min_bin_occupancy)

    response_mass_grid = np.asarray(response_mass_grid, dtype=float)
    occupancy_grid = np.asarray(occupancy_grid, dtype=float)
    smoothed_response_mass_grid = response_mass_grid.copy()
    smoothed_occupancy_grid = occupancy_grid.copy()
    if smooth_sigma > 0:
        smoothed_response_mass_grid = gaussian_filter(
            smoothed_response_mass_grid, sigma=float(smooth_sigma), mode="constant", cval=0.0,
        )
        smoothed_occupancy_grid = gaussian_filter(
            smoothed_occupancy_grid, sigma=float(smooth_sigma), mode="constant", cval=0.0,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        rate_map = np.where(smoothed_occupancy_grid > 0, smoothed_response_mass_grid / smoothed_occupancy_grid, np.nan)
    valid_mask = occupancy_grid >= float(min_bin_occupancy)
    rate_map = np.where(valid_mask, rate_map, np.nan)

    return PreparedRateMap(
        location_responses=location_responses,
        location_counts=location_counts,
        response_mass_grid=response_mass_grid,
        occupancy_grid=occupancy_grid,
        smoothed_response_mass_grid=smoothed_response_mass_grid,
        smoothed_occupancy_grid=smoothed_occupancy_grid,
        rate_map=rate_map,
        valid_mask=valid_mask,
        extent=extent,
        smooth_sigma=float(smooth_sigma),
        min_bin_occupancy=float(min_bin_occupancy),
    )
