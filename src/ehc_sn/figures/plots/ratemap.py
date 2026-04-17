"""Prepared spatial rate-map utilities for figure diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from ehc_sn.figures.utils.axes import _environment_n_locations
from ehc_sn.figures.utils.rasterize import rasterize_locations_additive
from ehc_sn.figures.utils.spatial import aggregate_rate_map

DEFAULT_RATE_MAP_MIN_BIN_OCCUPANCY = 1.0


@dataclass(frozen=True)
class PreparedRateMap:
    """Prepared rate-map surfaces for one cell."""

    location_responses: NDArray
    location_counts: NDArray
    response_mass_grid: NDArray
    occupancy_grid: NDArray
    smoothed_response_mass_grid: NDArray
    smoothed_occupancy_grid: NDArray
    rate_map: NDArray
    valid_mask: NDArray
    extent: tuple[float, float, float, float]
    smooth_sigma: float
    min_bin_occupancy: float


def rectify_response(values: NDArray) -> NDArray:
    """Return the default nonnegative spatial response surrogate.

    Args:
        values: Activation array with any shape.

    Returns:
        Array with the same shape after rectifying negative values to zero.
    """
    response = np.asarray(values, dtype=float)
    return np.maximum(response, 0.0)


def compute_location_responses(
    cells_trace: NDArray,
    location_ids: Sequence[int] | NDArray,
    n_locations: int,
) -> tuple[NDArray, NDArray]:
    """Aggregate rectified cell responses into per-location means.

    Args:
        cells_trace: Cell activations with shape ``(T, C)`` or ``(T, B, C)``.
        location_ids: Visited location ids with shape ``(T,)`` or ``(T, B)``.
        n_locations: Total number of locations in the environment.

    Returns:
        Tuple ``(location_responses, location_counts)`` where
        ``location_responses`` has shape ``(C, n_locations)`` and
        ``location_counts`` has shape ``(n_locations,)``.
    """
    return aggregate_rate_map(rectify_response(cells_trace), location_ids, n_locations)


def compute_cell_location_responses(
    world: object,
    cells_trace: NDArray,
    location_ids: Sequence[int] | NDArray,
    cell_idx: int,
) -> tuple[NDArray, NDArray]:
    """Return per-location mean responses and occupancy counts for one cell.

    Args:
        world: Environment world with location coordinates.
        cells_trace: Cell activations with shape ``(T, C)`` or ``(T, B, C)``.
        location_ids: Ordered list of visited location indices.
        cell_idx: Index of the selected cell.

    Returns:
        Tuple ``(location_responses, location_counts)`` for the selected cell.
    """
    n_locations = _environment_n_locations(world)
    location_response_matrix, location_counts = compute_location_responses(cells_trace, location_ids, n_locations)
    if location_response_matrix.size == 0 or cell_idx < 0 or cell_idx >= location_response_matrix.shape[0]:
        empty = np.zeros((0,), dtype=float)
        return empty, np.zeros((0,), dtype=int)
    return location_response_matrix[cell_idx], location_counts


def _empty_prepared_rate_map(
    *,
    smooth_sigma: float,
    min_bin_occupancy: float,
) -> PreparedRateMap:
    """Return an empty prepared rate map."""
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
    world: object,
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
    world: object,
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

    prepared_rate_maps: list[PreparedRateMap] = []
    for idx in indices:
        prepared_rate_maps.append(
            prepare_rate_map_from_location_responses(
                world,
                location_response_matrix[idx],
                location_counts,
                grid_res=grid_res,
                smooth_sigma=smooth_sigma,
                min_bin_occupancy=min_bin_occupancy,
            )
        )
    return tuple(prepared_rate_maps)


def prepare_rate_map_from_location_responses(
    world: object,
    location_responses: NDArray,
    location_counts: NDArray,
    *,
    grid_res: float | None = None,
    smooth_sigma: float,
    min_bin_occupancy: float = DEFAULT_RATE_MAP_MIN_BIN_OCCUPANCY,
) -> PreparedRateMap:
    """Build a prepared rate map from location-level responses.

    Args:
        world: Environment world with location coordinates.
        location_responses: Per-location mean responses for one cell.
        location_counts: Per-location visit counts.
        grid_res: Optional grid resolution for rasterization.
        smooth_sigma: Gaussian smoothing sigma in raster pixels.

    Returns:
        A prepared object containing additive response and occupancy surfaces,
        their smoothed variants, and the final masked rate map.
    """
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
        world,
        response_mass_by_location,
        grid_res=grid_res,
        include_mask=occupied_locations,
    )
    occupancy_grid, _, _ = rasterize_locations_additive(
        world,
        location_counts,
        grid_res=grid_res,
        include_mask=occupied_locations,
    )
    if response_mass_grid.size == 0 or occupancy_grid.size == 0:
        return _empty_prepared_rate_map(smooth_sigma=smooth_sigma, min_bin_occupancy=min_bin_occupancy)

    response_mass_grid = np.asarray(response_mass_grid, dtype=float)
    occupancy_grid = np.asarray(occupancy_grid, dtype=float)
    smoothed_response_mass_grid = response_mass_grid.copy()
    smoothed_occupancy_grid = occupancy_grid.copy()
    if smooth_sigma > 0:
        smoothed_response_mass_grid = gaussian_filter(
            smoothed_response_mass_grid,
            sigma=float(smooth_sigma),
            mode="constant",
            cval=0.0,
        )
        smoothed_occupancy_grid = gaussian_filter(
            smoothed_occupancy_grid,
            sigma=float(smooth_sigma),
            mode="constant",
            cval=0.0,
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


def plot_rate_map(
    ax: Axes,
    prepared_rate_map: PreparedRateMap,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "copper_r",
) -> Axes:
    """Plot one prepared rate map as a raster image."""
    rate_map = np.asarray(prepared_rate_map.rate_map, dtype=float)
    extent = _expand_degenerate_extent(prepared_rate_map.extent)

    if rate_map.size == 0 or not np.isfinite(rate_map).any():
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return ax

    finite = np.isfinite(rate_map)
    if vmin is None:
        vmin = float(np.nanmin(rate_map[finite])) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(rate_map[finite])) if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#d9d9d9")
    ax.imshow(
        rate_map,
        origin="upper",
        interpolation="nearest",
        cmap=cm,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="equal",
    )
    ax.axis("off")
    return ax


def plot_rate_map_mosaic(
    axes: Sequence[Axes] | Axes,
    prepared_rate_maps: Sequence[PreparedRateMap],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "copper_r",
) -> Sequence[Axes]:
    """Render multiple prepared rate maps into provided axes."""
    axes_list = list(np.ravel(axes)) if isinstance(axes, np.ndarray) else ([axes] if isinstance(axes, Axes) else list(axes))
    if not axes_list:
        return axes_list

    if not prepared_rate_maps:
        axes_list[0].text(0.5, 0.5, "No data", ha="center", va="center")
        for empty_ax in axes_list:
            empty_ax.axis("off")
        return axes_list

    for ax, prepared_rate_map in zip(axes_list, prepared_rate_maps):
        plot_rate_map(ax, prepared_rate_map, vmin=vmin, vmax=vmax, cmap=cmap)

    for ax in axes_list[len(prepared_rate_maps) :]:
        ax.axis("off")

    return axes_list


def _expand_degenerate_extent(
    extent: tuple[float, float, float, float],
    *,
    min_span: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return an ``imshow`` extent with non-zero span on both axes."""
    xmin, xmax, ymin, ymax = extent
    if xmin == xmax:
        half = min_span / 2
        xmin -= half
        xmax += half
    if ymin == ymax:
        half = min_span / 2
        ymin -= half
        ymax += half
    return xmin, xmax, ymin, ymax
