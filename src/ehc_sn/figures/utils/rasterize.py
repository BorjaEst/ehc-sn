"""Rasterization utilities for irregular spatial layouts."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ehc_sn.figures.utils.axes import _environment_locations


def rasterize_locations_additive(
    world: object,
    values: NDArray,
    *,
    grid_res: float | None = None,
    include_mask: NDArray | None = None,
) -> tuple[NDArray, NDArray, tuple[float, float, float, float]]:
    """Rasterize per-location values by additive accumulation."""
    coords = _world_coords(world)
    if coords.size == 0:
        empty = np.zeros((0, 0), dtype=float)
        return empty, empty.astype(int), (0.0, 0.0, 0.0, 0.0)

    values = np.asarray(values, dtype=float)
    n_locations = coords.shape[0]
    if values.shape[0] < n_locations:
        pad = np.full((n_locations - values.shape[0],), np.nan, dtype=float)
        values = np.concatenate([values, pad])
    elif values.shape[0] > n_locations:
        values = values[:n_locations]

    if include_mask is None:
        include = np.isfinite(values)
    else:
        include = np.asarray(include_mask, dtype=bool)
        if include.shape[0] < n_locations:
            pad = np.zeros((n_locations - include.shape[0],), dtype=bool)
            include = np.concatenate([include, pad])
        elif include.shape[0] > n_locations:
            include = include[:n_locations]
        include &= np.isfinite(values)

    if grid_res is None:
        grid_res = _estimate_grid_res(coords)
    if not np.isfinite(grid_res) or grid_res <= 0:
        grid_res = 1.0

    xmin, ymin = np.min(coords, axis=0)
    xmax, ymax = np.max(coords, axis=0)

    nx = int(np.ceil((xmax - xmin) / grid_res)) + 1
    ny = int(np.ceil((ymax - ymin) / grid_res)) + 1
    nx = max(nx, 1)
    ny = max(ny, 1)

    sum_grid = np.zeros((ny, nx), dtype=float)
    hit_grid = np.zeros((ny, nx), dtype=int)

    for (x, y), value, use_value in zip(coords, values, include, strict=False):
        if not use_value:
            continue
        ix = int(round((x - xmin) / grid_res))
        iy = int(round((y - ymin) / grid_res))
        if 0 <= ix < nx and 0 <= iy < ny:
            sum_grid[iy, ix] += value
            hit_grid[iy, ix] += 1

    extent = (float(xmin), float(xmax), float(ymin), float(ymax))
    return sum_grid, hit_grid, extent


def rasterize_locations(
    world: object,
    values: NDArray,
    *,
    grid_res: float | None = None,
) -> tuple[NDArray, NDArray, tuple[float, float, float, float]]:
    """Rasterize per-location values onto a regular grid.

    Args:
        world: Environment world with location coordinates.
        values: Per-location values aligned with ``world.locations``.
        grid_res: Grid resolution in world units. If None, estimated from
            nearest-neighbor distances.

    Returns:
        Tuple of (grid, mask, extent) where grid is a 2D array of rasterized
        values, mask indicates valid pixels, and extent is
        (xmin, xmax, ymin, ymax).
    """
    sum_grid, count_grid, extent = rasterize_locations_additive(world, values, grid_res=grid_res)
    ny, nx = sum_grid.shape
    grid = np.full((ny, nx), np.nan, dtype=float)
    valid = count_grid > 0
    grid[valid] = sum_grid[valid] / count_grid[valid]
    return grid, valid, extent


def _world_coords(world: object) -> NDArray:
    locations = _environment_locations(world)
    coords = [[loc["o"], loc["y"]] for loc in locations]
    return np.asarray(coords, dtype=float)


def _estimate_grid_res(coords: NDArray) -> float:
    if coords.shape[0] < 2:
        return 1.0
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nn = distances[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if nn.size == 0:
        return 1.0
    return float(np.median(nn))
