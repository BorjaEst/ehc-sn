"""Rasterization utilities for irregular spatial layouts."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


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
    coords = _world_coords(world)
    if coords.size == 0:
        empty = np.zeros((0, 0), dtype=float)
        return empty, empty.astype(bool), (0.0, 0.0, 0.0, 0.0)

    values = np.asarray(values, dtype=float)
    n_locations = coords.shape[0]
    if values.shape[0] < n_locations:
        pad = np.full((n_locations - values.shape[0],), np.nan, dtype=float)
        values = np.concatenate([values, pad])
    elif values.shape[0] > n_locations:
        values = values[:n_locations]

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
    count_grid = np.zeros((ny, nx), dtype=int)

    for (x, y), value in zip(coords, values, strict=False):
        if not np.isfinite(value):
            continue
        ix = int(round((x - xmin) / grid_res))
        iy = int(round((y - ymin) / grid_res))
        if 0 <= ix < nx and 0 <= iy < ny:
            sum_grid[iy, ix] += value
            count_grid[iy, ix] += 1

    grid = np.full((ny, nx), np.nan, dtype=float)
    valid = count_grid > 0
    grid[valid] = sum_grid[valid] / count_grid[valid]
    extent = (float(xmin), float(xmax), float(ymin), float(ymax))
    return grid, valid, extent


def _world_coords(world: object) -> NDArray:
    locations = getattr(world, "locations", [])
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
