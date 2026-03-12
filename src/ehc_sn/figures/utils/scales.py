"""Shared scaling utilities for figure normalization."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from matplotlib.colors import Normalize

from ehc_sn.rollouts.analysis import aggregate_rate_map


def build_shared_norm(arrays: Iterable[np.ndarray]) -> Normalize:
    """Create a shared normalization for multiple arrays.

    Args:
        arrays: Iterable of arrays to combine when computing min/max.

    Returns:
        A Normalize instance covering the finite range of the arrays.
    """
    array_list = list(arrays)
    values = np.concatenate([arr.ravel() for arr in array_list if arr.size]) if array_list else np.array([])
    finite_mask = np.isfinite(values)
    if values.size == 0 or not finite_mask.any():
        return Normalize(vmin=0.0, vmax=1.0)
    vmin = float(values[finite_mask].min())
    vmax = float(values[finite_mask].max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return Normalize(vmin=vmin, vmax=vmax)


def build_shared_minmax(arrays: Iterable[np.ndarray]) -> tuple[float, float]:
    """Compute a shared min/max range for multiple arrays.

    Args:
        arrays: Iterable of arrays to combine when computing min/max.

    Returns:
        Tuple of (vmin, vmax) for the finite values.
    """
    array_list = list(arrays)
    values = np.concatenate([arr.ravel() for arr in array_list if arr.size]) if array_list else np.array([])
    finite_mask = np.isfinite(values)
    if values.size == 0 or not finite_mask.any():
        return 0.0, 1.0
    vmin = float(values[finite_mask].min())
    vmax = float(values[finite_mask].max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def build_shared_map_range(
    cells_traces: np.ndarray | Sequence[np.ndarray],
    location_ids: np.ndarray,
    n_locations: int,
    cell_indices: int | Sequence[int],
) -> tuple[float, float]:
    """Compute a shared min/max range for rate-map values.

    Args:
        cells_traces: One or more cell activation traces with shape
            (T, B, C) or (T, C).
        location_ids: Location ids aligned with the time dimension.
        n_locations: Number of locations in the environment.
        cell_indices: Cell index or indices to include in the range.

    Returns:
        Tuple of (vmin, vmax) for the selected cells across traces.
    """
    if isinstance(cell_indices, int):
        indices = [cell_indices]
    else:
        indices = list(cell_indices)

    if isinstance(cells_traces, np.ndarray):
        traces = [cells_traces]
    else:
        traces = list(cells_traces)

    values: list[np.ndarray] = []
    for trace in traces:
        rate_map, _ = aggregate_rate_map(trace, location_ids, n_locations)
        if rate_map.size == 0:
            continue
        for idx in indices:
            if 0 <= idx < rate_map.shape[0]:
                values.append(rate_map[idx])

    return build_shared_minmax(values)
