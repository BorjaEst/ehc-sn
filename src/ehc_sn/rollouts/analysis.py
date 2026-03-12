"""Rollout analysis helpers shared by figures and evaluation code."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# =================================================================================================
def aggregate_rate_map(  # ------------------------------------------------------------------------
    cells_trace: NDArray, location_ids: NDArray | list[int], n_locations: int,
) -> tuple[NDArray, NDArray]:  # fmt: skip
    """Aggregate cell activity into per-location mean activity maps.

    Args:
        cells_trace: Cell activations with shape ``(T, C)`` or ``(T, B, C)``.
        location_ids: Visited location ids with shape ``(T,)`` or ``(T, B)``.
        n_locations: Total number of locations in the environment.

    Returns:
        Tuple ``(rate_map, counts)`` where ``rate_map`` has shape
        ``(C, n_locations)`` and ``counts`` has shape ``(n_locations,)``.
        Locations never visited are filled with ``NaN`` in ``rate_map``.
    """
    if n_locations <= 0:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=int)

    cells = np.asarray(cells_trace, dtype=float)
    locs = np.asarray(location_ids, dtype=int)

    if cells.ndim == 2:
        if locs.ndim != 1 or locs.shape[0] != cells.shape[0]:
            raise ValueError("location_ids must have shape (T,) when cells_trace has shape (T, C)")
        cells = cells[:, None, :]
        locs = locs[:, None]
    elif cells.ndim == 3:
        if locs.ndim == 1:
            if cells.shape[1] != 1 or locs.shape[0] != cells.shape[0]:
                raise ValueError("location_ids must have shape (T, B) for batched cell traces")
            locs = locs[:, None]
        elif locs.ndim != 2 or locs.shape[:2] != cells.shape[:2]:
            raise ValueError("location_ids must match the leading (T, B) dimensions of cells_trace")
    else:
        raise ValueError("cells_trace must have shape (T, C) or (T, B, C)")

    n_cells = int(cells.shape[-1])
    sums = np.zeros((n_cells, n_locations), dtype=float)
    counts = np.zeros((n_locations,), dtype=int)

    flat_cells = cells.reshape(-1, n_cells)
    flat_locs = locs.reshape(-1)
    valid_rows = np.isfinite(flat_cells).all(axis=1)

    for row_values, loc_id, is_valid in zip(flat_cells, flat_locs, valid_rows, strict=False):
        if not is_valid:
            continue
        if 0 <= int(loc_id) < n_locations:
            sums[:, int(loc_id)] += row_values
            counts[int(loc_id)] += 1

    rate_map = np.full((n_cells, n_locations), np.nan, dtype=float)
    visited = counts > 0
    if visited.any():
        rate_map[:, visited] = sums[:, visited] / counts[visited]
    return rate_map, counts


# =================================================================================================
__all__ = ["aggregate_rate_map"]
