"""Selectors for MEC figure templates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ehc_sn.analysis.spatial import SpatialBinGeometry
from ehc_sn.analysis.spatial.gridness import (
    compute_gridness,
    estimate_grid_spacing_orientation,
)
from ehc_sn.figures._contracts import AnyWorld, PreparedRateMap
from ehc_sn.figures.plots.autocorr import compute_spatial_autocorrelogram
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.spatial import prepare_rate_maps, spatial_rate_smooth_sigma
from ehc_sn.traces.trace_tree import TraceTree

# ── Canonical trace / meta path constants ────────────────────────────────────
TRACE_KEY_LOCATION_IDS = "world_step/location_ids"
TRACE_KEY_MEC_CELLS = "diagnostic/mec/location_mean"
META_KEY_ENVIRONMENTS = "environments"


@dataclass
class MECSummaryFigureData:
    n_freq: int
    env_idx: int
    freq_idxs: list[int]
    world: AnyWorld
    location_ids: NDArray
    cells: list[NDArray]
    prepared_rate_maps: list[tuple[PreparedRateMap, ...]]


@dataclass
class MECCellFigureData:
    env_idx: int
    freq_idx: int
    world: AnyWorld
    location_ids: NDArray
    cells: NDArray
    prepared_rate_maps: tuple[PreparedRateMap, ...]


# ── Grid-metrics data and selector ───────────────────────────────────────────


@dataclass(frozen=True)
class MECGridMetricsData:
    """Per-frequency, per-cell gridness metrics for MEC cells.

    Attributes
    ----------
    freq_indices : NDArray
        1D array of frequency indices, shape ``(F,)``.
    cell_indices : NDArray
        1D array of cell indices, shape ``(C,)``.
    gridness : NDArray
        Gridness scores, shape ``(F, C)``.  ``NaN`` for cells where the
        annular mask had too few valid pixels.
    spacing : NDArray
        Grid spacing in world units, shape ``(F, C)``.  ``NaN`` when fewer
        than six credible first-ring peaks were found.
    orientation_deg : NDArray
        Grid orientation wrapped to ``[0°, 60°)``, shape ``(F, C)``.
        ``NaN`` when spacing is ``NaN``.
    valid_pixel_count : NDArray
        Number of valid pixel pairs used in each rotational correlation,
        shape ``(F, C)``.
    inner_radius : float
        Inner annular-mask radius in world units (shared across cells).
    outer_radius : float
        Outer annular-mask radius in world units (shared across cells).
    mask_strategy : str
        How mask radii were chosen.
    """

    freq_indices: NDArray
    cell_indices: NDArray
    gridness: NDArray
    spacing: NDArray
    orientation_deg: NDArray
    valid_pixel_count: NDArray
    inner_radius: float
    outer_radius: float
    mask_strategy: str


def select_mec_grid_metrics(trace: TraceTree, ctx: FigureContext) -> MECGridMetricsData:
    """Compute gridness, spacing, and orientation for all MEC cells.

    Parameters
    ----------
    trace : TraceTree
        Trace containing MEC location-mean activation and location-id leaves.
    ctx : FigureContext
        Figure context specifying environment index.

    Returns
    -------
    MECGridMetricsData
        Per-frequency, per-cell gridness metrics.
    """
    n_freq = trace.n_freq(TRACE_KEY_MEC_CELLS)
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idxs = [trace.validate_freq_idx(TRACE_KEY_MEC_CELLS, f) for f in range(n_freq)]
    world = trace.get_world(env_idx)
    location_ids = trace.get(TRACE_KEY_LOCATION_IDS)[:, env_idx]

    # Determine shared cell count from the first frequency band.
    first_cells = trace.get(f"{TRACE_KEY_MEC_CELLS}/0")[:, env_idx, :]
    n_cells = int(first_cells.shape[-1])
    cell_indices = np.arange(n_cells)

    # Prepare rate maps for all cells at each frequency.
    all_rate_maps: list[tuple[PreparedRateMap, ...]] = []
    for f in freq_idxs:
        cells_t = trace.get(f"{TRACE_KEY_MEC_CELLS}/{f}")[:, env_idx, :]
        rms = prepare_rate_maps(world, cells_t, location_ids, smooth_sigma=spatial_rate_smooth_sigma(world))
        all_rate_maps.append(rms)

    # Derive spatial bin geometry from the first rate map.
    first_rm = all_rate_maps[0][0] if all_rate_maps and all_rate_maps[0] else None
    if first_rm is None or first_rm.rate_map.size == 0:
        # No data — return all-NaN arrays.
        return MECGridMetricsData(
            freq_indices=np.asarray(freq_idxs, dtype=int),
            cell_indices=cell_indices,
            gridness=np.full((n_freq, n_cells), np.nan, dtype=float),
            spacing=np.full((n_freq, n_cells), np.nan, dtype=float),
            orientation_deg=np.full((n_freq, n_cells), np.nan, dtype=float),
            valid_pixel_count=np.zeros((n_freq, n_cells), dtype=int),
            inner_radius=0.0,
            outer_radius=0.0,
            mask_strategy="fractional",
        )

    geom = SpatialBinGeometry.from_extent_and_shape(first_rm.extent, first_rm.rate_map.shape)

    gridness = np.full((n_freq, n_cells), np.nan, dtype=float)
    spacing = np.full((n_freq, n_cells), np.nan, dtype=float)
    orientation_deg = np.full((n_freq, n_cells), np.nan, dtype=float)
    valid_pixel_count = np.zeros((n_freq, n_cells), dtype=int)
    inner_radius: float = 0.0
    outer_radius: float = 0.0
    mask_strategy: str = "fractional"

    for f_idx, freq_rate_maps in enumerate(all_rate_maps):
        for c_idx, rm in enumerate(freq_rate_maps):
            if rm.rate_map.size == 0:
                continue

            autocorr = compute_spatial_autocorrelogram(rm.rate_map, rm.valid_mask, min_overlap=4)

            try:
                g_result = compute_gridness(autocorr, geometry=geom)
            except (ValueError, RuntimeError):
                continue

            inner_radius = g_result.inner_radius
            outer_radius = g_result.outer_radius
            mask_strategy = g_result.mask_strategy
            gridness[f_idx, c_idx] = g_result.score
            valid_pixel_count[f_idx, c_idx] = g_result.valid_pixel_count

            s_result = estimate_grid_spacing_orientation(autocorr, geometry=geom)
            spacing[f_idx, c_idx] = s_result.spacing
            orientation_deg[f_idx, c_idx] = s_result.orientation_deg

    return MECGridMetricsData(
        freq_indices=np.asarray(freq_idxs, dtype=int),
        cell_indices=cell_indices,
        gridness=gridness,
        spacing=spacing,
        orientation_deg=orientation_deg,
        valid_pixel_count=valid_pixel_count,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        mask_strategy=mask_strategy,
    )
