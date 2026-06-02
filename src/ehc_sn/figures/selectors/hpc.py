"""Selectors for HPC figure templates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ehc_sn.analysis.spatial import (
    SpatialBinGeometry,
    compute_rate_map_stats,
)
from ehc_sn.figures._contracts import AnyWorld, PreparedRateMap
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.spatial import (
    prepare_rate_maps,
    spatial_rate_smooth_sigma,
)
from ehc_sn.traces.trace_tree import TraceTree

# ── Canonical trace / meta path constants ────────────────────────────────────
TRACE_KEY_LOCATION_IDS = "world_step/location_ids"
TRACE_KEY_HPC_CELLS = "diagnostic/hpc/location_mean"
TRACE_KEY_HPC_MEMORY = "diagnostic/hpc/memory"
META_KEY_ENVIRONMENTS = "environments"


@dataclass
class HPCSummaryFigureData:
    n_freq: int
    env_idx: int
    freq_idxs: list[int]
    world: AnyWorld
    location_ids: NDArray
    cells: list[NDArray]
    prepared_rate_maps: list[tuple[PreparedRateMap, ...]]
    memory_g_cued: NDArray
    memory_x_cued: NDArray


@dataclass
class HPCCellFigureData:
    env_idx: int
    freq_idx: int
    world: AnyWorld
    location_ids: NDArray
    cells: NDArray
    prepared_rate_maps: tuple[PreparedRateMap, ...]


# ── Place-metrics data and selector ──────────────────────────────────────────


@dataclass(frozen=True)
class HPCPlaceMetricsData:
    """Per-cell place-field metrics for HPC cells.

    Attributes
    ----------
    cell_indices : NDArray
        1D array of cell indices, shape ``(C,)``.
    peak_rate : NDArray
        Peak firing rate per cell, shape ``(C,)``.
    mean_rate : NDArray
        Occupancy-weighted mean rate per cell, shape ``(C,)``.
    spatial_information : NDArray
        Skaggs-style spatial information per cell, shape ``(C,)``.
    sparsity : NDArray
        Sparsity per cell, shape ``(C,)``.
    field_x : NDArray
        Field center world x-coordinate per cell, shape ``(C,)``.
        ``NaN`` when SI is below ``min_spatial_information_for_field``.
    field_y : NDArray
        Field center world y-coordinate per cell, shape ``(C,)``.
        ``NaN`` when SI is below ``min_spatial_information_for_field``.
    field_area : NDArray
        Field area (world units²) per cell, shape ``(C,)``.
        ``NaN`` when SI is below ``min_spatial_information_for_field``.

    top_rate_maps : tuple[NDArray[np.floating], ...]
        Rate-map arrays for the top-*max_examples* cells (spatial information
        descending).  Length at most *max_examples*.
    top_cell_indices : NDArray
        Cell indices corresponding to ``top_rate_maps``.
    top_spatial_information : NDArray
        Spatial information values for the top examples.
    extent : tuple[float, float, float, float]
        World extent shared by all rate maps.
    """

    cell_indices: NDArray
    peak_rate: NDArray
    mean_rate: NDArray
    spatial_information: NDArray
    sparsity: NDArray
    field_x: NDArray
    field_y: NDArray
    field_area: NDArray

    top_rate_maps: tuple[NDArray[np.floating], ...] = ()
    top_cell_indices: NDArray[np.integer] = field(
        default_factory=lambda: np.empty(0, dtype=int)
    )
    top_spatial_information: NDArray[np.floating] = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )

    extent: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def select_hpc_place_metrics(
    trace: TraceTree,
    ctx: FigureContext,
    *,
    max_examples: int = 4,
    min_spatial_information_for_field: float = 0.1,
) -> HPCPlaceMetricsData:
    """Compute place-field metrics for all HPC cells.

    Iterates over the first HPC frequency band only (``freq=0``), computes
    rate-map statistics for every cell, and selects top-N examples by
    descending finite spatial information.

    Parameters
    ----------
    trace : TraceTree
        Trace containing HPC location-mean activation and location-id leaves.
    ctx : FigureContext
        Figure context specifying environment index.
    max_examples : int
        Maximum number of top-example rate maps to return (sorted by
        descending finite spatial information).
    min_spatial_information_for_field : float
        Minimum spatial information (bits) for ``field_x`` / ``field_y`` /
        ``field_area`` to be returned as valid numbers rather than ``NaN``.

    Returns
    -------
    HPCPlaceMetricsData
        Per-cell place-field metrics with optional top-example rate maps.
    """
    env_idx = trace.validate_env_idx(ctx.env_idx)
    world = trace.get_world(env_idx)
    location_ids = trace.get(TRACE_KEY_LOCATION_IDS)[:, env_idx]

    # Use the first HPC frequency band (all layers collapsed).
    cells = trace.get(f"{TRACE_KEY_HPC_CELLS}/0")[:, env_idx, :]
    n_cells = int(cells.shape[-1])
    cell_indices = np.arange(n_cells)

    # ── Prepare rate maps ───────────────────────────────────────────────────
    prepared = prepare_rate_maps(
        world,
        cells,
        location_ids,
        smooth_sigma=spatial_rate_smooth_sigma(world),
    )
    if not prepared:
        return HPCPlaceMetricsData(
            cell_indices=cell_indices,
            peak_rate=np.full(n_cells, np.nan, dtype=float),
            mean_rate=np.full(n_cells, np.nan, dtype=float),
            spatial_information=np.full(n_cells, np.nan, dtype=float),
            sparsity=np.full(n_cells, np.nan, dtype=float),
            field_x=np.full(n_cells, np.nan, dtype=float),
            field_y=np.full(n_cells, np.nan, dtype=float),
            field_area=np.full(n_cells, np.nan, dtype=float),
        )

    extent = prepared[0].extent
    geometry = SpatialBinGeometry.from_extent_and_shape(
        extent, prepared[0].rate_map.shape
    )

    # ── Per-cell metrics ────────────────────────────────────────────────────
    peak_rate = np.full(n_cells, np.nan, dtype=float)
    mean_rate = np.full(n_cells, np.nan, dtype=float)
    spatial_information = np.full(n_cells, np.nan, dtype=float)
    sparsity = np.full(n_cells, np.nan, dtype=float)
    field_x = np.full(n_cells, np.nan, dtype=float)
    field_y = np.full(n_cells, np.nan, dtype=float)
    field_area = np.full(n_cells, np.nan, dtype=float)

    for c_idx in range(min(n_cells, len(prepared))):
        rm = prepared[c_idx]
        if rm.rate_map.size == 0:
            continue
        stats = compute_rate_map_stats(
            rm.rate_map,
            occupancy=rm.occupancy_grid,
            extent=extent,
            geometry=geometry,
            min_spatial_information_for_field=min_spatial_information_for_field,
        )
        peak_rate[c_idx] = stats.peak_rate
        mean_rate[c_idx] = stats.mean_rate
        spatial_information[c_idx] = stats.spatial_information
        sparsity[c_idx] = stats.sparsity
        field_x[c_idx] = stats.field_x
        field_y[c_idx] = stats.field_y
        field_area[c_idx] = stats.field_area

    # ── Select top-N examples by spatial information ────────────────────────
    top_rate_maps: tuple[NDArray[np.floating], ...] = ()
    top_cell_indices_arr = np.empty(0, dtype=int)
    top_si_arr = np.empty(0, dtype=float)

    if max_examples > 0:
        candidates: list[tuple[int, float, NDArray]] = []
        for c_idx in range(n_cells):
            si = float(spatial_information[c_idx])
            if np.isfinite(si) and c_idx < len(prepared):
                candidates.append((c_idx, si, prepared[c_idx].rate_map))
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = candidates[:max_examples]
        if selected:
            top_rate_maps = tuple(s[2] for s in selected)
            top_cell_indices_arr = np.array([s[0] for s in selected], dtype=int)
            top_si_arr = np.array([s[1] for s in selected], dtype=float)

    return HPCPlaceMetricsData(
        cell_indices=cell_indices,
        peak_rate=peak_rate,
        mean_rate=mean_rate,
        spatial_information=spatial_information,
        sparsity=sparsity,
        field_x=field_x,
        field_y=field_y,
        field_area=field_area,
        top_rate_maps=top_rate_maps,
        top_cell_indices=top_cell_indices_arr,
        top_spatial_information=top_si_arr,
        extent=extent,
    )


# ── Existing selectors (unchanged) ───────────────────────────────────────────


def select_hpc_summary(
    trace: TraceTree, ctx: FigureContext
) -> HPCSummaryFigureData:
    n_freq = trace.n_freq(TRACE_KEY_HPC_CELLS)
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idxs = [
        trace.validate_freq_idx(TRACE_KEY_HPC_CELLS, f) for f in range(n_freq)
    ]
    world = trace.get_world(env_idx)
    location_ids = trace.get(TRACE_KEY_LOCATION_IDS)[:, env_idx]
    cells = [
        trace.get(f"{TRACE_KEY_HPC_CELLS}/{f}")[:, env_idx, :]
        for f in range(n_freq)
    ]
    rate_maps = [
        prepare_rate_maps(
            world,
            c,
            location_ids,
            smooth_sigma=spatial_rate_smooth_sigma(world),
        )
        for c in cells
    ]
    memory_g_cued = trace.get(f"{TRACE_KEY_HPC_MEMORY}/g_cued")[-1, env_idx]
    memory_x_cued = trace.get(f"{TRACE_KEY_HPC_MEMORY}/x_cued")[-1, env_idx]
    return HPCSummaryFigureData(
        n_freq=n_freq,
        env_idx=env_idx,
        freq_idxs=freq_idxs,
        world=world,
        location_ids=location_ids,
        cells=cells,
        prepared_rate_maps=rate_maps,
        memory_g_cued=memory_g_cued,
        memory_x_cued=memory_x_cued,
    )


def select_hpc_cell(trace: TraceTree, ctx: FigureContext) -> HPCCellFigureData:
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idx = trace.validate_freq_idx(TRACE_KEY_HPC_CELLS, ctx.freq_idx)
    world = trace.get_world(env_idx)
    location_ids = trace.get(TRACE_KEY_LOCATION_IDS)[:, env_idx]
    cells = trace.get(f"{TRACE_KEY_HPC_CELLS}/{freq_idx}")[:, env_idx, :]
    rate_maps = prepare_rate_maps(
        world,
        cells,
        location_ids,
        smooth_sigma=spatial_rate_smooth_sigma(world),
    )
    return HPCCellFigureData(
        env_idx=env_idx,
        freq_idx=freq_idx,
        world=world,
        location_ids=location_ids,
        cells=cells,
        prepared_rate_maps=rate_maps,
    )
