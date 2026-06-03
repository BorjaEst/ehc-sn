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
from ehc_sn.traces.keys import (
    HPC_TRACE_KEY_CELLS,
    HPC_TRACE_KEY_MEMORY,
    META_KEY_ENVIRONMENTS,
    WORLD_TRACE_KEY_LOCATION_IDS,
)
from ehc_sn.traces.trace_tree import TraceTree


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
class _HPCPlaceComputed:
    """Internal container for the private compute-once helper.

    Caches the prepared rate maps and per-cell stats so that the
    place-metrics and rate-map-mosaic selectors can share the same
    computation without rebuilding rate maps.
    """

    prepared: tuple[PreparedRateMap, ...]
    cell_indices: NDArray[np.integer]
    peak_rate: NDArray[np.floating]
    mean_rate: NDArray[np.floating]
    spatial_information: NDArray[np.floating]
    sparsity: NDArray[np.floating]
    field_x: NDArray[np.floating]
    field_y: NDArray[np.floating]
    field_area: NDArray[np.floating]
    extent: tuple[float, float, float, float]


def _compute_hpc_place_metrics(
    trace: TraceTree,
    ctx: FigureContext,
    *,
    min_spatial_information_for_field: float = 0.1,
) -> _HPCPlaceComputed:
    """Compute rate maps and per-cell place-field stats — shared helper.

    Both ``select_hpc_place_metrics`` and ``select_hpc_rate_map_mosaic``
    delegate to this function so that the expensive rate-map preparation
    and metric computation happen exactly once per call path.

    Parameters
    ----------
    trace : TraceTree
        Trace containing HPC location-mean activation and location-id leaves.
    ctx : FigureContext
        Figure context specifying environment index.
    min_spatial_information_for_field : float
        Minimum spatial information for valid field centers.

    Returns
    -------
    _HPCPlaceComputed
        Prepared rate maps and per-cell metrics.
    """
    env_idx = trace.validate_env_idx(ctx.env_idx)
    world = trace.get_world(env_idx)
    location_ids = trace.get(WORLD_TRACE_KEY_LOCATION_IDS)[:, env_idx]

    cells = trace.get(f"{HPC_TRACE_KEY_CELLS}/0")[:, env_idx, :]
    n_cells = int(cells.shape[-1])
    cell_indices = np.arange(n_cells)

    prepared = prepare_rate_maps(
        world,
        cells,
        location_ids,
        smooth_sigma=spatial_rate_smooth_sigma(world),
    )

    if not prepared:
        return _HPCPlaceComputed(
            prepared=(),
            cell_indices=cell_indices,
            peak_rate=np.full(n_cells, np.nan, dtype=float),
            mean_rate=np.full(n_cells, np.nan, dtype=float),
            spatial_information=np.full(n_cells, np.nan, dtype=float),
            sparsity=np.full(n_cells, np.nan, dtype=float),
            field_x=np.full(n_cells, np.nan, dtype=float),
            field_y=np.full(n_cells, np.nan, dtype=float),
            field_area=np.full(n_cells, np.nan, dtype=float),
            extent=(0.0, 0.0, 0.0, 0.0),
        )

    extent = prepared[0].extent
    geometry = SpatialBinGeometry.from_extent_and_shape(
        extent, prepared[0].rate_map.shape
    )

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

    return _HPCPlaceComputed(
        prepared=prepared,
        cell_indices=cell_indices,
        peak_rate=peak_rate,
        mean_rate=mean_rate,
        spatial_information=spatial_information,
        sparsity=sparsity,
        field_x=field_x,
        field_y=field_y,
        field_area=field_area,
        extent=extent,
    )


@dataclass(frozen=True)
class HPCPlaceMetricsData:
    """Per-cell place-field metrics for HPC cells.
    ...
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

    Delegates to the private :func:`_compute_hpc_place_metrics` shared
    helper so that the rate-map / metric computation is not duplicated
    when the mosaic selector calls the same pathway.

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
    computed = _compute_hpc_place_metrics(
        trace,
        ctx,
        min_spatial_information_for_field=min_spatial_information_for_field,
    )

    # Select top-N examples by spatial information.
    top_rate_maps: tuple[NDArray[np.floating], ...] = ()
    top_cell_indices_arr = np.empty(0, dtype=int)
    top_si_arr = np.empty(0, dtype=float)

    if max_examples > 0:
        candidates: list[tuple[int, float, NDArray]] = []
        for c_idx in range(len(computed.cell_indices)):
            si = float(computed.spatial_information[c_idx])
            if np.isfinite(si) and c_idx < len(computed.prepared):
                candidates.append(
                    (c_idx, si, computed.prepared[c_idx].rate_map)
                )
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = candidates[:max_examples]
        if selected:
            top_rate_maps = tuple(s[2] for s in selected)
            top_cell_indices_arr = np.array([s[0] for s in selected], dtype=int)
            top_si_arr = np.array([s[1] for s in selected], dtype=float)

    return HPCPlaceMetricsData(
        cell_indices=computed.cell_indices,
        peak_rate=computed.peak_rate,
        mean_rate=computed.mean_rate,
        spatial_information=computed.spatial_information,
        sparsity=computed.sparsity,
        field_x=computed.field_x,
        field_y=computed.field_y,
        field_area=computed.field_area,
        top_rate_maps=top_rate_maps,
        top_cell_indices=top_cell_indices_arr,
        top_spatial_information=top_si_arr,
        extent=computed.extent,
    )


# ── Rate-map mosaic data and selector ────────────────────────────────────────


@dataclass(frozen=True)
class HPCRateMapMosaicData:
    """Population rate-map mosaic for HPC cells.

    Attributes
    ----------
    tiles : tuple[HPCRateMapMosaicTile, ...]
        Flat list of tiles ordered by descending finite spatial information.
    extent : tuple[float, float, float, float]
        World extent shared by all rate maps.
    """

    tiles: tuple[HPCRateMapMosaicTile, ...]
    extent: tuple[float, float, float, float]


@dataclass(frozen=True)
class HPCRateMapMosaicTile:
    """One tile in the HPC rate-map population mosaic."""

    cell: int
    spatial_information: float
    peak_rate: float
    rate_map: NDArray[np.floating]


def select_hpc_rate_map_mosaic(
    trace: TraceTree,
    ctx: FigureContext,
    *,
    max_cells: int = 64,
    min_spatial_information_for_field: float = 0.1,
) -> HPCRateMapMosaicData:
    """Select rate maps for a population mosaic, ordered by spatial information.

    Parameters
    ----------
    trace : TraceTree
        Trace containing HPC location-mean activation and location-id leaves.
    ctx : FigureContext
        Figure context specifying environment index.
    max_cells : int
        Maximum number of cells (tiles) in the mosaic.
    min_spatial_information_for_field : float
        Minimum spatial information for valid field centers (passed through
        to the shared helper).

    Returns
    -------
    HPCRateMapMosaicData
        Flat tile list ordered by descending finite spatial information.
    """
    computed = _compute_hpc_place_metrics(
        trace,
        ctx,
        min_spatial_information_for_field=min_spatial_information_for_field,
    )

    # Collect cells with finite SI, sorted descending.
    si = np.asarray(computed.spatial_information, dtype=float)
    valid = np.isfinite(si)
    order = np.flatnonzero(valid)[np.argsort(si[valid])[::-1]][:max_cells]

    tiles: list[HPCRateMapMosaicTile] = []
    for idx in order:
        c_idx = int(idx)
        tiles.append(
            HPCRateMapMosaicTile(
                cell=c_idx,
                spatial_information=float(si[c_idx]),
                peak_rate=float(computed.peak_rate[c_idx]),
                rate_map=computed.prepared[c_idx].rate_map,
            )
        )

    return HPCRateMapMosaicData(
        tiles=tuple(tiles),
        extent=computed.extent,
    )


# ── Existing selectors (unchanged) ───────────────────────────────────────────


def select_hpc_summary(
    trace: TraceTree, ctx: FigureContext
) -> HPCSummaryFigureData:
    n_freq = trace.n_freq(HPC_TRACE_KEY_CELLS)
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idxs = [
        trace.validate_freq_idx(HPC_TRACE_KEY_CELLS, f) for f in range(n_freq)
    ]
    world = trace.get_world(env_idx)
    location_ids = trace.get(WORLD_TRACE_KEY_LOCATION_IDS)[:, env_idx]
    cells = [
        trace.get(f"{HPC_TRACE_KEY_CELLS}/{f}")[:, env_idx, :]
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
    memory_g_cued = trace.get(f"{HPC_TRACE_KEY_MEMORY}/g_cued")[-1, env_idx]
    memory_x_cued = trace.get(f"{HPC_TRACE_KEY_MEMORY}/x_cued")[-1, env_idx]
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
    freq_idx = trace.validate_freq_idx(HPC_TRACE_KEY_CELLS, ctx.freq_idx)
    world = trace.get_world(env_idx)
    location_ids = trace.get(WORLD_TRACE_KEY_LOCATION_IDS)[:, env_idx]
    cells = trace.get(f"{HPC_TRACE_KEY_CELLS}/{freq_idx}")[:, env_idx, :]
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
