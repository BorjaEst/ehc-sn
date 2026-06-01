"""Selectors for MEC figure templates."""

from __future__ import annotations

from dataclasses import dataclass, field

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
from ehc_sn.figures.selectors.spatial import (
    prepare_rate_maps,
    spatial_rate_smooth_sigma,
)
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


# ── Grid-metrics data and selectors ──────────────────────────────────────────


@dataclass(frozen=True)
class _CellMetrics:
    """Internal per-cell metrics used by both grid-metrics and example selectors."""

    gridness: float
    spacing: float
    orientation_deg: float
    valid_pixel_count: int
    rate_map: PreparedRateMap
    autocorr: NDArray


def _compute_mec_grid_metrics(
    world: AnyWorld,
    cells_traces: list[NDArray],
    location_ids: NDArray,
    *,
    smooth_sigma: float,
) -> tuple[list[list[_CellMetrics]], SpatialBinGeometry | None]:
    """Compute rate maps, autocorrelograms, and gridness for all (freq, cell).

    Parameters
    ----------
    world : AnyWorld
        Environment world object.
    cells_traces : list[NDArray]
        One ``(T, C)`` array per frequency band.
    location_ids : NDArray
        Per-timestep location indices, shape ``(T,)``.
    smooth_sigma : float
        Rate-map smoothing sigma.

    Returns
    -------
    all_metrics : list[list[_CellMetrics]]
        ``all_metrics[f][c]`` with per-cell metrics.
    geometry : SpatialBinGeometry or None
        Bin geometry derived from the first valid rate map, or ``None``.
    """
    all_rate_maps: list[tuple[PreparedRateMap, ...]] = []
    for cells_t in cells_traces:
        rms = prepare_rate_maps(
            world, cells_t, location_ids, smooth_sigma=smooth_sigma
        )
        all_rate_maps.append(rms)

    if not all_rate_maps or not all_rate_maps[0]:
        return [], None

    first_rm = all_rate_maps[0][0]
    if first_rm.rate_map.size == 0:
        return [], None

    geom = SpatialBinGeometry.from_extent_and_shape(
        first_rm.extent, first_rm.rate_map.shape
    )

    all_metrics: list[list[_CellMetrics]] = []
    for freq_rate_maps in all_rate_maps:
        freq_metrics: list[_CellMetrics] = []
        for rm in freq_rate_maps:
            if rm.rate_map.size == 0:
                freq_metrics.append(
                    _CellMetrics(
                        gridness=float("nan"),
                        spacing=float("nan"),
                        orientation_deg=float("nan"),
                        valid_pixel_count=0,
                        rate_map=rm,
                        autocorr=np.zeros((0, 0), dtype=float),
                    )
                )
                continue

            autocorr = compute_spatial_autocorrelogram(
                rm.rate_map, rm.valid_mask, min_overlap=4
            )

            g_score = float("nan")
            spacing = float("nan")
            orientation_deg = float("nan")
            valid_px = 0

            try:
                g_result = compute_gridness(autocorr, geometry=geom)
                g_score = g_result.score
                valid_px = g_result.valid_pixel_count
            except (ValueError, RuntimeError):
                pass

            try:
                s_result = estimate_grid_spacing_orientation(
                    autocorr, geometry=geom
                )
                spacing = s_result.spacing
                orientation_deg = s_result.orientation_deg
            except (ValueError, RuntimeError):
                pass

            freq_metrics.append(
                _CellMetrics(
                    gridness=g_score,
                    spacing=spacing,
                    orientation_deg=orientation_deg,
                    valid_pixel_count=valid_px,
                    rate_map=rm,
                    autocorr=autocorr,
                )
            )
        all_metrics.append(freq_metrics)

    return all_metrics, geom


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
    # ── Top-example fields (populated when select_mec_grid_metrics receives
    #    max_examples > 0) ───────────────────────────────────────────────────
    top_autocorrs: tuple[NDArray[np.floating], ...] = ()
    top_freq_indices: NDArray[np.integer] = field(
        default_factory=lambda: np.empty(0, dtype=int)
    )
    top_cell_indices: NDArray[np.integer] = field(
        default_factory=lambda: np.empty(0, dtype=int)
    )
    top_gridness: NDArray[np.floating] = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )


# ── MEC autocorr mosaic data and selector ────────────────────────────────────


@dataclass(frozen=True)
class MECAutocorrMosaicData:
    """Population autocorrelogram mosaic across frequency bands.

    Attributes
    ----------
    freq_indices : NDArray
        1D array of frequency indices, shape ``(F,)``.
    gridness_by_freq : list[NDArray]
        Gridness scores per frequency band, each of shape ``(C_f,)``.
    autocorrs_by_freq : list[tuple[NDArray[np.floating], ...]]
        Autocorrelograms per frequency band, one tuple per band.
    cell_indices_by_freq : list[NDArray]
        Cell indices per frequency band.
    extent : tuple[float, float, float, float]
        World extent shared by all rate maps.
    """

    freq_indices: NDArray
    gridness_by_freq: list[NDArray]
    autocorrs_by_freq: list[tuple[NDArray[np.floating], ...]]
    cell_indices_by_freq: list[NDArray]
    extent: tuple[float, float, float, float]


def select_mec_autocorr_mosaic(
    trace: TraceTree,
    ctx: FigureContext,
    *,
    max_cells_per_freq: int = 36,
) -> MECAutocorrMosaicData:
    """Select autocorrelograms for a population mosaic, ordered by gridness.

    Parameters
    ----------
    trace : TraceTree
        Trace containing MEC location-mean activation and location-id leaves.
    ctx : FigureContext
        Figure context specifying environment index.
    max_cells_per_freq : int
        Maximum number of cells to show per frequency band.

    Returns
    -------
    MECAutocorrMosaicData
        Per-frequency autocorrelograms and gridness scores.
    """
    n_freq = trace.n_freq(TRACE_KEY_MEC_CELLS)
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idxs = [
        trace.validate_freq_idx(TRACE_KEY_MEC_CELLS, f) for f in range(n_freq)
    ]
    world = trace.get_world(env_idx)
    location_ids = trace.get(TRACE_KEY_LOCATION_IDS)[:, env_idx]

    cells_traces = [
        trace.get(f"{TRACE_KEY_MEC_CELLS}/{f}")[:, env_idx, :]
        for f in freq_idxs
    ]

    all_metrics, _geom = _compute_mec_grid_metrics(
        world,
        cells_traces,
        location_ids,
        smooth_sigma=spatial_rate_smooth_sigma(world),
    )

    if not all_metrics or _geom is None:
        return MECAutocorrMosaicData(
            freq_indices=np.asarray(freq_idxs, dtype=int),
            gridness_by_freq=[],
            autocorrs_by_freq=[],
            cell_indices_by_freq=[],
            extent=(0.0, 0.0, 0.0, 0.0),
        )

    extent = all_metrics[0][0].rate_map.extent

    gridness_by_freq: list[NDArray] = []
    autocorrs_by_freq: list[tuple[NDArray[np.floating], ...]] = []
    cell_indices_by_freq: list[NDArray] = []

    for f_idx, freq_metrics in enumerate(all_metrics):
        # Collect cells with finite gridness, sorted descending.
        scored = [
            (cm.gridness, c_idx, cm.autocorr)
            for c_idx, cm in enumerate(freq_metrics)
            if np.isfinite(cm.gridness)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_cells_per_freq]

        if not top:
            gridness_by_freq.append(np.empty(0, dtype=float))
            autocorrs_by_freq.append(())
            cell_indices_by_freq.append(np.empty(0, dtype=int))
        else:
            gridness_by_freq.append(np.array([s[0] for s in top], dtype=float))
            autocorrs_by_freq.append(tuple(s[2] for s in top))
            cell_indices_by_freq.append(
                np.array([s[1] for s in top], dtype=int)
            )

    return MECAutocorrMosaicData(
        freq_indices=np.asarray(freq_idxs, dtype=int),
        gridness_by_freq=gridness_by_freq,
        autocorrs_by_freq=autocorrs_by_freq,
        cell_indices_by_freq=cell_indices_by_freq,
        extent=extent,
    )


def select_mec_grid_metrics(
    trace: TraceTree,
    ctx: FigureContext,
    *,
    max_examples: int = 4,
) -> MECGridMetricsData:
    """Compute gridness, spacing, and orientation for all MEC cells.

    Parameters
    ----------
    trace : TraceTree
        Trace containing MEC location-mean activation and location-id leaves.
    ctx : FigureContext
        Figure context specifying environment index.
    max_examples : int
        If > 0, select the top-*max_examples* cells by gridness across all
        frequency bands and attach their autocorrelograms to the returned
        ``MECGridMetricsData``.

    Returns
    -------
    MECGridMetricsData
        Per-frequency, per-cell gridness metrics, optionally with top-example
        autocorrelograms.
    """
    n_freq = trace.n_freq(TRACE_KEY_MEC_CELLS)
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idxs = [
        trace.validate_freq_idx(TRACE_KEY_MEC_CELLS, f) for f in range(n_freq)
    ]
    world = trace.get_world(env_idx)
    location_ids = trace.get(TRACE_KEY_LOCATION_IDS)[:, env_idx]

    cells_traces = [
        trace.get(f"{TRACE_KEY_MEC_CELLS}/{f}")[:, env_idx, :]
        for f in freq_idxs
    ]

    all_metrics, _geom = _compute_mec_grid_metrics(
        world,
        cells_traces,
        location_ids,
        smooth_sigma=spatial_rate_smooth_sigma(world),
    )

    if not all_metrics or _geom is None:
        first_cells = trace.get(f"{TRACE_KEY_MEC_CELLS}/0")[:, env_idx, :]
        n_cells = int(first_cells.shape[-1])
        return MECGridMetricsData(
            freq_indices=np.asarray(freq_idxs, dtype=int),
            cell_indices=np.arange(n_cells),
            gridness=np.full((n_freq, n_cells), np.nan, dtype=float),
            spacing=np.full((n_freq, n_cells), np.nan, dtype=float),
            orientation_deg=np.full((n_freq, n_cells), np.nan, dtype=float),
            valid_pixel_count=np.zeros((n_freq, n_cells), dtype=int),
            inner_radius=0.0,
            outer_radius=0.0,
            mask_strategy="fractional",
        )

    n_cells = len(all_metrics[0])
    cell_indices = np.arange(n_cells)
    inner_radius = 0.0
    outer_radius = 0.0
    mask_strategy = "fractional"

    gridness = np.full((n_freq, n_cells), np.nan, dtype=float)
    spacing = np.full((n_freq, n_cells), np.nan, dtype=float)
    orientation_deg = np.full((n_freq, n_cells), np.nan, dtype=float)
    valid_pixel_count = np.zeros((n_freq, n_cells), dtype=int)

    for f_idx, freq_metrics in enumerate(all_metrics):
        for c_idx, cm in enumerate(freq_metrics):
            gridness[f_idx, c_idx] = cm.gridness
            spacing[f_idx, c_idx] = cm.spacing
            orientation_deg[f_idx, c_idx] = cm.orientation_deg
            valid_pixel_count[f_idx, c_idx] = cm.valid_pixel_count
            if (
                not np.isnan(cm.gridness)
                and inner_radius == 0.0
                and _geom is not None
            ):
                g_result = compute_gridness(cm.autocorr, geometry=_geom)
                inner_radius = g_result.inner_radius
                outer_radius = g_result.outer_radius
                mask_strategy = g_result.mask_strategy

    # ── Select top-N examples across all frequencies ────────────────────────
    top_autocorrs: tuple[NDArray[np.floating], ...] = ()
    top_freq_idx = np.empty(0, dtype=int)
    top_cell_idx = np.empty(0, dtype=int)
    top_gridness = np.empty(0, dtype=float)

    if max_examples > 0:
        candidates: list[tuple[int, int, float, NDArray]] = []
        for f_idx, freq_metrics in enumerate(all_metrics):
            for c_idx, cm in enumerate(freq_metrics):
                if np.isfinite(cm.gridness):
                    candidates.append(
                        (freq_idxs[f_idx], c_idx, cm.gridness, cm.autocorr)
                    )
        candidates.sort(key=lambda x: x[2], reverse=True)
        selected = candidates[:max_examples]
        if selected:
            top_autocorrs = tuple(s[3] for s in selected)
            top_freq_idx = np.array([s[0] for s in selected], dtype=int)
            top_cell_idx = np.array([s[1] for s in selected], dtype=int)
            top_gridness = np.array([s[2] for s in selected], dtype=float)

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
        top_autocorrs=top_autocorrs,
        top_freq_indices=top_freq_idx,
        top_cell_indices=top_cell_idx,
        top_gridness=top_gridness,
    )


def select_mec_summary(
    trace: TraceTree, ctx: FigureContext
) -> MECSummaryFigureData:
    n_freq = trace.n_freq(TRACE_KEY_MEC_CELLS)
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idxs = [
        trace.validate_freq_idx(TRACE_KEY_MEC_CELLS, f) for f in range(n_freq)
    ]
    world = trace.get_world(env_idx)
    location_ids = trace.get(TRACE_KEY_LOCATION_IDS)[:, env_idx]
    cells = [
        trace.get(f"{TRACE_KEY_MEC_CELLS}/{f}")[:, env_idx, :]
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
    return MECSummaryFigureData(
        n_freq=n_freq,
        env_idx=env_idx,
        freq_idxs=freq_idxs,
        world=world,
        location_ids=location_ids,
        cells=cells,
        prepared_rate_maps=rate_maps,
    )


def select_mec_cell(trace: TraceTree, ctx: FigureContext) -> MECCellFigureData:
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idx = trace.validate_freq_idx(TRACE_KEY_MEC_CELLS, ctx.freq_idx)
    world = trace.get_world(env_idx)
    location_ids = trace.get(TRACE_KEY_LOCATION_IDS)[:, env_idx]
    cells = trace.get(f"{TRACE_KEY_MEC_CELLS}/{freq_idx}")[:, env_idx, :]
    rate_maps = prepare_rate_maps(
        world,
        cells,
        location_ids,
        smooth_sigma=spatial_rate_smooth_sigma(world),
    )
    return MECCellFigureData(
        env_idx=env_idx,
        freq_idx=freq_idx,
        world=world,
        location_ids=location_ids,
        cells=cells,
        prepared_rate_maps=rate_maps,
    )
