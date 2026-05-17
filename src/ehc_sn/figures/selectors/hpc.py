"""Selectors for HPC figure templates."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

from ehc_sn.figures._contracts import AnyWorld, PreparedRateMap
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.spatial import prepare_rate_maps, spatial_rate_smooth_sigma
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


def select_hpc_summary(trace: TraceTree, ctx: FigureContext) -> HPCSummaryFigureData:
    n_freq = trace.n_freq(TRACE_KEY_HPC_CELLS)
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idxs = [trace.validate_freq_idx(TRACE_KEY_HPC_CELLS, f) for f in range(n_freq)]
    world = trace.get_world(env_idx)
    location_ids = trace.get(TRACE_KEY_LOCATION_IDS)[:, env_idx]
    cells = [trace.get(f"{TRACE_KEY_HPC_CELLS}/{f}")[:, env_idx, :] for f in range(n_freq)]
    rate_maps = [
        prepare_rate_maps(world, c, location_ids, smooth_sigma=spatial_rate_smooth_sigma(world))
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
    rate_maps = prepare_rate_maps(world, cells, location_ids, smooth_sigma=spatial_rate_smooth_sigma(world))
    return HPCCellFigureData(
        env_idx=env_idx,
        freq_idx=freq_idx,
        world=world,
        location_ids=location_ids,
        cells=cells,
        prepared_rate_maps=rate_maps,
    )
