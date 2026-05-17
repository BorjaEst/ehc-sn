"""Selectors for LEC figure templates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.trace_tree import TraceTree

# ── Canonical trace / meta path constants ────────────────────────────────────
TRACE_KEY_OBSERVATION = "world_step/observation"
TRACE_KEY_LEC_CELLS = "diagnostic/lec/cells"
TRACE_KEY_LEC_FILTERED = "diagnostic/lec/filtered"
META_KEY_LEC_ALPHA = "lec/filter/alpha_sigmoid"
META_KEY_LEC_WF = "lec/w_f_sigmoid"


def _to_cpu(value: object) -> object:
    """Move a tensor to CPU if needed; return other values unchanged."""
    return value.cpu() if hasattr(value, "cpu") else value  # type: ignore[union-attr]


@dataclass
class LECSummaryFigureData:
    n_freq: int
    env_idx: int
    freq_idxs: list[int]
    obs_values: NDArray
    cells: list[NDArray]
    mean_activity: NDArray
    peak_activity: NDArray
    alpha: NDArray
    w_f: NDArray


@dataclass
class LECPipelineFigureData:
    env_idx: int
    freq_idx: int
    observations: NDArray
    cell_series: NDArray
    filtered_series: NDArray


def select_lec_summary(trace: TraceTree, ctx: FigureContext) -> LECSummaryFigureData:
    n_freq = trace.n_freq(TRACE_KEY_LEC_CELLS)
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idxs = [trace.validate_freq_idx(TRACE_KEY_LEC_CELLS, f) for f in range(n_freq)]
    obs_values = trace.get(TRACE_KEY_OBSERVATION)[:, env_idx]
    cells = [trace.get(f"{TRACE_KEY_LEC_CELLS}/{f}")[:, env_idx, :] for f in range(n_freq)]
    mean_activity = np.asarray([float(np.mean(c)) for c in cells])
    peak_activity = np.asarray([float(np.max(c)) for c in cells])
    alpha = _require_param_vector(trace, META_KEY_LEC_ALPHA)
    w_f = _require_param_vector(trace, META_KEY_LEC_WF)
    return LECSummaryFigureData(
        n_freq=n_freq,
        env_idx=env_idx,
        freq_idxs=freq_idxs,
        obs_values=obs_values,
        cells=cells,
        mean_activity=mean_activity,
        peak_activity=peak_activity,
        alpha=alpha,
        w_f=w_f,
    )


def select_lec_pipeline(trace: TraceTree, ctx: FigureContext) -> LECPipelineFigureData:
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idx = trace.validate_freq_idx(TRACE_KEY_LEC_CELLS, ctx.freq_idx)
    return LECPipelineFigureData(
        env_idx=env_idx,
        freq_idx=freq_idx,
        observations=trace.get(TRACE_KEY_OBSERVATION)[:, env_idx],
        cell_series=trace.get(f"{TRACE_KEY_LEC_CELLS}/{freq_idx}")[:, env_idx, :],
        filtered_series=trace.get(f"{TRACE_KEY_LEC_FILTERED}/{freq_idx}")[:, env_idx, :],
    )


def _require_param_vector(trace: TraceTree, key: str) -> NDArray:
    value = _to_cpu(trace.get_meta_path(key))
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"Trace metadata '{key}' must be one-dimensional, got shape {vector.shape}")
    return vector
