"""Selectors for LEC figure templates.

Scientific framing
------------------
LEC is analysed as the sensory/content stream (``x``), not as a spatial
representation.  Figures test whether LEC activity is organised by
observation identity and whether it can be dissociated from MEC structural
coding and HPC conjunctive coding.

References
----------
Whittington et al. (2020). "The Tolman-Eichenbaum Machine".
    Cell 183(5):1248–1262 e23.  https://doi.org/10.1016/j.cell.2020.10.024
Whittington et al. (2022). "Relating transformers to models and neural
    representations of the hippocampal formation".
    arXiv:2112.04035.  https://arxiv.org/abs/2112.04035
Hargreaves et al. (2005). "Major dissociation between medial and lateral
    entorhinal input to dorsal hippocampus".  Science 308:1792–1794.
Deshmukh & Knierim (2011). "Representation of non-spatial and spatial
    information in the lateral entorhinal cortex".  Front Behav Neurosci 5:69.
Tsao, Moser & Moser (2013). "Traces of experience in the lateral entorhinal
    cortex".  Curr Biol 23:399–405.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ehc_sn.analysis.tem_representations import (
    ContentStructureRSAResult,
    StreamEffect,
    compute_content_structure_rsa,
    compute_observation_tuning,
    compute_selectivity_scores,
    compute_tuning_entropy,
)
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.arena_tem import (
    TEM_META_KEY_TARGET_OBS_ID,
)
from ehc_sn.traces.trace_tree import TraceTree
from ehc_sn.utils import to_cpu

# ── Canonical trace / meta path constants ────────────────────────────────────
LEC_TRACE_KEY_OBSERVATION = "world_step/observation"
LEC_TRACE_KEY_CELLS = "diagnostic/lec/cells"
LEC_TRACE_KEY_FILTERED = "diagnostic/lec/filtered"
LEC_TRACE_KEY_LOCATION_IDS = "world_step/location_ids"
LEC_META_KEY_ALPHA = "lec/filter/alpha_sigmoid"
LEC_META_KEY_WF = "lec/w_f_sigmoid"


# =============================================================================
# Shared helpers
# =============================================================================


def _to_cpu(value: object) -> object:
    """Move a tensor to CPU if needed; return other values unchanged."""
    return value.cpu() if hasattr(value, "cpu") else value


def _require_param_vector(trace: TraceTree, key: str) -> NDArray:
    """Fetch and validate a 1-D metadata vector from the trace.

    Returns an empty array if the metadata key is missing or ``None``.
    """
    try:
        value = _to_cpu(trace.get_meta_path(key))
    except (KeyError, FileNotFoundError, ValueError):
        return np.empty(0, dtype=float)
        return np.empty(0, dtype=float)
    if value is None:
        return np.empty(0, dtype=float)
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(
            f"Trace metadata '{key}' must be one-dimensional, "
            f"got shape {vector.shape}"
        )
    return vector


def _get_observation_ids(trace: TraceTree, env_idx: int) -> NDArray:
    """Return ground-truth observation IDs for one environment.

    Prefers target/observation_id metadata (whole trajectory), falls back
    to world_step/observation trace (per-step).
    """
    try:
        gt_raw = np.asarray(
            to_cpu(trace.get_meta_path(TEM_META_KEY_TARGET_OBS_ID))
        )
        # Shape: (B, T_max) -> select batch 0.
        if gt_raw.ndim == 2:
            return gt_raw[0]
    except (KeyError, FileNotFoundError):
        pass
    # Fallback: per-step trace.
    obs = trace.get(LEC_TRACE_KEY_OBSERVATION)
    if obs is not None:
        arr = np.asarray(to_cpu(obs))
        if arr.ndim >= 1:
            return arr[:, env_idx] if arr.ndim >= 2 else arr
    raise ValueError(
        "Could not resolve observation IDs from either "
        f"{TEM_META_KEY_TARGET_OBS_ID} or {LEC_TRACE_KEY_OBSERVATION}"
    )


def _get_location_ids(trace: TraceTree, env_idx: int) -> NDArray:
    """Return location IDs for one environment."""
    loc = trace.get(LEC_TRACE_KEY_LOCATION_IDS)
    if loc is None:
        raise ValueError(f"Trace key {LEC_TRACE_KEY_LOCATION_IDS} not found")
    arr = np.asarray(to_cpu(loc))
    return arr[:, env_idx] if arr.ndim >= 2 else arr


# =============================================================================
# 1. LEC activity trajectory
# =============================================================================


@dataclass
class LECActivityTrajectoryData:
    """Prepared data for the LEC activity-trajectory overview figure."""

    env_idx: int
    obs_ids: NDArray  # (T,) int
    location_ids: NDArray  # (T,) int
    cells: NDArray  # (T, n_units) float
    filtered: NDArray | None  # (T, n_units) float or None if unavailable


def select_lec_activity_trajectory(
    trace: TraceTree, ctx: FigureContext
) -> LECActivityTrajectoryData:
    """Extract LEC activity-trajectory data for one environment.

    Parameters
    ----------
    trace : TraceTree
        Must contain ``diagnostic/lec/cells`` and optionally
        ``diagnostic/lec/filtered``.
    ctx : FigureContext
        Figure context specifying environment index; uses freq 0.

    Returns
    -------
    LECActivityTrajectoryData
    """
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idx = trace.validate_freq_idx(LEC_TRACE_KEY_CELLS, 0)
    obs_ids = _get_observation_ids(trace, env_idx)
    location_ids = _get_location_ids(trace, env_idx)
    cells = np.asarray(
        to_cpu(trace.get(f"{LEC_TRACE_KEY_CELLS}/{freq_idx}")[:, env_idx, :])
    )
    # Try filtered; silently return None if missing.
    try:
        filtered = np.asarray(
            to_cpu(
                trace.get(f"{LEC_TRACE_KEY_FILTERED}/{freq_idx}")[:, env_idx, :]
            )
        )
    except (KeyError, FileNotFoundError):
        filtered = None

    return LECActivityTrajectoryData(
        env_idx=env_idx,
        obs_ids=obs_ids,
        location_ids=location_ids,
        cells=cells,
        filtered=filtered,
    )


# =============================================================================
# 2. LEC observation tuning
# =============================================================================


@dataclass
class LECObservationTuningData:
    """Prepared data for the LEC observation-ID tuning figure."""

    env_idx: int
    n_units: int
    n_observations: int
    tuning_matrix: NDArray  # (n_units, n_observations) float
    selectivity: NDArray  # (n_units,) float
    tuning_entropy: NDArray  # (n_units,) float
    fraction_selective: float


def select_lec_observation_tuning(
    trace: TraceTree, ctx: FigureContext
) -> LECObservationTuningData:
    """Compute observation-ID tuning for LEC units at frequency 0.

    Parameters
    ----------
    trace : TraceTree
        Must contain ``diagnostic/lec/cells``.
    ctx : FigureContext
        Figure context specifying environment index; uses freq 0.

    Returns
    -------
    LECObservationTuningData
    """
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idx = trace.validate_freq_idx(LEC_TRACE_KEY_CELLS, 0)
    obs_ids = _get_observation_ids(trace, env_idx)
    cells = np.asarray(
        to_cpu(trace.get(f"{LEC_TRACE_KEY_CELLS}/{freq_idx}")[:, env_idx, :])
    )
    trace_min, trace_max = int(obs_ids.min()), int(obs_ids.max())
    n_obs = trace_max - trace_min + 1
    n_units = cells.shape[1]

    tuning = compute_observation_tuning(cells, obs_ids, n_obs)
    si = compute_selectivity_scores(tuning)
    entropy = compute_tuning_entropy(tuning)

    # Fraction selective: selectivity > 0.3.
    fraction_selective = float(
        np.sum(np.isfinite(si) & (si > 0.3)) / max(n_units, 1)
    )

    return LECObservationTuningData(
        env_idx=env_idx,
        n_units=n_units,
        n_observations=n_obs,
        tuning_matrix=tuning,
        selectivity=si,
        tuning_entropy=entropy,
        fraction_selective=fraction_selective,
    )


# =============================================================================
# 3. LEC / MEC / HPC representational similarity (RSA)
# =============================================================================


def select_lec_content_structure_rsa(
    trace: TraceTree, ctx: FigureContext
) -> ContentStructureRSAResult:
    """Compare LEC, MEC, and HPC representational geometry.

    Parameters
    ----------
    trace : TraceTree
        Must contain diagnostic traces for LEC, MEC, and HPC, plus
        observation and location IDs.
    ctx : FigureContext
        Figure context specifying environment index; uses freq 0 for all
        systems.

    Returns
    -------
    ContentStructureRSAResult
    """
    env_idx = trace.validate_env_idx(ctx.env_idx)
    obs_ids = _get_observation_ids(trace, env_idx)
    location_ids = _get_location_ids(trace, env_idx)

    # --- LEC ---
    freq_0 = trace.validate_freq_idx(LEC_TRACE_KEY_CELLS, 0)
    lec_act = np.asarray(
        to_cpu(trace.get(f"{LEC_TRACE_KEY_CELLS}/{freq_0}")[:, env_idx, :])
    )

    # --- MEC ---
    mec_freq_0 = trace.validate_freq_idx(LEC_TRACE_KEY_CELLS, 0)
    mec_act = np.asarray(
        to_cpu(trace.get(f"{LEC_TRACE_KEY_CELLS}/{mec_freq_0}")[:, env_idx, :])
    )

    # --- HPC ---
    hpc_freq_0 = trace.validate_freq_idx(LEC_TRACE_KEY_CELLS, 0)
    hpc_act = np.asarray(
        to_cpu(trace.get(f"{LEC_TRACE_KEY_CELLS}/{hpc_freq_0}")[:, env_idx, :])
    )

    return compute_content_structure_rsa(
        lec_act, mec_act, hpc_act, obs_ids, location_ids
    )


# =============================================================================
# 4. LEC content filtering diagnostic
# =============================================================================


@dataclass
class LECContentFilteringFigureData:
    """Prepared data for the LEC content filtering diagnostic figure."""

    env_idx: int
    n_freq: int
    freq_idxs: list[int]
    cells_by_freq: list[NDArray]  # per-frequency (T, n_units) float arrays
    filtered_by_freq: list[NDArray]  # per-frequency (T, n_units) or empty
    alpha: NDArray  # (n_freq,) or empty
    w_f: NDArray  # (n_freq,) or empty


def select_lec_content_filtering(
    trace: TraceTree, ctx: FigureContext
) -> LECContentFilteringFigureData:
    """Extract per-frequency LEC content-state and filtered-state data.

    Parameters
    ----------
    trace : TraceTree
        Must contain ``diagnostic/lec/cells`` and optionally
        ``diagnostic/lec/filtered``.
    ctx : FigureContext
        Figure context specifying environment index.

    Returns
    -------
    LECContentFilteringFigureData
    """
    n_freq = trace.n_freq(LEC_TRACE_KEY_CELLS)
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idxs = [
        trace.validate_freq_idx(LEC_TRACE_KEY_CELLS, f) for f in range(n_freq)
    ]

    cells_by_freq = []
    filtered_by_freq = []
    for f in freq_idxs:
        cells_by_freq.append(
            np.asarray(
                to_cpu(trace.get(f"{LEC_TRACE_KEY_CELLS}/{f}")[:, env_idx, :])
            )
        )
        try:
            filtered_by_freq.append(
                np.asarray(
                    to_cpu(
                        trace.get(f"{LEC_TRACE_KEY_FILTERED}/{f}")[
                            :, env_idx, :
                        ]
                    )
                )
            )
        except (KeyError, FileNotFoundError):
            filtered_by_freq.append(np.empty((0, 0), dtype=float))

    alpha = _require_param_vector(trace, LEC_META_KEY_ALPHA)
    w_f = _require_param_vector(trace, LEC_META_KEY_WF)

    return LECContentFilteringFigureData(
        env_idx=env_idx,
        n_freq=n_freq,
        freq_idxs=freq_idxs,
        cells_by_freq=cells_by_freq,
        filtered_by_freq=filtered_by_freq,
        alpha=alpha,
        w_f=w_f,
    )


# =============================================================================
# Existing (kept for backward compatibility)
# =============================================================================


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


def select_lec_summary(
    trace: TraceTree, ctx: FigureContext
) -> LECSummaryFigureData:
    n_freq = trace.n_freq(LEC_TRACE_KEY_CELLS)
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idxs = [
        trace.validate_freq_idx(LEC_TRACE_KEY_CELLS, f) for f in range(n_freq)
    ]
    obs_values = trace.get(LEC_TRACE_KEY_OBSERVATION)[:, env_idx]
    cells = [
        trace.get(f"{LEC_TRACE_KEY_CELLS}/{f}")[:, env_idx, :]
        for f in range(n_freq)
    ]
    mean_activity = np.asarray([float(np.mean(c)) for c in cells])
    peak_activity = np.asarray([float(np.max(c)) for c in cells])
    alpha = _require_param_vector(trace, LEC_META_KEY_ALPHA)
    w_f = _require_param_vector(trace, LEC_META_KEY_WF)
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


def select_lec_pipeline(
    trace: TraceTree, ctx: FigureContext
) -> LECPipelineFigureData:
    env_idx = trace.validate_env_idx(ctx.env_idx)
    freq_idx = trace.validate_freq_idx(LEC_TRACE_KEY_CELLS, ctx.freq_idx)
    return LECPipelineFigureData(
        env_idx=env_idx,
        freq_idx=freq_idx,
        observations=trace.get(LEC_TRACE_KEY_OBSERVATION)[:, env_idx],
        cell_series=trace.get(f"{LEC_TRACE_KEY_CELLS}/{freq_idx}")[
            :, env_idx, :
        ],
        filtered_series=trace.get(f"{LEC_TRACE_KEY_FILTERED}/{freq_idx}")[
            :, env_idx, :
        ],
    )
