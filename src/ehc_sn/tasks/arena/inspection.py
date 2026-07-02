"""Arena sample inspection — human-oriented decoded view of a corpus sample.

Produces :class:`ArenaSampleInspection` from raw persisted arrays.
Pattern mirrors ``routebind/inspection.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


# =============================================================================
@dataclass(frozen=True)
class ArenaSampleInspection:
    """Human-oriented decoded view of one arena corpus sample (episode).

    Attributes:
        sample_id: Human-readable sample identifier.
        split: Dataset split label.
        index: Sample index within the split.
        episode_length: Number of valid trajectory steps.
        revisit_count: Number of revisit steps.
        revisit_rate: Fraction of steps that are revisits.
        observation_sequence: Observation IDs per step (valid steps only).
        coordinate_sequence: (row, col) tuples per step.
        start_position: (row, col) of the first step.
        wall_density: Fraction of wall cells in the topology.
        grid_shape: (H, W) of the topology.
        warnings: Non-fatal anomalies.
    """

    sample_id: str
    split: str
    index: int

    episode_length: int
    revisit_count: int
    revisit_rate: float
    observation_sequence: list[int]
    coordinate_sequence: list[tuple[int, int]]
    start_position: tuple[int, int]
    wall_density: float
    grid_shape: tuple[int, int]

    warnings: list[str] = field(default_factory=list)


# =============================================================================
def prepare_sample_inspection(
    sample: dict[str, np.ndarray],
    split: str = "",
    index: int = -1,
) -> ArenaSampleInspection:
    """Build a :class:`ArenaSampleInspection` from raw persisted arrays.

    Args:
        sample: Channel dict for one sample.
        split: Split label.
        index: Sample index.

    Returns:
        Filled :class:`ArenaSampleInspection`.

    Raises:
        KeyError: When a required channel is missing.
    """
    required = (
        "trajectory_row",
        "trajectory_col",
        "trajectory_observation_id",
        "trajectory_length",
        "trajectory_is_revisit",
        "trajectory_valid_step",
        "topology",
    )
    for k in required:
        if k not in sample:
            raise KeyError(
                f"prepare_sample_inspection: '{k}' missing from sample."
            )

    traj_len = int(np.asarray(sample["trajectory_length"]).flat[0])
    rows = np.asarray(sample["trajectory_row"])[:traj_len]
    cols = np.asarray(sample["trajectory_col"])[:traj_len]
    obs_ids = np.asarray(sample["trajectory_observation_id"])[:traj_len]
    revisit = np.asarray(sample["trajectory_is_revisit"])[:traj_len]
    topology = np.asarray(sample["topology"])

    H, W = topology.shape
    coords = list(zip(rows.tolist(), cols.tolist()))
    obs_seq = [int(o) for o in obs_ids.tolist()]
    revisit_count = int(revisit.sum())
    revisit_rate = revisit_count / max(traj_len, 1)
    start_pos = (int(rows[0]), int(cols[0]))
    wall_density = float((~topology).sum() / topology.size)

    warnings: list[str] = []
    if traj_len == 0:
        warnings.append("empty trajectory (length 0)")

    sample_id = f"{split}/{index}" if split and index >= 0 else "?"

    return ArenaSampleInspection(
        sample_id=sample_id,
        split=split,
        index=index,
        episode_length=traj_len,
        revisit_count=revisit_count,
        revisit_rate=revisit_rate,
        observation_sequence=obs_seq,
        coordinate_sequence=coords,
        start_position=start_pos,
        wall_density=wall_density,
        grid_shape=(H, W),
        warnings=warnings,
    )


# =============================================================================
# Pathway metric prefixes — used by report-data derived resources
# =============================================================================

ARENA_TEM_PATHWAY_PREFIXES: dict[str, str] = {
    "ancestral": "accuracy_path",
    "retrieved": "accuracy_recall",
    "inference": "accuracy_post",
}
"""Map from TEM prediction pathway name to metric name prefix."""


def build_pathway_metrics_dataframe(
    metrics: dict[str, float],
) -> "pd.DataFrame":
    """Build a per-pathway metrics table from an Arena TEM regime metrics dict.

    Returns a DataFrame with columns ``pathway``, ``all_steps``,
    ``revisit_steps``.

    Args:
        metrics: Flat ``{metric_name: value}`` dict from a regime artifact.

    Returns:
        DataFrame with one row per prediction pathway.
    """
    import pandas as pd

    rows: list[dict[str, object]] = []
    for path_name, prefix in ARENA_TEM_PATHWAY_PREFIXES.items():
        row: dict[str, object] = {"pathway": path_name}
        all_key = f"{prefix}_all"
        revisit_key = f"{prefix}_revisit"
        if all_key in metrics:
            row["all_steps"] = metrics[all_key]
        if revisit_key in metrics:
            row["revisit_steps"] = metrics[revisit_key]
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "ArenaSampleInspection",
    "prepare_sample_inspection",
]
