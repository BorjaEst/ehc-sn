"""Goaltrace sample inspection — human-oriented decoded view of a corpus sample.

Produces :class:`GoaltraceSampleInspection` from raw persisted arrays.
Pattern mirrors ``routebind/inspection.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


# =============================================================================
@dataclass(frozen=True)
class GoaltraceSampleInspection:
    """Human-oriented decoded view of one goaltrace corpus sample.

    Attributes:
        sample_id: Human-readable sample identifier.
        split: Dataset split label.
        index: Sample index within the split.
        n_valid: Number of valid (non-padded) nodes.
        n_padded: Number of padding slots.
        current_observation_id: Observation ID of the current node.
        goal_observation_id: Observation ID of the goal node.
        current_idx: Padded-space index of the current node.
        goal_idx: Padded-space index of the goal node.
        optimal_path: List of padded-space node indices on the optimal path
            (derived from target field).
        path_length: Number of edge hops in the optimal path.
        weight_stats: Dict with min/mean/max of edge weights on the path.
        decay_consistent: Whether target field follows gamma^d pattern.
        current_target: Target field value at the current node.
        goal_target: Target field value at the goal node.
        field_decay: Field decay factor used by the corpus.
        total_edges: Number of directed edges in the DAG.
        warnings: Non-fatal anomalies.
    """

    sample_id: str
    split: str
    index: int

    n_valid: int
    n_padded: int
    current_observation_id: int
    goal_observation_id: int
    current_idx: int
    goal_idx: int
    optimal_path: list[int]
    path_length: int

    weight_stats: dict[str, float]
    decay_consistent: bool
    current_target: float
    goal_target: float
    field_decay: float
    total_edges: int

    warnings: list[str] = field(default_factory=list)


# =============================================================================
def prepare_sample_inspection(
    sample: dict[str, np.ndarray],
    split: str = "",
    index: int = -1,
    *,
    field_decay: float = 0.8,
) -> GoaltraceSampleInspection:
    """Build a :class:`GoaltraceSampleInspection` from raw persisted arrays.

    Args:
        sample: Channel dict for one sample.
        split: Split label.
        index: Sample index within the split.
        field_decay: Field decay factor gamma.

    Returns:
        Filled :class:`GoaltraceSampleInspection`.

    Raises:
        KeyError: When a required channel is missing.
    """
    required = (
        "observation_id",
        "weight",
        "current_flag",
        "goal_flag",
        "node_mask",
        "target_field",
    )
    for k in required:
        if k not in sample:
            raise KeyError(
                f"prepare_sample_inspection: '{k}' missing from sample."
            )

    observation_id: NDArray[np.int32] = np.asarray(sample["observation_id"])
    weight: NDArray[np.float32] = np.asarray(sample["weight"])
    current_flag: NDArray[np.bool_] = np.asarray(sample["current_flag"])
    goal_flag: NDArray[np.bool_] = np.asarray(sample["goal_flag"])
    node_mask: NDArray[np.bool_] = np.asarray(sample["node_mask"])
    target_field: NDArray[np.float32] = np.asarray(sample["target_field"])

    n_valid = int(node_mask.sum())
    n_padded = len(node_mask) - n_valid
    current_idx = int(np.argmax(current_flag))
    goal_idx = int(np.argmax(goal_flag))

    current_obs = int(observation_id[current_idx])
    goal_obs = int(observation_id[goal_idx])

    # Extract optimal path from target field
    tf = target_field[:n_valid]
    on_path = np.where(tf > 0)[0]
    order = np.argsort(-tf[on_path])
    optimal_path = on_path[order].tolist() if len(on_path) > 0 else []
    path_length = max(len(optimal_path) - 1, 0)

    # Weight stats for edges on path
    edge_weights = []
    for node in optimal_path:
        if int(node) != current_idx or int(node) == goal_idx:
            pass
    actual_edge_weights = []
    for step in range(path_length):
        u = optimal_path[step]
        v = optimal_path[step + 1]
        actual_edge_weights.append(float(weight[u, v]))

    ws = {}
    if actual_edge_weights:
        ws = {
            "min": float(np.min(actual_edge_weights)),
            "mean": float(np.mean(actual_edge_weights)),
            "max": float(np.max(actual_edge_weights)),
        }

    # Decay consistency check
    decay_consistent = True
    if len(optimal_path) > 1:
        for d, node in enumerate(optimal_path):
            expected = float(field_decay**d)
            actual = float(tf[node])
            if abs(actual - expected) > 0.02:
                decay_consistent = False
                break

    current_target = float(target_field[current_idx])
    goal_target = float(target_field[goal_idx])

    # Estimate total edges from successor_indices if available
    total_edges = 0
    succ_idx = sample.get("successor_indices")
    succ_mask = sample.get("successor_mask")
    if succ_idx is not None and succ_mask is not None:
        for u in range(n_valid):
            total_edges += int(succ_mask[u].sum())

    warnings: list[str] = []
    if current_idx == goal_idx:
        warnings.append("current == goal: no path exists")
    if len(optimal_path) == 0:
        warnings.append("no nodes on optimal path (all-zero target field)")
    if not decay_consistent:
        warnings.append("target field does not follow gamma^d pattern")

    sample_id = f"{split}/{index}" if split and index >= 0 else "?"

    return GoaltraceSampleInspection(
        sample_id=sample_id,
        split=split,
        index=index,
        n_valid=n_valid,
        n_padded=n_padded,
        current_observation_id=current_obs,
        goal_observation_id=goal_obs,
        current_idx=current_idx,
        goal_idx=goal_idx,
        optimal_path=optimal_path,
        path_length=path_length,
        weight_stats=ws,
        decay_consistent=decay_consistent,
        current_target=current_target,
        goal_target=goal_target,
        field_decay=field_decay,
        total_edges=total_edges,
        warnings=warnings,
    )


__all__ = [
    "GoaltraceSampleInspection",
    "prepare_sample_inspection",
]
