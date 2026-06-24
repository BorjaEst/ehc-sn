"""SeqMaze sample inspection — human-oriented decoded view of a corpus sample.

Produces :class:`SeqMazeSampleInspection` from raw persisted arrays.
Pattern mirrors ``goaltrace/inspection.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


# =============================================================================
@dataclass(frozen=True)
class SeqMazeSampleInspection:
    """Human-oriented decoded view of one seqmaze corpus sample.

    Attributes:
        sample_id: Human-readable sample identifier.
        split: Dataset split label.
        index: Sample index within the split.
        n_max: Maximum candidate node count.
        n_actual: Actual (non-padded) node count.
        t_max: Maximum path length.
        path_length: Length of the optimal path (hops).
        start_obs_id: Observation ID of the start node.
        goal_obs_id: Observation ID of the goal node.
        start_candidate_index: Candidate index of the start node.
        goal_candidate_index: Candidate index of the goal node.
        target_path: List of candidate indices on the optimal path.
        warnings: Non-fatal anomalies.
    """

    sample_id: str
    split: str
    index: int

    n_max: int
    n_actual: int
    t_max: int
    path_length: int
    start_obs_id: int
    goal_obs_id: int
    start_candidate_index: int
    goal_candidate_index: int
    target_path: list[int]

    warnings: list[str] = field(default_factory=list)


# =============================================================================
def prepare_sample_inspection(
    sample: dict[str, np.ndarray],
    split: str = "",
    index: int = -1,
    *,
    n_max: int = 45,
    t_max: int = 16,
) -> SeqMazeSampleInspection:
    """Build a :class:`SeqMazeSampleInspection` from raw persisted arrays.

    Args:
        sample: Channel dict for one sample.
        split: Split label.
        index: Sample index within the split.
        n_max: Maximum candidate node count (from manifest).
        t_max: Maximum path length (from manifest).

    Returns:
        Filled :class:`SeqMazeSampleInspection`.

    Raises:
        KeyError: When a required channel is missing.
    """
    required = (
        "node_obs_id",
        "node_candidate_index",
        "node_start_flag",
        "node_goal_flag",
        "node_mask",
        "target_path",
        "path_length",
    )
    for k in required:
        if k not in sample:
            raise KeyError(
                f"prepare_sample_inspection: '{k}' missing from sample."
            )

    node_obs_id: NDArray[np.int32] = np.asarray(sample["node_obs_id"])
    node_candidate_index: NDArray[np.int32] = np.asarray(
        sample["node_candidate_index"]
    )
    node_start_flag: NDArray[np.bool_] = np.asarray(sample["node_start_flag"])
    node_goal_flag: NDArray[np.bool_] = np.asarray(sample["node_goal_flag"])
    node_mask: NDArray[np.bool_] = np.asarray(sample["node_mask"])
    target_path: NDArray[np.int32] = np.asarray(sample["target_path"])
    path_length_arr: NDArray[np.int32] = np.asarray(sample["path_length"])

    n_actual = int(node_mask.sum())
    path_length = int(path_length_arr.flat[0])

    start_idx = int(np.argmax(node_start_flag))
    goal_idx = int(np.argmax(node_goal_flag))
    start_obs = int(node_obs_id[start_idx])
    goal_obs = int(node_obs_id[goal_idx])
    start_candidate = int(node_candidate_index[start_idx])
    goal_candidate = int(node_candidate_index[goal_idx])

    # Decode target path (EOS = n_max, PAD = n_max + 1)
    decoded_path: list[int] = []
    for token in target_path:
        if token >= n_max:
            break
        decoded_path.append(int(token))

    warnings: list[str] = []
    if n_actual < 2:
        warnings.append(f"n_actual={n_actual} is too small for a valid graph.")
    if path_length < 1:
        warnings.append(f"path_length={path_length} is suspicious.")

    sample_id = f"{split}/{index}"

    return SeqMazeSampleInspection(
        sample_id=sample_id,
        split=split,
        index=index,
        n_max=n_max,
        n_actual=n_actual,
        t_max=t_max,
        path_length=path_length,
        start_obs_id=start_obs,
        goal_obs_id=goal_obs,
        start_candidate_index=start_candidate,
        goal_candidate_index=goal_candidate,
        target_path=decoded_path,
        warnings=warnings,
    )


__all__ = [
    "SeqMazeSampleInspection",
    "prepare_sample_inspection",
]
