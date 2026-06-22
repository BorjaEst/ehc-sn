"""Routebind sample inspection — human-oriented decoded view of a corpus sample.

Produces :class:`RoutebindSampleInspection` from raw persisted arrays.
Inspection is a read-only, descriptive operation.  It does not determine
correctness — that belongs in ``validation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ehc_sn.tasks.routebind.contracts import CELL_OBSERVATION, CELL_WALL
from ehc_sn.tasks.routebind.decoding import (
    extract_route_from_trajectory_field,
    extract_waypoint_sequence,
)


# =============================================================================
@dataclass(frozen=True)
class RoutebindSampleInspection:
    """Human-oriented decoded view of one routebind corpus sample.

    All positional and route fields are derived from the persisted target
    arrays via the canonical greedy decoder.  Non-fatal anomalies (e.g.
    minor decay deviations) are collected in *warnings* rather than
    producing errors.
    """

    sample_id: str
    split: str
    index: int

    # Raw input fields (read from corpus)
    cell_type: NDArray[np.int32]
    observation_id: NDArray[np.int32]

    # Decoded positional fields
    start_position: int
    goal_observation: int
    goal_positions: NDArray[np.int32]

    # Decoded route and waypoints
    physical_route: list[int]
    waypoint_events: list[tuple[int, int, float]]
    """Ordered waypoints as ``(position, observation_id, activation)``."""

    next_direction: int
    """First physical movement direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT),
    or -1 when the route has fewer than 2 positions."""
    next_observation: int
    """First post-start accepted observation ID, or -1."""

    # Summary statistics
    route_length: int
    semantic_length: int
    wall_density: float

    # Non-fatal anomalies collected during preparation
    warnings: list[str] = field(default_factory=list)


# =============================================================================
def prepare_sample_inspection(
    sample: dict[str, np.ndarray],
    split: str = "",
    index: int = -1,
    *,
    dag_adjacency: list[list[int]] | None = None,
    gamma_space: float = 0.9848,
    gamma_semantic: float = 0.8,
    grid_width: int | None = None,
) -> RoutebindSampleInspection:
    """Build a :class:`RoutebindSampleInspection` from raw persisted arrays.

    Args:
        sample: Channel dict for one sample (keys from
            :data:`ROUTEBIND_SCHEMA.all_channels`).
        split: Split label for the sample.
        index: Sample index within the split.
        dag_adjacency: Optional DAG adjacency list for semantic-edge
            consistency warnings.  ``dag_adjacency[o]`` lists valid
            successor public observation IDs of ``o``.
        gamma_space: Spatial decay factor (for decay-consistency warnings).
        gamma_semantic: Semantic decay factor (for decay-consistency warnings).

    Returns:
        Filled :class:`RoutebindSampleInspection`.

    Raises:
        KeyError: When a required channel is missing.
    """
    required = (
        "cell_type",
        "observation_id",
        "start_flag",
        "goal_flag",
        "target_trajectory",
        "target_waypoint",
        "target_next_dir",
        "target_next_obs",
    )
    for k in required:
        if k not in sample:
            raise KeyError(
                f"prepare_sample_inspection: '{k}' missing from sample."
            )

    ct: NDArray[np.int32] = np.asarray(sample["cell_type"])
    oid: NDArray[np.int32] = np.asarray(sample["observation_id"])
    sf: NDArray[np.bool_] = np.asarray(sample["start_flag"])
    gf: NDArray[np.bool_] = np.asarray(sample["goal_flag"])
    tf: NDArray[np.float32] = np.asarray(sample["target_trajectory"])
    wf: NDArray[np.float32] = np.asarray(sample["target_waypoint"])

    nd_raw = sample.get("target_next_dir", -1)
    no_raw = sample.get("target_next_obs", -1)
    nd: int = int(nd_raw.item()) if hasattr(nd_raw, "item") else int(nd_raw)
    no: int = int(no_raw.item()) if hasattr(no_raw, "item") else int(no_raw)

    n_slots = len(ct)
    S = n_slots

    warnings: list[str] = []

    # Start
    if int(sf.sum()) != 1:
        start_pos = -1
    else:
        start_pos = int(np.argmax(sf))

    # Goal
    goal_obs: int = -1
    goal_positions: NDArray[np.int32] = np.array([], dtype=np.int32)
    if int(gf.sum()) >= 1:
        goal_idx = int(np.argmax(gf))
        goal_obs = int(oid[goal_idx])
        goal_positions = np.where(gf)[0].astype(np.int32)

    # Decode route and waypoints
    physical_route: list[int] = []
    waypoint_events: list[tuple[int, int, float]] = []

    if start_pos >= 0 and int(sf.sum()) == 1:
        route = extract_route_from_trajectory_field(
            tf, start_pos, grid_width=grid_width
        )
        if route:
            physical_route = route
            waypoint_events = extract_waypoint_sequence(wf, route, oid)

    route_length = len(physical_route)
    semantic_length = len(waypoint_events)

    if start_pos >= 0 and route_length > 0 and float(tf[start_pos]) != 1.0:
        warnings.append(f"Start activation {float(tf[start_pos]):.4f} != 1.0")

    if route_length >= 2:
        width = int(np.sqrt(S))
        if grid_width is not None:
            width = grid_width
        sr, sc = divmod(start_pos, width)
        fs = physical_route[1]
        fr, fc = divmod(fs, width)
        exp_dir = -1
        dr, dc = fr - sr, fc - sc
        for dir_val, ddr, ddc in [
            (0, -1, 0),
            (1, 0, 1),
            (2, 1, 0),
            (3, 0, -1),
        ]:
            if (dr, dc) == (ddr, ddc):
                exp_dir = dir_val
                break
        if exp_dir >= 0 and nd != exp_dir:
            warnings.append(
                f"target_next_dir={nd} but first physical step direction={exp_dir}"
            )

    # DAG transition check
    if dag_adjacency is not None and waypoint_events:
        for i in range(len(waypoint_events) - 1):
            src_obs = waypoint_events[i][1]
            dst_obs = waypoint_events[i + 1][1]
            if src_obs >= 0 and dst_obs >= 0:
                if dst_obs not in dag_adjacency[src_obs]:
                    warnings.append(
                        f"DAG edge missing: {src_obs} -> {dst_obs} "
                        f"(waypoints {i} -> {i + 1})"
                    )

    wall_density = float((ct == CELL_WALL).mean())

    sample_id = f"{split}/{index}" if split and index >= 0 else "?"

    return RoutebindSampleInspection(
        sample_id=sample_id,
        split=split,
        index=index,
        cell_type=ct,
        observation_id=oid,
        start_position=start_pos,
        goal_observation=goal_obs,
        goal_positions=goal_positions,
        physical_route=physical_route,
        waypoint_events=waypoint_events,
        next_direction=nd,
        next_observation=no,
        route_length=route_length,
        semantic_length=semantic_length,
        wall_density=wall_density,
        warnings=warnings,
    )


__all__ = [
    "RoutebindSampleInspection",
    "prepare_sample_inspection",
]
