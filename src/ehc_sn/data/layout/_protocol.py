"""Layout record protocol for arena trajectory builders.

Defines :class:`SpatialLayout`, the graph-indexed world record that any layout
source (dungeongen, openfield, etc.) produces and the arena trajectory builder
consumes.  The arena task remains topology-agnostic; all topology-specific
complexity lives in the layout source.

See ``spec/spec-openfield-layout.md`` for the full design.
"""

from __future__ import annotations

from typing import Final, NotRequired, TypedDict

import numpy as np

# =============================================================================
# Action-space descriptor
# =============================================================================


class ActionSpace(TypedDict):
    """Action-space descriptor for a spatial layout.

    Declares the number of actions, their names, and the row/col deltas
    that implement them.  The ``stay_action`` field is the action id of
    the no-op action, or ``None`` if the layout has no stay action.
    """

    name: str
    """Canonical name for this action space (e.g. ``"grid4_dir"``, ``"hex6_dir"``)."""
    action_count: int
    """Number of discrete actions."""
    stay_action: int | None
    """Action id of the no-op action, or ``None``."""
    action_names: list[str]
    """Human-readable names for each action (length ``action_count``)."""
    action_deltas: list[tuple[int, int]]
    """(row, col) deltas for each action (length ``action_count``)."""


# =============================================================================
# Core layout record
# =============================================================================


class SpatialLayout(TypedDict):
    """Graph-indexed spatial world record.

    All arrays use the graph state index as their first axis.  Valid walkable
    states are those where ``valid_state_mask`` is True.

    Fields (in declaration order):
    """

    layout_id: str
    """Unique identifier for this layout instance."""
    layout_family: str
    """Layout source family (e.g. ``"openfield"``, ``"dungeongen"``)."""
    topology_type: str
    """Topology kind (e.g. ``"square"``, ``"hex"``, ``"grid2d"``)."""

    graph_state_count: int
    """Number of nodes in the graph index space (may include pruned states)."""
    valid_state_mask: np.ndarray
    """Boolean ``(graph_state_count,)`` — True for walkable states."""

    state_to_row_col: np.ndarray
    """``(graph_state_count, 2)`` int32 — row, col for each graph state."""
    observation_id: np.ndarray
    """``(graph_state_count,)`` int32 — sensory observation id per graph state."""

    adjacency: np.ndarray
    """``(graph_state_count, graph_state_count)`` bool — undirected connectivity."""

    action_space: ActionSpace
    """Action-space descriptor for this layout."""

    transition_matrix: np.ndarray
    """``(graph_state_count, graph_state_count)`` float64 — row-stochastic transition probs."""

    # Provenance
    topology_seed: int
    """Seed used to generate the topology (width, shape)."""
    sensory_seed: int
    """Seed used for the observation_id random assignment."""
    sensory_vocab_size: int
    """Number of distinct observation ids (equivalent to legacy TEM ``s_size``)."""
    split: NotRequired[str]
    """Split assignment (e.g. ``"train"``, ``"val"``, ``"test"``)."""


# =============================================================================
# Default action spaces
# =============================================================================

DEFAULT_GRID_ACTION_SPACE: Final[ActionSpace] = {
    "name": "grid4_dir",
    "action_count": 5,
    "stay_action": 0,
    "action_names": ["STAY", "UP", "RIGHT", "DOWN", "LEFT"],
    "action_deltas": [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)],
}
"""Default 4-direction + stay action space for square/dungeon grids."""

HEX_ACTION_SPACE: Final[ActionSpace] = {
    "name": "hex6_dir",
    "action_count": 7,
    "stay_action": 0,
    "action_names": [
        "STAY",
        "DOWN_LEFT",
        "DOWN_RIGHT",
        "UP_LEFT",
        "UP_RIGHT",
        "LEFT",
        "RIGHT",
    ],
    "action_deltas": [
        (0, 0),
        (0, -1),
        (0, 1),
        (0, -1),
        (0, 1),
        (0, -1),
        (0, 1),
    ],
}
"""Hex 6-direction + stay action space.

.. note::
    The delta values here are placeholders; hex movement requires
    offset-aware row/col deltas that depend on the parity of the
    current row.  The actual mapping is computed by the hex layout
    generator and encoded in the adjacency matrix.
"""


# =============================================================================
# Validator
# =============================================================================


def validate_spatial_layout(layout: SpatialLayout) -> None:
    """Validate a :class:`SpatialLayout` record against the protocol contract.

    Args:
        layout: Layout record to validate.

    Raises:
        ValueError: On any contract violation.
    """
    N = layout["graph_state_count"]

    # — Shape contracts —
    _check_array("valid_state_mask", layout["valid_state_mask"], (N,), bool)
    _check_array(
        "state_to_row_col", layout["state_to_row_col"], (N, 2), np.int32
    )
    _check_array("observation_id", layout["observation_id"], (N,), np.int32)
    _check_array("adjacency", layout["adjacency"], (N, N), bool)
    _check_array(
        "transition_matrix", layout["transition_matrix"], (N, N), np.floating
    )

    # — Action space —
    as_ = layout["action_space"]
    if as_["action_count"] < 1:
        raise ValueError(
            f"action_space.action_count must be >= 1, got {as_['action_count']}."
        )
    if len(as_["action_names"]) != as_["action_count"]:
        raise ValueError(
            f"action_names length {len(as_['action_names'])} != "
            f"action_count {as_['action_count']}."
        )
    if len(as_["action_deltas"]) != as_["action_count"]:
        raise ValueError(
            f"action_deltas length {len(as_['action_deltas'])} != "
            f"action_count {as_['action_count']}."
        )

    # — Valid mask —
    n_valid = int(layout["valid_state_mask"].sum())
    if n_valid == 0:
        raise ValueError("valid_state_mask has no True entries.")

    # — Adjacency symmetry —
    adj = layout["adjacency"]
    if not (adj == adj.T).all():
        raise ValueError("adjacency must be symmetric (undirected graph).")

    # — Self-loops — stay-still is expected, not enforced, but check diagonal
    #   entries exist for stay-state compatibility.
    for i in range(N):
        if layout["valid_state_mask"][i] and not adj[i, i]:
            raise ValueError(
                f"Valid state {i} has no self-loop (required for stay action)."
            )

    # — Transition matrix row-normalized —
    tm = layout["transition_matrix"]
    row_sums = tm.sum(axis=1)
    for i in range(N):
        if layout["valid_state_mask"][i]:
            if not np.isclose(row_sums[i], 1.0):
                raise ValueError(
                    f"transition_matrix row {i} (valid) sums to {row_sums[i]}, "
                    f"expected 1.0."
                )

    # — Invertible state_to_row_col mapping —
    row_col = layout["state_to_row_col"]
    if np.any(row_col < 0):
        raise ValueError("state_to_row_col contains negative values.")
    unique_pairs = set()
    for i in range(N):
        pair = (int(row_col[i, 0]), int(row_col[i, 1]))
        if pair in unique_pairs:
            raise ValueError(
                f"Duplicate (row, col) pair {pair} in state_to_row_col."
            )
        unique_pairs.add(pair)

    # — Provenance —
    if layout["sensory_vocab_size"] < 0:
        raise ValueError(
            "sensory_vocab_size must be >= 0 (0 = topology-only record)."
        )


# =============================================================================
# Internal helpers
# =============================================================================


def _check_array(
    name: str,
    arr: np.ndarray,
    expected_shape: tuple[int, ...],
    expected_dtype: type | tuple,
) -> None:
    if arr.shape != expected_shape:
        raise ValueError(
            f"Layout field {name!r}: expected shape {expected_shape}, "
            f"got {arr.shape}."
        )
    if isinstance(expected_dtype, tuple):
        if not any(np.issubdtype(arr.dtype, d) for d in expected_dtype):
            raise ValueError(
                f"Layout field {name!r}: expected dtype compatible with "
                f"{expected_dtype}, got {arr.dtype}."
            )
    elif not np.issubdtype(arr.dtype, expected_dtype):
        raise ValueError(
            f"Layout field {name!r}: expected dtype {expected_dtype}, "
            f"got {arr.dtype}."
        )


__all__ = [
    "ActionSpace",
    "DEFAULT_GRID_ACTION_SPACE",
    "HEX_ACTION_SPACE",
    "SpatialLayout",
    "validate_spatial_layout",
]
