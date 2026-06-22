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

    Declares the number of actions, their names, the row/col deltas that
    implement them, and the movement kind.  The ``stay_action`` field is
    the action id of the no-op action, or ``None`` if the layout has no
    stay action.  ``movement_kind`` distinguishes orthogonal grid
    movement from hexagonal or future movement types.
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
    movement_kind: str
    """Movement topology kind (``"grid4"``, ``"hex6"``).  Independent of
    the raster type (square vs rectangle vs hex), which is declared by
    ``topology_type``."""


# =============================================================================
# Core layout record
# =============================================================================


class SpatialLayout(TypedDict):
    """Graph-indexed spatial world record — compact representation.

    Every stored row represents exactly one traversable state.  Walls,
    padding, and pruned backing cells are excluded from the exported
    artifact.  There are no invalid rows.

    Fields (in declaration order):
    """

    layout_id: str
    """Unique identifier for this layout instance."""
    layout_family: str
    """Layout source family (e.g. ``"openfield"``, ``"dungeongen"``)."""
    topology_type: str
    """Geometric refinement of the topology (e.g. ``"square"``, ``"rectangle"``, ``"hex"``).
    Independent of mathematical topology class (``topology_kind``)."""

    topology_kind: NotRequired[str]
    """Mathematical topology class: ``"grid2d"``, ``"dag"``, ``"line1d"``.
    Describes the abstract structure independent of geometric refinement."""

    graph_state_count: int
    """Number of nodes in the compact graph index space (only traversable states)."""

    state_to_row_col: np.ndarray
    """``(graph_state_count, 2)`` int32 — row, col for each graph state."""
    observation_id: np.ndarray
    """``(graph_state_count,)`` int32 — observation id per graph state.
    IDs are drawn from ``{0, ..., observation_vocabulary_size-1}`` and may
    repeat.  Sentinel ``-1`` indicates no observation (topology-only stage)."""

    extent: NotRequired[tuple[int, int]]
    """Declared canvas dimensions ``(height, width)`` in cells.  When present,
    every coordinate in ``state_to_row_col`` satisfies ``0 ≤ row < height``
    and ``0 ≤ col < width``.  Per-layout extent is authoritative; the
    manifest-level copy is a convenience field."""

    next_state: np.ndarray
    """``(graph_state_count, action_count)`` int32 — destination compact
    state index for each action from each state.  For the STAY action
    ``next_state[s, STAY] == s``.  For an invalid action
    ``next_state[s, a] == s`` (self-loop sentinel)."""

    action_valid: np.ndarray
    """``(graph_state_count, action_count)`` bool — whether action ``a``
    is a valid state-changing movement from state ``s``.  STAY is always
    ``True``.  An action that would cross a wall or boundary with no
    traversable neighbor is ``False``."""

    action_space: ActionSpace
    """Action-space descriptor for this layout."""

    # Provenance
    topology_seed: int
    """Seed used to generate the topology (width, shape)."""
    observation_seed: int
    """Seed used for the observation_id random assignment."""
    observation_vocabulary_size: int
    """Number of distinct observation ids (size of the observation vocabulary)."""
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
    "movement_kind": "grid4",
}
"""Default 4-direction + stay action space for square/dungeon grids.

``movement_kind = "grid4"`` indicates orthogonal grid-neighbor movement."""

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
    "movement_kind": "hex6",
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
    A = layout["action_space"]["action_count"]

    # — Shape contracts —
    _check_array(
        "state_to_row_col", layout["state_to_row_col"], (N, 2), np.int32
    )
    _check_array("observation_id", layout["observation_id"], (N,), np.int32)
    _check_array("next_state", layout["next_state"], (N, A), np.int32)
    _check_array("action_valid", layout["action_valid"], (N, A), bool)

    # — Action space —
    as_ = layout["action_space"]
    if A < 1:
        raise ValueError(f"action_space.action_count must be >= 1, got {A}.")
    if len(as_["action_names"]) != A:
        raise ValueError(
            f"action_names length {len(as_['action_names'])} != "
            f"action_count {A}."
        )
    if len(as_["action_deltas"]) != A:
        raise ValueError(
            f"action_deltas length {len(as_['action_deltas'])} != "
            f"action_count {A}."
        )
    if not as_.get("name"):
        raise ValueError(
            "action_space.name is required but was empty or missing."
        )
    if not as_.get("movement_kind"):
        raise ValueError(
            "action_space.movement_kind is required but was empty or missing."
        )

    # — Next state invariants —
    next_s = layout["next_state"]
    act_val = layout["action_valid"]
    stay_idx = as_.get("stay_action")
    if stay_idx is not None:
        # STAY must self-map and be valid.
        for s in range(N):
            if next_s[s, stay_idx] != s:
                raise ValueError(
                    f"next_state[s={s}, stay_action={stay_idx}] = "
                    f"{next_s[s, stay_idx]}, expected {s} (self-loop)."
                )
            if not act_val[s, stay_idx]:
                raise ValueError(
                    f"action_valid[s={s}, stay_action={stay_idx}] is False, "
                    f"expected True."
                )

    # Every next_state value must be in [0, N).
    if np.any(next_s < 0) or np.any(next_s >= N):
        raise ValueError(
            "next_state contains values outside [0, graph_state_count)."
        )

    # action_valid must be False when next_state == s for non-STAY actions.
    for a in range(A):
        if stay_idx is not None and a == stay_idx:
            continue
        for s in range(N):
            if not act_val[s, a] and next_s[s, a] != s:
                raise ValueError(
                    f"action_valid[s={s}, a={a}] is False but "
                    f"next_state[s, a] = {next_s[s, a]} != s. "
                    "Invalid actions must self-loop."
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
    if layout["observation_vocabulary_size"] < 0:
        raise ValueError(
            "observation_vocabulary_size must be >= 0 "
            "(0 = topology-only record)."
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
