"""Routebind task-owned contracts.

Routebind is a goal-conditioned spatial prospective-field prediction task.  Each
sample supplies a complete 2-D layout with walls, free cells, stable observation
identities, one start position, and one semantic goal observation.  The model
must predict a discounted trajectory field and a semantic waypoint field over
the 900 spatial positions, plus auxiliary next-direction and next-observation
heads.

All fields model the task-level input/output surface, not the adapter embedding
surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final

import numpy as np
from torch import Tensor

# Canonical ignore label for supervised outputs that should not contribute to
# loss (e.g. next_observation_logits when the task uses fewer than N_obs).
ROUTEBIND_IGNORE_LABEL_ID: Final[int] = -100

# Canonical target-semantics identifier (Routebind v1 contract)
TARGET_SEMANTICS: Final[str] = "optimal_subgraph_support"
"""Optimal-product-state-subgraph contract:
trajectory/waypoint support + minimum forward/semantic depth channels,
multi-label first-action masks, no single-path uniqueness requirement."""

TARGET_SCHEMA_VERSION: Final[int] = 1
"""Current corpus-schema version for ``target_semantics: optimal_subgraph_support``."""

# Depth sentinel for unsupported positions
ROUTEBIND_DEPTH_SENTINEL: Final[int] = -1
"""Value of trajectory_forward_depth or waypoint_semantic_depth when
support is False.  Depths >= 0 indicate valid optimal depth."""

# Cell type constants
CELL_WALL: Final[int] = 0
"""Cell is a wall (not traversable)."""
CELL_FREE: Final[int] = 1
"""Cell is traversable but contains no observation."""
CELL_OBSERVATION: Final[int] = 2
"""Cell is traversable and contains an observation identity."""
CELL_PAD: Final[int] = 3
"""Cell is outside the natural spatial domain of this sample (storage padding).
Not traversable, not a real wall, not part of the topology."""


# =============================================================================
# Direction vocabulary (single source of truth)
# =============================================================================


class Direction(IntEnum):
    """Canonical direction encoding for routebind task.

    Order: UP=0, RIGHT=1, DOWN=2, LEFT=3.
    This is the canonical neighbor-tie-breaking order for the greedy decoder.
    """

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


DIRECTION_DELTA: Final[dict[Direction, tuple[int, int]]] = {
    Direction.UP: (-1, 0),
    Direction.RIGHT: (0, 1),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
}
"""Row, col delta for each direction."""

DELTA_TO_DIRECTION: Final[dict[tuple[int, int], Direction]] = {
    (-1, 0): Direction.UP,
    (0, 1): Direction.RIGHT,
    (1, 0): Direction.DOWN,
    (0, -1): Direction.LEFT,
}
"""Reverse lookup from (row, col) delta to Direction."""


# =============================================================================
# Corpus channel schema — single descriptor for names, dtypes, and groups
# =============================================================================


@dataclass(frozen=True)
class RoutebindCorpusSchema:
    """Declares the routebind corpus channel vocabulary.

    Derives model-input channels, target channels, all channels, and expected
    dtypes from one declarative source.
    """

    # Model input channels
    cell_type: str = "cell_type"
    observation_id: str = "observation_id"
    start_flag: str = "start_flag"
    goal_flag: str = "goal_flag"
    spatial_mask: str = "spatial_mask"

    # Metadata channels (per-sample scalar, not model input)
    natural_height: str = "natural_height"
    natural_width: str = "natural_width"
    row_offset: str = "row_offset"
    col_offset: str = "col_offset"

    # Decayed target fields (derived from support × γ^depth)
    target_trajectory: str = "target_trajectory"
    target_waypoint: str = "target_waypoint"

    # Optimal-support channels (canonical oracle projection)
    trajectory_support: str = "trajectory_support"
    trajectory_forward_depth: str = "trajectory_forward_depth"
    waypoint_support: str = "waypoint_support"
    waypoint_semantic_depth: str = "waypoint_semantic_depth"
    target_optimal_directions: str = "target_optimal_directions"
    target_optimal_next_observations: str = "target_optimal_next_observations"

    # Scalar metadata
    total_physical_cost: str = "total_physical_cost"

    @property
    def model_input_channels(self) -> tuple[str, ...]:
        return (
            self.cell_type,
            self.observation_id,
            self.start_flag,
            self.goal_flag,
            self.spatial_mask,
        )

    @property
    def metadata_channels(self) -> tuple[str, ...]:
        return (
            self.natural_height,
            self.natural_width,
            self.row_offset,
            self.col_offset,
        )

    @property
    def target_channels(self) -> tuple[str, ...]:
        """All target and support channels."""
        return (
            self.target_trajectory,
            self.target_waypoint,
            self.trajectory_support,
            self.trajectory_forward_depth,
            self.waypoint_support,
            self.waypoint_semantic_depth,
            self.target_optimal_directions,
            self.target_optimal_next_observations,
            self.total_physical_cost,
        )

    @property
    def support_channels(self) -> tuple[str, ...]:
        """Optimal-support channels."""
        return (
            self.trajectory_support,
            self.trajectory_forward_depth,
            self.waypoint_support,
            self.waypoint_semantic_depth,
            self.target_optimal_directions,
            self.target_optimal_next_observations,
        )

    @property
    def all_channels(self) -> tuple[str, ...]:
        return (
            self.model_input_channels
            + self.metadata_channels
            + self.target_channels
            + self.support_channels
        )

    @property
    def training_target_channels(self) -> tuple[str, ...]:
        """Channels typically used as training supervision."""
        return (
            self.target_trajectory,
            self.target_waypoint,
            self.trajectory_support,
            self.trajectory_forward_depth,
            self.waypoint_support,
            self.waypoint_semantic_depth,
            self.target_optimal_directions,
            self.target_optimal_next_observations,
            self.total_physical_cost,
        )

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        return {
            self.cell_type: np.dtype(np.int32),
            self.observation_id: np.dtype(np.int32),
            self.start_flag: np.dtype(bool),
            self.goal_flag: np.dtype(bool),
            self.spatial_mask: np.dtype(bool),
            self.natural_height: np.dtype(np.int32),
            self.natural_width: np.dtype(np.int32),
            self.row_offset: np.dtype(np.int32),
            self.col_offset: np.dtype(np.int32),
            self.target_trajectory: np.dtype(np.float32),
            self.target_waypoint: np.dtype(np.float32),
            self.trajectory_support: np.dtype(bool),
            self.trajectory_forward_depth: np.dtype(np.int16),
            self.waypoint_support: np.dtype(bool),
            self.waypoint_semantic_depth: np.dtype(np.int16),
            self.target_optimal_directions: np.dtype(bool),
            self.target_optimal_next_observations: np.dtype(bool),
            self.total_physical_cost: np.dtype(np.int16),
        }


# Singleton for convenient import.
ROUTEBIND_SCHEMA: Final[RoutebindCorpusSchema] = RoutebindCorpusSchema()


# =============================================================================
@dataclass(frozen=True)
class RoutebindTaskInput:
    """Task-owned input for one routebind field-prediction sample.

    All fields are populated at the adapter level by the adapter encoder
    before the model sees this struct.

    Attributes:
        cell_type: Per-cell type category, shape ``(B, S)`` int32.
            Values are ``CELL_WALL (0)``, ``CELL_FREE (1)``,
            ``CELL_OBSERVATION (2)``, ``CELL_PAD (3)``.
        observation_id: Stable observation identity per cell, shape
            ``(B, S)`` int64.  Sentinel for non-observation cells.
        start_flag: ``True`` for exactly one traversable position, shape
            ``(B, S)`` bool.
        goal_flag: ``True`` for all positions containing the goal
            observation, shape ``(B, S)`` bool.
        spatial_mask: ``True`` where the stored slot corresponds to a real
            position in the sample's natural spatial domain; ``False`` where
            the slot is corpus-storage padding, shape ``(B, S)`` bool.
    """

    cell_type: Tensor
    observation_id: Tensor
    start_flag: Tensor
    goal_flag: Tensor
    spatial_mask: Tensor


# =============================================================================
@dataclass(frozen=True)
class RoutebindTaskOutput:
    """Task-owned routebind output from the model decoder.

    Attributes:
        trajectory_field: Predicted spatial trajectory field, shape
            ``(B, S)`` float32, each component in ``[0, 1]`` via sigmoid.
        waypoint_field: Predicted semantic waypoint field, shape ``(B, S)``
            float32, each component in ``[0, 1]`` via sigmoid.
        next_direction_logits: Categorical logits over {UP, DOWN, LEFT,
            RIGHT}, shape ``(B, 4)`` float32.
        next_observation_logits: Categorical logits over the observation
            vocabulary, shape ``(B, N_obs)`` float32.
    """

    trajectory_field: Tensor
    waypoint_field: Tensor
    next_direction_logits: Tensor
    next_observation_logits: Tensor


# =============================================================================
@dataclass(frozen=True)
class RoutebindTargets:
    """Routebind supervision targets derived from the corpus oracle.

    Attributes:
        target_trajectory: Oracle spatial trajectory field, shape ``(B, S)``
            float32, values in ``[0, 1]``.
        target_waypoint: Oracle semantic waypoint field, shape ``(B, S)``
            float32, values in ``[0, 1]``.
        target_next_dir: Ground-truth first physical movement direction,
            shape ``(B,)`` int32, values in ``{0, 1, 2, 3}`` for
            {UP, RIGHT, DOWN, LEFT}.
        target_next_obs: Ground-truth first post-start accepted observation
            identity, shape ``(B,)`` int64.
    """

    target_trajectory: Tensor
    target_waypoint: Tensor
    target_next_dir: Tensor
    target_next_obs: Tensor


# =============================================================================

# =============================================================================
# Policy transition types (for oracle kernel)
# =============================================================================


class PolicyTransition(IntEnum):
    """Transition kind for the reverse oracle policy array.

    Values:
        UNSET = 0 — state not reached / no policy assigned.
        MOVE = 1 — physical step: (p, o) -> (q, o).
        ACCEPT = 2 — semantic acceptance: (p, o) -> (p, o').
        GOAL = 3 — terminal: (p, g) at a goal occurrence.
    """

    UNSET = 0
    MOVE = 1
    ACCEPT = 2
    GOAL = 3


# =============================================================================
__all__ = [
    "CELL_FREE",
    "CELL_OBSERVATION",
    "CELL_WALL",
    "DELTA_TO_DIRECTION",
    "DIRECTION_DELTA",
    "Direction",
    "PolicyTransition",
    "ROUTEBIND_IGNORE_LABEL_ID",
    "ROUTEBIND_SCHEMA",
    "RoutebindCorpusSchema",
    "RoutebindTargets",
    "RoutebindTaskInput",
    "RoutebindTaskOutput",
]
