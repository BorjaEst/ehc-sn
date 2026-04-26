"""Dungeon task-owned contracts.

Dungeon is the goal-directed navigation task family: observation/action
ontology, episode semantics, and goal-success score for dungeon-world
navigation.  Episode success is defined by reaching the goal within the
episode horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from torch import Tensor

from ehc_sn.tasks._movement import MOVEMENT_ACTION_COUNT, MovementAction

DUNGEON_ACTION_COUNT: Final[int] = MOVEMENT_ACTION_COUNT

DungeonAction = MovementAction
"""Canonical movement action type for dungeon navigation tasks."""


# =============================================================================
@dataclass(frozen=True)
class DungeonTaskInput:
    """Task-owned dungeon navigation payload for one episode step."""

    observation: Tensor
    """Encoded sensory observation vector, shape ``(B, obs_dim)``."""
    observation_id: Tensor
    """Categorical observation id, shape ``(B, 1)`` int64."""
    previous_action: Tensor
    """Action that produced the current state, shape ``(B, 1)`` int64."""
    location_id: Tensor
    """Flattened cell index ``row * W + col``, shape ``(B, 1)`` int64."""
    valid_action_mask: Tensor
    """Legal movement mask, shape ``(B, A)`` bool."""
    step_count: Tensor
    """Transitions taken in episode, shape ``(B, 1)`` int32."""
    region_id: Tensor | None = None
    """Optional region annotation, shape ``(B, 1)`` int64."""
    landmark_id: Tensor | None = None
    """Optional landmark / shiny-cue id, shape ``(B, 1)`` int64."""
    episode_start: Tensor | None = None
    """True on the first step of an episode, shape ``(B,)`` bool."""
    is_revisit: Tensor | None = None
    """True when the current location has been visited before, shape ``(B,)`` bool."""


# =============================================================================
@dataclass(frozen=True)
class DungeonScoreReport:
    """Benchmark-time score report for a batch of completed dungeon episodes.

    Token-supervision targets are not defined in dungeon v1.
    """

    success: Tensor
    """Goal reached within episode horizon, shape ``(B,)`` bool."""
    score: Tensor
    """Sparse path-efficiency score: ``1 / episode_steps`` for successful episodes, 0 for failures.

    Shape ``(B,)`` float.  Computed by
    :func:`~ehc_sn.tasks.dungeon.evaluation.build_dungeon_score_report` from the
    task-owned episode-end tensors ``success`` and ``episode_steps``.
    Higher is better; a perfect one-step success scores 1.0.
    """
    episode_steps: Tensor
    """Steps taken in the episode, shape ``(B,)`` int64."""


# =============================================================================
__all__ = [
    "DUNGEON_ACTION_COUNT",
    "DungeonAction",
    "DungeonScoreReport",
    "DungeonTaskInput",
]
