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
    goal_row: Tensor | None = None
    """Goal cell row index per slot, shape ``(B, 1)`` int64.  None when the corpus predates goal channels."""
    goal_col: Tensor | None = None
    """Goal cell column index per slot, shape ``(B, 1)`` int64.  None when the corpus predates goal channels."""


# =============================================================================
@dataclass(frozen=True)
class DungeonTaskOutput:
    """Task-owned dungeon navigation output.

    Carries the model's per-step action prediction.
    """

    action_logits: Tensor
    """Predicted action logits, shape ``(B, A)`` float."""


# =============================================================================
__all__ = [
    "DUNGEON_ACTION_COUNT",
    "DungeonAction",
    "DungeonTaskInput",
    "DungeonTaskOutput",
]
