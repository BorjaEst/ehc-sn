"""Dungeon task-owned contracts for reward-first online control.

Dungeon is the canonical task/runtime family backing B1-B3-style online
control benchmark reports.  Success semantics are goal/return-driven and
must not share the arena structural-knowledge benchmark claim (REQ-002).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from torch import Tensor

from ehc_sn.tasks._movement import MOVEMENT_ACTION_COUNT, MovementAction

DUNGEON_ACTION_COUNT: Final[int] = MOVEMENT_ACTION_COUNT

DungeonAction = MovementAction
"""Canonical movement action type for dungeon online control tasks."""


# =============================================================================
@dataclass(frozen=True)
class DungeonObservation:
    """Current-step environment observation emitted by the dungeon task."""

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


# =============================================================================
@dataclass(frozen=True)
class DungeonTaskInput(DungeonObservation):
    """Task-owned dungeon control payload for one controller step."""

    episode_start: Tensor | None = None
    """True on the first step of an episode, shape ``(B,)`` bool."""
    is_revisit: Tensor | None = None
    """True when the current location has been visited before, shape ``(B,)`` bool."""


# =============================================================================
@dataclass(frozen=True)
class DungeonScoreReport:
    """Benchmark-time score report for a batch of completed dungeon episodes.

    Used for B1-B3-style online control benchmark evaluation.
    Token-supervision targets are not defined in dungeon v1.
    """

    success: Tensor
    """Goal reached within episode horizon, shape ``(B,)`` bool."""
    total_return: Tensor
    """Cumulative reward over the episode, shape ``(B,)`` float."""
    episode_steps: Tensor
    """Steps taken in the episode, shape ``(B,)`` int64."""


# =============================================================================
__all__ = [
    "DUNGEON_ACTION_COUNT",
    "DungeonAction",
    "DungeonObservation",
    "DungeonScoreReport",
    "DungeonTaskInput",
]
