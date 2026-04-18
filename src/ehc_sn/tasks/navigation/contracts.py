"""Navigation task-owned contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

from torch import Tensor

from ehc_sn.envs.dungeon_walk import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, ACTION_UP, DEFAULT_ACTION_COUNT

NAVIGATION_ACTION_COUNT: Final[int] = DEFAULT_ACTION_COUNT
"""Canonical navigation action count for the dungeon-walk task."""


# =============================================================================
class NavigationAction(IntEnum):
    """Canonical discrete movement ontology for navigation tasks."""

    STAY = ACTION_STAY
    UP = ACTION_UP
    RIGHT = ACTION_RIGHT
    DOWN = ACTION_DOWN
    LEFT = ACTION_LEFT


# =============================================================================
@dataclass(frozen=True)
class NavigationObservation:
    """Current-step environment observation emitted by the navigation task."""

    observation: Tensor
    observation_id: Tensor
    previous_action: Tensor
    location_id: Tensor
    valid_action_mask: Tensor
    step_count: Tensor
    region_id: Tensor | None = None
    landmark_id: Tensor | None = None


# =============================================================================
@dataclass(frozen=True)
class NavigationTaskInput(NavigationObservation):
    """Task-owned TEM-ready navigation payload for one controller step."""

    episode_start: Tensor | None = None
    is_revisit: Tensor | None = None


# =============================================================================
@dataclass(frozen=True)
class NavigationTargets:
    """Navigation supervision targets derived from the current task step."""

    observation_id: Tensor
    is_revisit: Tensor | None = None


# =============================================================================
@dataclass(frozen=True)
class NavigationTaskOutput:
    """Task-owned navigation decoder payload.

    Carries only the canonical observation-logit prediction surface.
    TEM-specific multi-path diagnostics (retrieved, ancestral pathways) live
    in the adapter bridge layer, not here.
    """

    obs_logits: Tensor


# =============================================================================
__all__ = [
    "NAVIGATION_ACTION_COUNT",
    "NavigationAction",
    "NavigationObservation",
    "NavigationTargets",
    "NavigationTaskInput",
    "NavigationTaskOutput",
]
