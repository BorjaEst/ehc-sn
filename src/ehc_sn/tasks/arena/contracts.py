"""Arena task-owned contracts.

Arena is the stepwise teacher-forced replay task family that backs
structural-knowledge (B1-style) benchmark claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from torch import Tensor

from ehc_sn.tasks._movement import MOVEMENT_ACTION_COUNT, MovementAction

ARENA_ACTION_COUNT: Final[int] = MOVEMENT_ACTION_COUNT

ArenaAction = MovementAction
"""Canonical movement action type for arena replay tasks."""


# =============================================================================
@dataclass(frozen=True)
class ArenaTaskInput:
    """Task-owned current-step input for one arena replay step.

    All required fields are populated by the adapter encoder before the model
    sees this struct.  ``observation`` carries the encoded (not raw) sensory
    vector; encoding is adapter-side so the task contract remains
    model-agnostic.
    """

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
class ArenaTargets:
    """Arena supervision targets derived from the current task step."""

    observation_id: Tensor
    """Ground-truth observation id for this step, shape ``(B,)`` or ``(B, 1)``."""
    is_revisit: Tensor | None = None
    """Revisit mask for revisit-split evaluation, shape ``(B,)`` bool."""


# =============================================================================
@dataclass(frozen=True)
class ArenaTaskOutput:
    """Arena decoder output surface.

    Carries only the canonical observation-logit prediction.
    TEM-specific multi-pathway diagnostics live in the adapter bridge layer.
    """

    obs_logits: Tensor
    """Predicted observation logits, shape ``(B, obs_dim)``."""


# =============================================================================
__all__ = [
    "ARENA_ACTION_COUNT",
    "ArenaAction",
    "ArenaTargets",
    "ArenaTaskInput",
    "ArenaTaskOutput",
]
