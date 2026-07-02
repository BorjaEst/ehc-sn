"""Arena task-owned contracts.

Arena is the structural-navigation task family: observation/action ontology,
revisit semantics, and additive structural score for maze-world navigation.

Contracts owns: semantic task input, output, targets, actions, and constants.
Score report lives in :mod:`ehp_sn.tasks.arena.evaluation`.

Arena replay v1 is topology-free.  ``ArenaTaskInput`` carries only what the
model needs: ids and step-semantic flags precomputed at build time.
``valid_action_mask`` and ``location_id`` are not part of Arena replay v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from torch import Tensor

from ehc_sn.tasks._movement import MovementAction

ArenaAction = MovementAction
"""Canonical movement action type for arena navigation tasks."""


# =============================================================================
@dataclass(frozen=True)
class ArenaTaskInput:
    """Task-owned current-step input for one arena navigation step.

    All fields are populated by the adapter encoder before the model sees this
    struct.  Observation encoding (e.g. one-hot) is adapter-side so the task
    contract remains model-agnostic.

    Arena replay v1 contract:
    - No topology is present in the batch.
    - ``valid_action_mask`` is not exposed.
    - ``location_id`` is not exposed.
    - Provenance (parent_sample_id etc.) is not exposed.
    """

    observation_id: Tensor
    """Categorical observation id, shape ``(B, 1)`` int64."""
    previous_action: Tensor
    """Action that produced the current state, shape ``(B, 1)`` int64."""
    landmark_id: Tensor | None
    """Optional landmark / shiny-cue id, shape ``(B, 1)`` int64."""
    step_count: Tensor
    """Transitions taken in episode, shape ``(B, 1)`` int32."""
    episode_start: Tensor
    """True on the first step of an episode, shape ``(B,)`` bool."""


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
    "ArenaAction",
    "ArenaTargets",
    "ArenaTaskInput",
    "ArenaTaskOutput",
]
