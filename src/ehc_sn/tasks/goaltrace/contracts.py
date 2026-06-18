"""Goaltrace task-owned contracts.

Goaltrace is the isolated HRM/PFC training task for goal-conditioned
prospective field prediction.  All fields model the task-level input/output
surface, not the adapter embedding surface.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


# =============================================================================
@dataclass(frozen=True)
class GoaltraceTaskInput:
    """Task-owned input for one goaltrace field-prediction sample.

    All fields are populated at the adapter level by the adapter encoder
    before the model sees this struct.  Observation encoding
    (e.g. learned embeddings for observation_id) is adapter-side so the
    task contract remains model-agnostic.

    Attributes:
        observation_id: Stable observation identity per slot, shape ``(B, N)`` int64.
        weight: Relational weight from current location to candidate j, shape ``(B, N)`` float32.
        current_flag: True for exactly one slot (the current location g_t), shape ``(B, N)`` bool.
        goal_flag: True for exactly one slot (the goal observation x_goal), shape ``(B, N)`` bool.
        node_mask: True for valid (non-padded) nodes, shape ``(B, N)`` bool.
    """

    observation_id: Tensor
    weight: Tensor
    current_flag: Tensor
    goal_flag: Tensor
    node_mask: Tensor


# =============================================================================
@dataclass(frozen=True)
class GoaltraceTaskOutput:
    """Task-owned goaltrace output from the model decoder.

    Attributes:
        firing_field: Predicted goal-conditioned prospective firing field,
            shape ``(B, N)`` float32, each component in ``[0, 1]`` via sigmoid.
    """

    firing_field: Tensor


# =============================================================================
@dataclass(frozen=True)
class GoaltraceTargets:
    """Goaltrace supervision targets derived from the corpus oracle.

    Attributes:
        target_field: Oracle goal-conditioned prospective firing field,
            shape ``(B, N)`` float32, each component in ``[0, 1]``.
    """

    target_field: Tensor


# =============================================================================
__all__ = [
    "GoaltraceTaskInput",
    "GoaltraceTaskOutput",
    "GoaltraceTargets",
]
