"""Deliberation value-control controller-to-learner contracts.

These types define the canonical controller-facing public surface for
value-based ACT deliberation rollouts.  They are not online-RL-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from ehc_sn.controllers._base import RolloutBackbone
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class ValueControlPolicyOutput(Protocol):
    """Policy payload produced by a value-control head."""

    q_values: Tensor
    valid_action_mask: Tensor | None


class ValueControlCriticOutput(Protocol):
    """Critic payload produced by the state-value head."""

    state_value: Tensor  # (B, 1)


class ValueControlBackboneOutput(Protocol):
    """Named backbone output split into task, policy, and critic surfaces."""

    task: object
    policy: ValueControlPolicyOutput
    critic: ValueControlCriticOutput


# =============================================================================
class ValueControlRolloutBackbone[ModelState](
    RolloutBackbone[ModelState, ValueControlBackboneOutput], Protocol
):
    """Backbone protocol required by value-control controllers."""


# =============================================================================
class ValueControlExecutionSnapshot(Protocol):
    """Minimal execution context required by the TD(0) batch assembler."""

    steps: Tensor  # (B,) per-slot step counters
    halted: Tensor  # (B,) per-slot halt flags


# =============================================================================
@dataclass
class ValueControlInteractionRecord(DetachMixin):
    """Controller-to-learner interaction record for value-control rollouts."""

    observation_used_for_decision: dict[str, Tensor]
    q_values: Tensor  # (B, A) value scores for halt/continue
    sampled_action: Tensor  # (B,) sampled action indices
    reward: Tensor  # (B, 1) task-finalized reward
    done: Tensor  # (B,) combined termination flag
    terminated: Tensor  # (B,) episode terminated flag
    truncated: Tensor  # (B,) episode truncated flag
    state_value: Tensor  # (B, 1) critic state value
    task_output: object | None = None  # optional task-side model output


# =============================================================================
__all__ = [
    "ValueControlBackboneOutput",
    "ValueControlCriticOutput",
    "ValueControlExecutionSnapshot",
    "ValueControlInteractionRecord",
    "ValueControlPolicyOutput",
    "ValueControlRolloutBackbone",
]
