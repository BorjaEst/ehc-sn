"""Neutral actor-critic controller-to-learner contracts.

These types are the canonical controller-facing public surface for any
actor-critic rollout family.  They are **not** online-RL-specific: any
controller that drives a policy-plus-value backbone and emits interaction
records may use them.

Online RL pieces (``RLRolloutState``, ``RLController``) live in
:mod:`ehc_sn.controllers.online.actor_critic`; the task-environment adapter
lives in :mod:`ehc_sn.contracts.task_environment`.

The two minimal bootstrap protocols (:class:`OnlineBootstrapCarry` and
:class:`OnlineBootstrapRuntime`) are defined here so that both the training
layer and the online controller can depend on the same neutral owner without
creating a training→controller import inversion.

Canonical import path::

    from ehc_sn.controllers.contracts.actor_critic import (
        QHaltingInteractionRecord,
        QHaltingRolloutBackbone,
        QHaltingBackboneOutput,
        QHaltingPolicyOutput,
        QHaltingCriticOutput,
        QHaltingExecutionSnapshot,
        OnlineBootstrapCarry,
        OnlineBootstrapRuntime,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor

from ehc_sn.controllers._base import RolloutBackbone
from ehc_sn.policies._base import PolicyDecision
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class QHaltingPolicyOutput(Protocol):
    """Policy payload produced by the actor head of a backbone step.

    Actor-critic controllers populate ``policy_logits``; value-control
    controllers populate ``q_values``.  Exactly one must be non-``None``.
    """

    policy_logits: Tensor
    q_values: Tensor | None = None
    valid_action_mask: Tensor | None


class QHaltingCriticOutput(Protocol):
    """Critic payload produced by the value head of a backbone step."""

    state_value: Tensor  # (B, 1)


class QHaltingBackboneOutput(Protocol):
    """Named backbone output split into task, policy, and critic surfaces.

    All actor-critic backbones **must** return a non-None critic output.
    A backbone that omits the value head violates this contract.
    """

    task: object
    policy: QHaltingPolicyOutput
    critic: QHaltingCriticOutput


# =============================================================================
class QHaltingRolloutBackbone[ModelState](
    RolloutBackbone[ModelState, QHaltingBackboneOutput], Protocol
):
    """Backbone protocol required by any actor-critic controller."""


# =============================================================================
class QHaltingExecutionSnapshot(Protocol):
    """Minimal execution context required by the generic TD(0) batch assembler.

    Any rollout state that carries per-slot step counters and halt flags
    satisfies this protocol.
    """

    steps: Tensor  # (B,) per-slot step counters
    halted: Tensor  # (B,) per-slot halt flags


# =============================================================================
class OnlineBootstrapCarry(QHaltingExecutionSnapshot, Protocol):
    """Minimal online rollout carry required for TD(0) bootstrap value computation.

    Extends :class:`QHaltingExecutionSnapshot` (``steps``, ``halted``) with
    ``model_state``, which is the only additional field accessed during online
    bootstrap value computation.

    :class:`~ehc_sn.controllers.online.actor_critic.RLRolloutState` satisfies
    this protocol structurally.  No concrete RL types need be imported into the
    training layer.
    """

    model_state: Any  # opaque recurrent model state; passed directly to backbone.__call__


# =============================================================================
class OnlineBootstrapRuntime(Protocol):
    """Minimal task-runtime surface required for online TD(0) bootstrap extraction.

    Implement this protocol in the task-runtime layer (e.g.
    :class:`~ehc_sn.contracts.task_environment.TaskEnvironmentAdapter`) to expose
    only the one method that the learner requires.  The full env-step shaping
    contract is not needed by the training layer.
    """

    def extract_next_step_obs(self, carry: OnlineBootstrapCarry) -> Batch:
        """Return the next-step observation batch for bootstrap critic evaluation."""
        ...


# =============================================================================
@dataclass
class QHaltingInteractionRecord(DetachMixin):
    """Neutral controller-to-learner interaction record emitted per rollout step.

    This is a **controller-level public contract**, not an online-RL-specific
    type.  Any controller that drives an actor-critic backbone may emit this
    record.  Learners receive it directly and own all TD(0) / GAE
    post-processing.  Objectives consume fully materialised batches and must
    not access controller-internal data structures through this record.

    ``observation_used_for_decision`` is the exact ``data`` dict passed to the
    backbone forward pass — threaded explicitly, not reconstructed from carry.
    """

    observation_used_for_decision: dict[
        str, Tensor
    ]  # exact batch used for model step
    policy_logits: Tensor  # (B, A) actor-head policy logits
    sampled_action: Tensor  # (B,) sampled action indices
    reward: Tensor  # (B, 1) task-finalized reward
    done: Tensor  # (B,) combined termination flag
    terminated: Tensor  # (B,) episode terminated flag
    truncated: Tensor  # (B,) episode truncated flag
    value_estimate: Tensor  # (B, 1) critic state value
    policy_decision: PolicyDecision  # rollout-time log_prob, entropy
    q_values: Tensor | None = None  # (B, A) value scores for halt/continue
    task_output: object | None = None  # optional task-side model output


# =============================================================================
__all__ = [
    "QHaltingBackboneOutput",
    "QHaltingCriticOutput",
    "QHaltingExecutionSnapshot",
    "QHaltingInteractionRecord",
    "QHaltingPolicyOutput",
    "QHaltingRolloutBackbone",
    "OnlineBootstrapCarry",
    "OnlineBootstrapRuntime",
]
