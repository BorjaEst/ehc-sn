"""Deliberation value-control rollout controller — canonical owner.

This controller drives an actor-critic backbone through slot-based deliberation
steps without an environment.  Reward and termination finalization are delegated
to an injected :class:`DeliberationStepFinalizer`, keeping the controller
generic over task semantics.

Emits one :class:`~ehc_sn.controllers.contracts.value_control.ValueControlInteractionRecord`
per step.  No ``env_td``, no :class:`~ehc_sn.controllers.online.actor_critic.RLTaskRuntime`,
no TorchRL dependency.

Canonical import path::

    from ehc_sn.controllers.deliberation.actor_critic import (
        DeliberationACController, DeliberationACControllerConfig,
        DeliberationACRolloutState, DeliberationStepFinalizer, DeliberationStepResult,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers._base import BaseController, RolloutState
from ehc_sn.controllers.contracts.value_control import (
    ValueControlInteractionRecord,
    ValueControlRolloutBackbone,
)
from ehc_sn.controllers.deliberation.act import (
    collapse_act_halt_continue_logits,
    maybe_flip_halt_decision,
)
from ehc_sn.policies.categorical import (
    CategoricalPolicy,
    CategoricalPolicyConfig,
    PolicyInput,
)
from ehc_sn.types import Batch


# =============================================================================
@dataclass(frozen=True)
class DeliberationStepResult:
    """Per-step finalization output produced by :class:`DeliberationStepFinalizer`.

    Attributes:
        reward: Task-finalized reward tensor of shape ``(B, 1)``.
        terminated: Per-slot episode termination flag of shape ``(B,)``.
        truncated: Per-slot episode truncation flag of shape ``(B,)``.
        next_runtime_state: Updated lightweight runtime carry, or ``None`` if
            the finalizer is stateless.
    """

    reward: Tensor  # (B, 1)
    terminated: Tensor  # (B,)
    truncated: Tensor  # (B,)
    next_runtime_state: object | None = None


# =============================================================================
class DeliberationStepFinalizer(Protocol):
    """Step-finalization seam injected into :class:`DeliberationACController`.

    Owns semantic reward and termination flag production given the current-step
    task output and batch.  The controller calls this after the backbone forward
    pass and policy sampling; the finalizer must not access controller-internal
    data structures.

    Implementations belong in task-owned or adapter-owned layers, not in the
    controller itself.
    """

    def finalize_step(
        self,
        data: Batch,
        task_output: object,
        action: Tensor,
        steps: Tensor,
        runtime_state: object | None,
    ) -> DeliberationStepResult:
        """Produce reward and termination flags for the current step.

        Args:
            data: Current per-slot batch.
            task_output: Task-side model output from ``backbone_output.task``.
            action: Sampled action tensor of shape ``(B,)``.
            steps: Per-slot step counters of shape ``(B,)`` after advancing.
            runtime_state: Lightweight runtime carry from the previous step.

        Returns:
            :class:`DeliberationStepResult` with reward, termination flags, and
            the updated runtime state.
        """
        ...


# =============================================================================
class DeliberationACControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`DeliberationACController`.

    Attributes:
        policy: Configuration for the categorical action policy.
    """

    policy: CategoricalPolicyConfig = Field(
        default_factory=CategoricalPolicyConfig,
        description="Categorical action policy configuration.",
    )


# =============================================================================
@dataclass
class DeliberationACRolloutState[ModelState](RolloutState[ModelState]):
    """Controller carry/state for deliberation actor-critic rollouts.

    Does not contain ``env_td`` and does not depend on
    :class:`~ehc_sn.controllers.online.actor_critic.RLTaskRuntime` or
    :class:`~ehc_sn.controllers.online.actor_critic.RLRolloutState`.

    Attributes:
        runtime_state: Lightweight carry owned by the injected
            :class:`DeliberationStepFinalizer`.  ``None`` for stateless
            finalizers.
    """

    runtime_state: object | None = None


# =============================================================================
class DeliberationACController[ModelState](
    BaseController[ModelState, DeliberationACControllerConfig]
):
    """Policy-driven deliberation value-control controller without an environment.

    The controller:
        - maintains per-slot buffers across steps via the inherited slot lifecycle
        - resets backbone state for halted slots
        - samples actions from ``q_values`` via :class:`~ehc_sn.policies.categorical.CategoricalPolicy`
        - delegates reward and termination to the injected :class:`DeliberationStepFinalizer`
        - emits one :class:`~ehc_sn.controllers.contracts.value_control.ValueControlInteractionRecord` per step

    No TorchRL environment, no ``env_td``, no :class:`~ehc_sn.controllers.online.actor_critic.RLTaskRuntime`.
    """

    def __init__(
        self,
        backbone: ValueControlRolloutBackbone[ModelState],
        config: DeliberationACControllerConfig,
        finalizer: DeliberationStepFinalizer,
    ) -> None:
        """Create a deliberation actor-critic controller."""
        super().__init__(backbone=cast(Any, backbone), config=config)
        self._policy = CategoricalPolicy(config.policy)
        self._finalizer = finalizer

    @property
    def backbone(self) -> ValueControlRolloutBackbone[ModelState]:
        """Return the wrapped backbone typed to the value-control protocol."""
        return cast(ValueControlRolloutBackbone[ModelState], super().backbone)

    @property
    def finalizer(self) -> DeliberationStepFinalizer:
        """Return the injected step finalizer."""
        return self._finalizer

    def initial_state(
        self,
        batch_sample: Batch,
        *,
        initial_runtime_state: object | None = None,
    ) -> DeliberationACRolloutState[ModelState]:
        """Build an initial rollout state from a batch sample."""
        slots = self.initial_slots(batch_sample)
        return DeliberationACRolloutState(
            model_state=slots.model_state,
            steps=slots.steps,
            halted=slots.halted,
            data=slots.data,
            runtime_state=initial_runtime_state,
        )

    def step(
        self,
        state: DeliberationACRolloutState[ModelState],
        batch: Batch,
        *,
        allow_halt: bool = True,
        explore: bool = True,
        **options: Any,
    ) -> tuple[
        DeliberationACRolloutState[ModelState], ValueControlInteractionRecord
    ]:
        """Advance the controller by one step.

        Sequence:
            1. Refresh per-slot data for halted rows.
            2. Reset backbone state for halted rows.
            3. Run the backbone forward pass.
            4. Sample an action via :class:`~ehc_sn.policies.categorical.CategoricalPolicy`.
            5. Call :meth:`DeliberationStepFinalizer.finalize_step` for reward/termination.
            6. Compute ``done`` as ``(terminated | truncated)``.
            7. Emit :class:`~ehc_sn.controllers.contracts.value_control.ValueControlInteractionRecord`.

        Args:
            state: Current rollout state.
            batch: Incoming batch used to refresh halted slots.
            allow_halt: If ``False``, the finalizer's ``terminated`` signal is
                suppressed (zeroed out), preventing learned-halt termination from
                closing slots.  Task-owned ``truncated`` from the finalizer always
                passes through regardless of this flag.
            explore: Passed to the categorical policy.
            halt_action: Optional action index treated as the halt action when
                applying ACT-style halt/continue semantics.
            max_halt_steps: Optional per-slot step budget used by ACT-style halt
                semantics to force halting.

        Returns:
            ``(new_state, record)``.
        """
        data = self.refresh_slot_data(batch, state)
        model_state = self.backbone.reset_state(state.halted, state.model_state)
        backbone_output, model_state = self.backbone(data, model_state)

        steps = self.advance_steps(state)

        policy = backbone_output.policy
        q_values = policy.q_values
        valid_action_mask = policy.valid_action_mask
        if valid_action_mask is None:
            valid_action_mask = torch.ones_like(q_values, dtype=torch.bool)

        halt_action = options.get("halt_action")
        max_halt_steps = options.get("max_halt_steps")
        if halt_action is not None and max_halt_steps is not None:
            scores = collapse_act_halt_continue_logits(
                q_values, done_action=halt_action
            )
            if allow_halt:
                exploration_prob = self.config.policy.exploration_prob or 0.0
                halt = maybe_flip_halt_decision(
                    scores.greedy_halt,
                    steps=steps,
                    explore=explore,
                    exploration_prob=exploration_prob,
                    max_halt_steps=max_halt_steps,
                )
                halt = halt | (steps >= max_halt_steps)
            else:
                halt = torch.zeros_like(scores.greedy_halt, dtype=torch.bool)

            non_halt_mask = valid_action_mask.clone()
            non_halt_mask[:, halt_action] = False
            halt_only_mask = torch.zeros_like(valid_action_mask)
            halt_only_mask[:, halt_action] = True
            valid_action_mask = torch.where(
                halt.unsqueeze(-1), halt_only_mask, non_halt_mask
            )

        policy_input = PolicyInput(
            logits=q_values, valid_action_mask=valid_action_mask
        )
        policy_decision = self._policy(policy_input, explore=explore)
        action = policy_decision.action.to(torch.int64)

        step_result = self._finalizer.finalize_step(
            data,
            backbone_output.task,
            action,
            steps,
            state.runtime_state,
        )

        if allow_halt:
            terminated = step_result.terminated
            truncated = step_result.truncated
            done = terminated | truncated
        else:
            # Suppress learned-halt termination only.  Task-owned truncated
            # (e.g. episode_horizon reached) passes through unchanged.
            terminated = torch.zeros_like(step_result.terminated)
            truncated = step_result.truncated
            done = truncated

        new_state = DeliberationACRolloutState(
            model_state=model_state,
            steps=steps,
            halted=done,
            data=data,
            runtime_state=step_result.next_runtime_state,
        )
        record = ValueControlInteractionRecord(
            observation_used_for_decision=data,
            q_values=q_values,
            sampled_action=action,
            reward=step_result.reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            state_value=backbone_output.critic.state_value,
            task_output=backbone_output.task,
        )
        return new_state, record


# =============================================================================
__all__ = [
    "DeliberationACController",
    "DeliberationACControllerConfig",
    "DeliberationACRolloutState",
    "DeliberationStepFinalizer",
    "DeliberationStepResult",
]
