"""Deliberation Q-halting rollout controller — canonical owner.

This controller drives a value-control backbone through slot-based deliberation
steps.  Reward, termination, and observation dynamics are delegated to an
injected :class:`~ehp_sn.contracts.task_runtime.TaskRuntime`, keeping the
controller generic over task semantics.

Emits one :class:`~ehp_sn.controllers.contracts.actor_critic.QHaltingInteractionRecord`
per step.  No ``env_td``, no :class:`~ehp_sn.contracts.task_step.TaskStepEvaluator`,
no :class:`~ehp_sn.contracts.task_environment.TaskEnvironmentAdapter`,
no TorchRL dependency.

Canonical import path::

    from ehc_sn.controllers.deliberation.q_halting import (
        DeliberationQHaltingController, DeliberationQHaltingControllerConfig,
        DeliberationQHaltingRolloutState,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar, cast

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.contracts.task_runtime import TaskRuntime
from ehc_sn.controllers._base import BaseController, RolloutState
from ehc_sn.controllers.contracts.actor_critic import (
    QHaltingInteractionRecord,
    QHaltingRolloutBackbone,
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
_RuntimeStateT = TypeVar("_RuntimeStateT")


# =============================================================================
class DeliberationQHaltingControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`DeliberationQHaltingController`.

    Attributes:
        policy: Configuration for the categorical action policy.
    """

    policy: CategoricalPolicyConfig = Field(
        default_factory=CategoricalPolicyConfig,
        description="Categorical action policy configuration.",
    )


# =============================================================================
@dataclass
class DeliberationQHaltingRolloutState[ModelState, RuntimeStateT](
    RolloutState[ModelState]
):
    """Controller carry/state for deliberation Q-halting rollouts.

    Does not contain ``env_td`` and does not depend on
    :class:`~ehp_sn.contracts.task_environment.TaskEnvironmentAdapter` or
    :class:`~ehp_sn.controllers.online.actor_critic.RLRolloutState`.

    Attributes:
        runtime_state: Task-owned runtime state, threaded through the
            injected :class:`~ehp_sn.contracts.task_runtime.TaskRuntime`.
    """

    runtime_state: RuntimeStateT


# =============================================================================
class DeliberationQHaltingController[ModelState, RuntimeStateT](
    BaseController[ModelState, DeliberationQHaltingControllerConfig]
):
    """Runtime-backed deliberation Q-halting value-control controller.

    The controller:
        - maintains per-slot buffers across steps via the inherited slot lifecycle
        - resets and refreshes halted slots via the injected :class:`TaskRuntime`
        - resets backbone state for halted slots
        - samples actions from ``q_values`` via :class:`~ehp_sn.policies.categorical.CategoricalPolicy`
        - delegates reward, termination, and observation dynamics to the injected
          :class:`~ehp_sn.contracts.task_runtime.TaskRuntime`
        - emits one :class:`~ehp_sn.controllers.contracts.actor_critic.QHaltingInteractionRecord`
          per step

    Task-agnostic: the controller imports no task-specific types.  The only
    task surface is ``self._runtime`` typed as ``TaskRuntime[RuntimeStateT]``.

    No TorchRL environment, no ``env_td``, no :class:`~ehp_sn.contracts.task_environment.TaskEnvironmentAdapter`.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        backbone: QHaltingRolloutBackbone[ModelState],
        config: DeliberationQHaltingControllerConfig,
        runtime: TaskRuntime[RuntimeStateT],
    ) -> None:
        """Create a deliberation Q-halting controller.

        Args:
            backbone: The value-control backbone (task-agnostic).
            config: Controller configuration.
            runtime: Task-owned runtime implementing :class:`TaskRuntime`.
                Replaces the legacy :class:`~ehp_sn.contracts.task_step.TaskStepEvaluator`.
        """
        super().__init__(backbone=cast(Any, backbone), config=config)
        self._policy = CategoricalPolicy(config.policy)
        self._runtime = runtime

    @property
    def backbone(self) -> QHaltingRolloutBackbone[ModelState]:
        """Return the wrapped backbone typed to the actor-critic protocol."""
        return cast(QHaltingRolloutBackbone[ModelState], super().backbone)

    @property
    def runtime(self) -> TaskRuntime[RuntimeStateT]:
        """Return the injected task runtime."""
        return self._runtime

    def initial_state(  # -----------------------------------------------------
        self,
        batch_sample: Batch,
    ) -> DeliberationQHaltingRolloutState[ModelState, RuntimeStateT]:
        """Build an initial rollout state from a batch sample.

        Delegates initial observation and runtime state to
        :meth:`TaskRuntime.reset`.
        """
        reset_result = self._runtime.reset(batch_sample)
        slots = self.initial_slots(reset_result.observation)
        return DeliberationQHaltingRolloutState(
            model_state=slots.model_state,
            steps=slots.steps,
            halted=slots.halted,
            data=reset_result.observation,
            runtime_state=reset_result.state,
        )

    def step(  # ----------------------------------------------------------
        self,
        state: DeliberationQHaltingRolloutState[ModelState, RuntimeStateT],
        batch: Batch,
        *,
        allow_halt: bool = True,
        explore: bool = True,
        **options: Any,
    ) -> tuple[
        DeliberationQHaltingRolloutState[ModelState, RuntimeStateT],
        QHaltingInteractionRecord,
    ]:
        """Advance the controller by one step.

        Sequence:
            1. Refresh halted slots via :meth:`TaskRuntime.reset_slots`.
            2. Reset backbone state for halted rows.
            3. Run the backbone forward pass.
            4. Sample an action via :class:`~ehp_sn.policies.categorical.CategoricalPolicy`.
            5. Call :meth:`TaskRuntime.step` for reward, termination, and next observation.
            6. Compute ``done`` as ``(terminated | truncated)``.
            7. Emit :class:`~ehp_sn.controllers.contracts.actor_critic.QHaltingInteractionRecord`.

        The controller interacts with the runtime at exactly two points:
        ``reset_slots`` (step 1) and ``step`` (step 5).  No task-specific
        types cross this boundary.

        Args:
            state: Current rollout state.
            batch: Incoming batch used to refresh halted slots.
            allow_halt: If ``False``, the runtime's ``terminated`` signal is
                suppressed (zeroed out), preventing learned-halt termination
                from closing slots.  Task-owned ``truncated`` from the runtime
                always passes through regardless of this flag.
            explore: Passed to the categorical policy.
            halt_action: Optional action index treated as the halt action when
                applying ACT-style halt/continue semantics.
            max_halt_steps: Optional per-slot step budget used by ACT-style
                halt semantics to force halting.

        Returns:
            ``(new_state, record)``.
        """
        # 1. Refresh halted slots via runtime
        reset_result = self._runtime.reset_slots(
            reset_mask=state.halted,
            batch=batch,
            state=state.runtime_state,
        )
        data = reset_result.observation
        runtime_state = reset_result.state

        # 2. Reset backbone state for halted rows
        model_state = self.backbone.reset_state(state.halted, state.model_state)

        # 3. Backbone forward pass
        backbone_output, model_state = self.backbone(data, model_state)

        # 4. Advance step counters
        steps = self.advance_steps(state)

        # 5. Action selection (ACT collapse + categorical policy)
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

        # 6. Runtime step — single task-interaction point
        feedback = self._runtime.step(
            runtime_state, backbone_output.task, action, steps
        )

        # 7. Compute done flags
        if allow_halt:
            terminated = feedback.terminated
            truncated = feedback.truncated
            done = terminated | truncated
        else:
            # Suppress learned-halt termination only.  Task-owned truncated
            # (e.g. episode_horizon reached) passes through unchanged.
            terminated = torch.zeros_like(feedback.terminated)
            truncated = feedback.truncated
            done = truncated

        # 8. Build next state and emit record
        new_state = DeliberationQHaltingRolloutState(
            model_state=model_state,
            steps=steps,
            halted=done,
            data=feedback.next_observation,
            runtime_state=feedback.next_state,
        )
        record = QHaltingInteractionRecord(
            observation_used_for_decision=data,
            q_values=q_values,
            policy_logits=q_values,
            sampled_action=action,
            reward=feedback.reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            state_value=backbone_output.critic.state_value,
            task_output=backbone_output.task,
        )
        return new_state, record


# =============================================================================
__all__ = [
    "DeliberationQHaltingController",
    "DeliberationQHaltingControllerConfig",
    "DeliberationQHaltingRolloutState",
]
