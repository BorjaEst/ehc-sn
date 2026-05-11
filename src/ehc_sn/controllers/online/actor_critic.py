"""Online actor-critic (RL) rollout controller — canonical owner.

The RL controller owns rollout-state transitions, policy-driven action
selection, and environment stepping.  It emits one
:class:`~ehc_sn.controllers.contracts.actor_critic.ActorCriticInteractionRecord`
per step.

Neutral actor-critic contracts live in
:mod:`ehc_sn.controllers.contracts.actor_critic`.  RL-specific pieces kept
here: ``RLTaskRuntime``, ``RLRolloutState``, ``RLControllerConfig``,
``RLController``.

Canonical import path::

    from ehc_sn.controllers.online.actor_critic import (
        RLController, RLControllerConfig, RLTaskRuntime, RLRolloutState,
    )
    from ehc_sn.controllers.contracts.actor_critic import (
        ActorCriticInteractionRecord, ActorCriticBackboneOutput,
        ActorCriticPolicyOutput, ActorCriticCriticOutput, ActorCriticRolloutBackbone,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, cast

import torch
from pydantic import BaseModel, Field
from tensordict import TensorDictBase
from torch import Tensor
from torchrl.envs import EnvBase

from ehc_sn.controllers._base import BaseController, RolloutState
from ehc_sn.controllers._env_rollout import initial_env_reset
from ehc_sn.controllers.contracts.actor_critic import (
    ActorCriticBackboneOutput,
    ActorCriticCriticOutput,
    ActorCriticInteractionRecord,
    ActorCriticPolicyOutput,
    ActorCriticRolloutBackbone,
    OnlineBootstrapCarry,
)
from ehc_sn.policies._base import PolicyDecision
from ehc_sn.policies.categorical import CategoricalPolicy, CategoricalPolicyConfig, PolicyInput
from ehc_sn.types import Batch


# =============================================================================
class RLControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`RLController`."""

    policy: CategoricalPolicyConfig = Field(
        default_factory=CategoricalPolicyConfig,
        description="Configuration for the categorical action policy used to sample rollout actions.",
    )
    max_steps: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum deliberation steps per slot before forced termination.",
    )


# =============================================================================
class RLTaskRuntime(Protocol):
    """Task-owned environment TensorDict shaping used by :class:`RLController`."""

    def build_reset_td(self, batch: Batch) -> TensorDictBase: ...

    def build_env_step_td(
        self,
        env_td: TensorDictBase,
        *,
        reset_mask: Tensor,
        action: Tensor,
        task_output: object,
        data: Batch,
    ) -> TensorDictBase: ...

    def finalize_env_transition(
        self,
        previous_env_td: TensorDictBase,
        next_env_td: TensorDictBase,
        *,
        reset_mask: Tensor,
        action: Tensor,
        task_output: object,
        data: Batch,
    ) -> TensorDictBase: ...

    def extract_next_step_obs(self, carry: OnlineBootstrapCarry) -> Batch:
        """Extract the next-step observation batch from the post-step carry."""
        ...


# =============================================================================
@dataclass
class RLRolloutState[ModelState](RolloutState[ModelState]):
    """Controller carry/state for RL rollouts."""

    env_td: TensorDictBase  # Per-slot TensorDict for interacting with the environment


# =============================================================================
class RLController[ModelState](BaseController[ModelState, RLControllerConfig]):
    """Action sampler + rollout state manager for online actor-critic training."""

    def __init__(
        self,
        backbone: RLRolloutBackbone[ModelState],
        env: EnvBase,
        config: RLControllerConfig,
        runtime: RLTaskRuntime,
    ) -> None:
        """Create an RL controller."""
        super().__init__(backbone=cast(Any, backbone), config=config)
        self._env = env
        self._policy = CategoricalPolicy(config.policy)
        self._runtime = runtime

    @property
    def backbone(self) -> ActorCriticRolloutBackbone[ModelState]:
        """Return the wrapped RL backbone typed to the local protocol."""
        return cast(ActorCriticRolloutBackbone[ModelState], super().backbone)

    @property
    def environment(self) -> EnvBase:
        """Return the TorchRL environment used for stepping."""
        return self._env

    @property
    def runtime(self) -> RLTaskRuntime:
        """Return the task-owned environment TensorDict runtime."""
        return self._runtime

    def initial_state(
        self, batch_sample: Batch,
    ) -> RLRolloutState[ModelState]:
        """Build an initial rollout state from a batch sample."""
        reset_td, env_td = initial_env_reset(batch_sample, self.runtime.build_reset_td, self._env)
        slots = self.initial_slots(batch_sample)
        return RLRolloutState(
            model_state=slots.model_state, steps=slots.steps, halted=slots.halted,
            data=slots.data, env_td=env_td,
        )

    def step(
        self, state: RLRolloutState[ModelState], batch: Batch, *,
        allow_halt: bool = True, explore: bool = True, **_: Any,
    ) -> tuple[RLRolloutState[ModelState], ActorCriticInteractionRecord]:
        """Advance the controller by one step and emit an :class:`~ehc_sn.controllers.contracts.actor_critic.ActorCriticInteractionRecord`."""
        data = self.refresh_slot_data(batch, state)
        model_state = self.backbone.reset_state(state.halted, state.model_state)
        backbone_output, model_state = self.backbone(data, model_state)

        steps = self.advance_steps(state)
        action, done, env_td, policy_decision = self._select_action_and_done(
            backbone_output, steps, data, state.env_td, state.halted, allow_halt, explore,
        )
        reward = env_td["reward"]
        terminated = env_td["terminated"].squeeze(-1)
        truncated = env_td["truncated"].squeeze(-1)

        state = RLRolloutState(model_state=model_state, steps=steps, halted=done, data=data, env_td=env_td)
        record = ActorCriticInteractionRecord(
            observation_used_for_decision=data,
            policy_logits=backbone_output.policy.policy_logits,
            sampled_action=action,
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            value_estimate=backbone_output.critic.state_value,
            policy_decision=policy_decision,
            task_output=backbone_output.task,
        )
        return state, record

    def _select_action_and_done(
        self,
        backbone_output: ActorCriticBackboneOutput,
        steps: Tensor,
        data: Batch,
        env_td: TensorDictBase,
        reset_mask: Tensor,
        allow_halt: bool,
        explore: bool,
    ) -> tuple[Tensor, Tensor, TensorDictBase, PolicyDecision]:
        policy = backbone_output.policy
        logits = policy.policy_logits
        valid_action_mask = policy.valid_action_mask
        if valid_action_mask is None:
            valid_action_mask = torch.ones_like(logits, dtype=torch.bool)

        policy_step_count = env_td.get("step_count")
        if policy_step_count is not None and torch.any(reset_mask):
            step_mask = reset_mask.view((-1,) + (1,) * (policy_step_count.ndim - 1))
            policy_step_count = torch.where(step_mask, torch.zeros_like(policy_step_count), policy_step_count)

        policy_input = PolicyInput(
            valid_action_mask=valid_action_mask,
            step_count=policy_step_count,
            logits=logits,
        )
        policy_decision = self._policy(policy_input, explore=explore)
        action = policy_decision.action.to(torch.int64)

        env_step_td = self.runtime.build_env_step_td(
            env_td,
            reset_mask=reset_mask,
            action=action,
            task_output=backbone_output.task,
            data=data,
        )
        next_env_td = self._env.step(env_step_td)["next"]
        env_td = self.runtime.finalize_env_transition(
            env_td, next_env_td,
            reset_mask=reset_mask, action=action,
            task_output=backbone_output.task, data=data,
        )

        terminated = env_td["terminated"].squeeze(-1)
        truncated = env_td["truncated"].squeeze(-1)
        done = terminated | truncated
        if self.config.max_steps is not None and allow_halt:
            done = done | (steps >= self.config.max_steps)

        return action, done, env_td, policy_decision


# =============================================================================
__all__ = [
    "RLController",
    "RLControllerConfig",
    "RLRolloutState",
    "RLTaskRuntime",
]
