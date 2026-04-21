"""RL rollout controller contracts and execution step wiring.

The RL controller owns rollout-state transitions, policy-driven action
selection, and environment stepping. It emits one ``InteractionRecord`` per
step. Learners own all TD(0) and GAE post-processing; objectives consume
fully materialized ``InteractionRecord`` batches without parsing controller
internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, cast

import torch
from pydantic import BaseModel, Field
from tensordict import TensorDictBase
from torch import Tensor
from torchrl.envs import EnvBase

from ehc_sn.controllers._base import BaseController, RolloutBackbone, RolloutState
from ehc_sn.controllers._env_rollout import initial_env_reset
from ehc_sn.policies._base import PolicyDecision
from ehc_sn.policies.categorical import CategoricalPolicy, CategoricalPolicyConfig, PolicyInput
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class RLControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`RLController`.

    Attributes:
        policy: Configuration for the categorical action policy used to sample rollout actions.
        max_steps: Hard cap on deliberation steps before forced termination.
            ``None`` delegates all termination to the environment.
    """

    policy: CategoricalPolicyConfig = Field(
        default_factory=CategoricalPolicyConfig,
        description="Configuration for the categorical action policy used to sample rollout actions.",
    )
    max_steps: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum deliberation steps per slot before forced termination.",
    )


class RLPolicyOutput(Protocol):
    """Policy payload consumed by the controller's action-selection step."""

    policy_logits: Tensor
    valid_action_mask: Tensor | None


class RLCriticOutput(Protocol):
    """Critic payload carried by the backbone output (not exposed on InteractionRecord)."""

    state_value: Tensor  # (B, 1)


class RLBackboneOutput(Protocol):
    """Named RL backbone output split into task, policy, and critic surfaces."""

    task: object
    policy: RLPolicyOutput
    critic: RLCriticOutput | None


# =============================================================================
class RLRolloutBackbone[ModelState](RolloutBackbone[ModelState, RLBackboneOutput], Protocol):
    """Backbone protocol expected by :class:`RLController`."""


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

    def extract_next_step_obs(self, carry: Any) -> Batch:
        """Extract the next-step observation batch from the post-step carry.

        Called by the learner to obtain the input needed for TD(0) bootstrap
        value computation. The carry is the :class:`RLRolloutState` produced
        after the most recent controller step.
        """
        ...


# =============================================================================
@dataclass
class RLRolloutState[ModelState](RolloutState[ModelState]):
    """Controller carry/state for RL rollouts."""

    env_td: TensorDictBase  # Per-slot TensorDict for interacting with the environment


# =================================================================================================
@dataclass
class InteractionRecord(DetachMixin):
    """Explicit actor-critic interaction record emitted by :class:`RLController` per step.

    Learners receive this directly and own TD(0) / GAE post-processing.
    Objectives consume ``InteractionRecord`` fields without parsing controller
    internals (no ``backbone_output`` access).

    ``observation_used_for_decision`` is the exact ``data`` dict passed to the
    backbone forward pass — threaded explicitly, not reconstructed from carry.
    """

    observation_used_for_decision: dict[str, Tensor]  # exact batch used for the model step
    policy_logits: Tensor  # (B, A) actor-head policy logits
    sampled_action: Tensor  # (B,) sampled action indices
    reward: Tensor  # (B, 1) task-finalized reward for the executed step
    done: Tensor  # (B,) combined termination flag (terminated | truncated | max_steps)
    terminated: Tensor  # (B,) episode terminated flag
    truncated: Tensor  # (B,) episode truncated flag
    value_estimate: Tensor  # (B, 1) critic state value
    policy_decision: PolicyDecision  # rollout-time policy decision (log_prob, entropy)
    task_output: object | None = None  # optional task-side model output


# =================================================================================================
class RLController[ModelState](BaseController[ModelState, RLControllerConfig]):
    """Action sampler + rollout state manager for HRM v2 actor-critic training.

    The controller:
        - maintains per-slot buffers across steps
        - resets backbone state for halted slots
        - samples actions from the actor policy logits (via categorical sampling)
        - steps the TorchRL environment and returns rewards
        - emits one :class:`InteractionRecord` per step

    Termination semantics are owned by the environment (and optionally augmented
    by ``config.max_steps``). The controller does not compute losses.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self,
        backbone: RLRolloutBackbone[ModelState],
        env: EnvBase,
        config: RLControllerConfig,
        runtime: RLTaskRuntime,
    ) -> None:  # fmt: skip
        """Create a controller.

        Args:
            backbone: Model implementing :class:`RLRolloutBackbone`.
            env: TorchRL environment used to compute rewards/termination.
            config: Controller-specific configuration.
            runtime: Task-owned environment TensorDict runtime.
        """
        super().__init__(backbone=cast(Any, backbone), config=config)
        self._env = env
        self._policy = CategoricalPolicy(config.policy)
        self._runtime = runtime

    @property
    def backbone(self) -> RLRolloutBackbone[ModelState]:
        """Return the wrapped RL backbone typed to the local protocol."""
        return cast(RLRolloutBackbone[ModelState], super().backbone)

    @property
    def environment(self) -> EnvBase:
        """Return the TorchRL environment used for stepping."""
        return self._env

    @property
    def runtime(self) -> RLTaskRuntime:
        """Return the task-owned environment TensorDict runtime."""
        return self._runtime

    def initial_state(  # -------------------------------------------------------------------------
        self, batch_sample: Batch,
    ) -> RLRolloutState[ModelState]:  # fmt: skip
        """Build an initial rollout state from a batch sample.

        This performs an environment reset using the provided batch contents.

        Args:
            batch_sample: Batch dict containing at least ``"input_ids"`` and
                ``"labels"`` of shape ``(B, S)``.

        Returns:
            Initialized :class:`RLRolloutState` with fresh backbone state and env state.
        """
        reset_td, env_td = initial_env_reset(batch_sample, self.runtime.build_reset_td, self._env)

        slots = self.initial_slots(batch_sample)
        return RLRolloutState(
            model_state=slots.model_state, steps=slots.steps, halted=slots.halted,
            data=slots.data, env_td=env_td,
        )  # fmt: skip

    def step(  # ----------------------------------------------------------------------------------
        self, state: RLRolloutState[ModelState], batch: Batch, *,
        allow_halt: bool = True, explore: bool = True, **_: Any,
    ) -> tuple[RLRolloutState[ModelState], InteractionRecord]:  # fmt: skip
        """Advance the controller by one step and emit an :class:`InteractionRecord`.

        The step:
            1) refreshes per-slot data for halted rows
            2) resets the backbone state for halted rows
            3) runs the backbone forward pass
            4) samples an action, steps the environment, and finalizes the
               reward-bearing transition through the task runtime

        Args:
            state: Current rollout state.
            batch: Incoming batch to use as a source of fresh rows for resets.
            allow_halt: Whether controller-side max-step halting is enabled.
            explore: Whether exploration is enabled (passed to the policy).

        Returns:
            ``(new_state, record)``.
        """
        data = self.refresh_slot_data(batch, state)
        model_state = self.backbone.reset_state(state.halted, state.model_state)
        backbone_output, model_state = self.backbone(data, model_state)

        steps = self.advance_steps(state)
        action, done, env_td, policy_decision = self._select_action_and_done(
            backbone_output,
            steps,
            data,
            state.env_td,
            state.halted,
            allow_halt,
            explore,
        )  # fmt: skip
        reward = env_td["reward"]
        terminated = env_td["terminated"].squeeze(-1)
        truncated = env_td["truncated"].squeeze(-1)

        if backbone_output.critic is None:
            raise RuntimeError(
                "Actor-critic training requires a critic output; backbone returned critic=None. "
                "Ensure the model has a critic head configured."
            )
        state = RLRolloutState(model_state=model_state, steps=steps, halted=done, data=data, env_td=env_td)
        record = InteractionRecord(
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
        backbone_output: RLBackboneOutput,
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
            env_td,
            next_env_td,
            reset_mask=reset_mask,
            action=action,
            task_output=backbone_output.task,
            data=data,
        )

        terminated = env_td["terminated"].squeeze(-1)
        truncated = env_td["truncated"].squeeze(-1)
        done = terminated | truncated
        if self.config.max_steps is not None and allow_halt:
            done = done | (steps >= self.config.max_steps)

        return action, done, env_td, policy_decision


# =================================================================================================
__all__ = [
    "RLController",
    "RLControllerConfig",
    "RLTaskRuntime",
    "InteractionRecord",
    "RLBackboneOutput",
    "RLPolicyOutput",
    "RLCriticOutput",
    "RLRolloutBackbone",
    "RLRolloutState",
]
