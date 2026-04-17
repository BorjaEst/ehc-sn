"""RL rollout controller contracts and execution step wiring.

The RL controller owns rollout-state transitions, policy-driven action
selection, and environment stepping. Objective-layer code scores executed
steps later through explicit policy and critic readouts.
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
    """Policy payload consumed by the controller and actor-side objectives."""

    q_logits: Tensor
    valid_action_mask: Tensor | None


class RLCriticOutput(Protocol):
    """Critic payload consumed by objective heads and traces."""

    state_value: Tensor  # (B, 1), not "logits"


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
        action: Tensor,
        task_output: object,
        data: Batch,
    ) -> TensorDictBase: ...


# =============================================================================
@dataclass
class RLRolloutState[ModelState](RolloutState[ModelState]):
    """Controller carry/state for RL rollouts."""

    env_td: TensorDictBase  # Per-slot TensorDict for interacting with the environment


# =================================================================================================
@dataclass
class RLStepOutput(DetachMixin):
    """Outputs produced by one RL controller step.

    ``backbone_output`` carries all model-produced data; ``action`` and ``reward``
    are the only controller- and environment-produced facts added by the RL step.

    Consumers must read model data through ``backbone_output.task.*``,
    ``backbone_output.policy.*``, and ``backbone_output.critic.*`` rather than
    through flat accessors on this class.
    """

    backbone_output: RLBackboneOutput
    action: Tensor  # (B,) selected action indices for this step
    reward: Tensor  # (B, 1) reward from the environment for this step


# =================================================================================================
class RLController[ModelState](BaseController[ModelState, RLControllerConfig]):
    """Action sampler + rollout state manager for HRM v2.

    The controller:
        - maintains per-slot buffers across steps
        - resets backbone state for halted slots
        - samples actions from STR policy logits (via categorical sampling)
        - steps the TorchRL environment and returns rewards

    Termination semantics are owned by the environment (and optionally augmented
    by ``config.max_steps``).
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
        reset_td = self.runtime.build_reset_td(batch_sample)
        env_td = self._env.reset(reset_td)

        slots = self.initial_slots(batch_sample)
        return RLRolloutState(
            model_state=slots.model_state, steps=slots.steps, halted=slots.halted,
            data=slots.data, env_td=env_td,
        )  # fmt: skip

    def step(  # ----------------------------------------------------------------------------------
        self, state: RLRolloutState[ModelState], batch: Batch, *,
        allow_halt: bool = True, explore: bool = True, **_: Any,
    ) -> tuple[RLRolloutState[ModelState], RLStepOutput]:  # fmt: skip
        """Advance the controller by one step.

        The step:
            1) refreshes per-slot data for halted rows
            2) resets the backbone state for halted rows
            3) runs the backbone forward pass
            4) samples an action and steps the environment

        Args:
            state: Current rollout state.
            batch: Incoming batch to use as a source of fresh rows for resets.
            allow_halt: Whether controller-side max-step halting is enabled.
            explore: Whether to apply exploration overrides.

        Returns:
            ``(new_state, output)``.
        """
        data = self.refresh_slot_data(batch, state)
        model_state = self.backbone.reset_state(state.halted, state.model_state)
        model_state, backbone_output = self.backbone(data, model_state)

        steps = self.advance_steps(state)
        action, done, env_td = self._select_action_and_done(backbone_output, steps, data, state.env_td, allow_halt, explore)  # fmt: skip
        reward = env_td["reward"]

        state = RLRolloutState(model_state=model_state, steps=steps, halted=done, data=data, env_td=env_td)
        output = RLStepOutput(backbone_output=backbone_output, action=action, reward=reward)

        return state, output

    def _select_action_and_done(
        self,
        backbone_output: RLBackboneOutput,
        steps: Tensor,
        data: Batch,
        env_td: TensorDictBase,
        allow_halt: bool,
        explore: bool,
    ) -> tuple[Tensor, Tensor, TensorDictBase]:
        policy = backbone_output.policy
        logits_q = policy.q_logits
        valid_action_mask = policy.valid_action_mask
        if valid_action_mask is None:
            valid_action_mask = torch.ones_like(logits_q, dtype=torch.bool)

        policy_input = PolicyInput(
            valid_action_mask=valid_action_mask,
            step_count=env_td.get("step_count"),
            logits=logits_q,
        )
        action = self._policy(policy_input, explore=explore).action.to(torch.int64)

        env_td = self.runtime.build_env_step_td(
            env_td,
            action=action,
            task_output=backbone_output.task,
            data=data,
        )
        env_td = self._env.step(env_td)["next"]

        terminated = env_td["terminated"].squeeze(-1)
        truncated = env_td["truncated"].squeeze(-1)
        done = terminated | truncated
        if self.config.max_steps is not None and allow_halt:
            done = done | (steps >= self.config.max_steps)

        return action, done, env_td


# =================================================================================================
__all__ = [
    "RLController",
    "RLControllerConfig",
    "RLTaskRuntime",
    "RLStepOutput",
    "RLBackboneOutput",
    "RLPolicyOutput",
    "RLCriticOutput",
    "RLRolloutBackbone",
    "RLRolloutState",
]
