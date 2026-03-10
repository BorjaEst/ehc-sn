"""RL controller: agent forward pass, action sampling, and recurrent state management.

The controller is action-semantic-agnostic: it samples an action from the STR
policy but does not interpret what the action means.  Termination, truncation,
and reward are decided by the environment (inline in the loss head).

Responsibilities:
    - Refresh per-slot data for halted slots (data[i] ← batch[i] when halted[i]).
    - Reset backbone state for halted slots.
    - Run the model forward pass: backbone → STR (detached theta).
    - Sample action from Categorical(policy_logits).
    - Track per-slot step counters.

What this module does NOT do:
    - Decide termination / truncation (env / loss head does this).
    - Compute reward (env does this).
    - Bootstrap V(s') or max Q(s') (loss head does this).
    - Apply exploration overrides on done (loss head does this).
    - Own γ (STR does this).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple

import torch
from pydantic import BaseModel, Field
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torch.distributions import Categorical
from torchrl.envs import EnvBase

from ehc_sn.controllers._base import BaseController, RolloutBackbone, RolloutState
from ehc_sn.types import Batch, Device
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class RLControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`RLController`.

    Attributes:
        exploration_prob: Probability of suppressing an early halt during
            exploration.  ``None`` disables random action substitution.
        max_steps: Hard cap on deliberation steps before forced termination.
            ``None`` delegates all termination to the environment.
    """

    exploration_prob: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Probability of suppressing an early halt during exploration.",
    )
    max_steps: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum deliberation steps per slot before forced termination.",
    )


# =================================================================================================
class RLRolloutBackbone[ModelState, ModelOutput](RolloutBackbone[ModelState, ModelOutput], Protocol):
    """Backbone protocol expected by :class:`RLController`."""


# =================================================================================================
@dataclass
class RLRolloutState[ModelState](RolloutState[ModelState]):
    """Controller carry/state for RL rollouts."""

    env_td: TensorDictBase  # Per-slot TensorDict for interacting with the environment


# =================================================================================================
@dataclass
class RLOutput(DetachMixin):
    """Outputs produced by one controller step.

    Named property accessors (``lm_logits``, ``q_logits``, ``value_logits``) are
    the preferred read path for heads.  Direct positional reads via ``logits[N]``
    are deprecated in head code and will be removed once all heads are migrated.
    """

    logits: Tuple[Tensor, ...]  # (lm_logits, q_logits, value_logits, ...); prefer named properties in heads
    theta_cls: Tensor  # (B, D) — theta CLS features
    action: Tensor  # (B,) selected action indices for this step
    reward: Tensor  # (B, 1) reward from the environment for this step

    @property
    def lm_logits(self) -> Tensor:
        """LM head logits, shape ``(B, S, V)``."""
        return self.logits[0]

    @property
    def q_logits(self) -> Tensor:
        """Action policy logits (STR), shape ``(B, A)``."""
        return self.logits[1]

    @property
    def value_logits(self) -> Tensor:
        """Value / reward-prediction logits, shape ``(B, 1)``."""
        return self.logits[2]


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
        self, backbone: RLRolloutBackbone, env: EnvBase, config: RLControllerConfig,
    ) -> None:  # fmt: skip
        """Create a controller.

        Args:
            backbone: Model implementing :class:`RLRolloutBackbone`.
            env: TorchRL environment used to compute rewards/termination.
            config: Controller-specific configuration.
        """
        super().__init__(backbone=backbone, config=config)
        self._env = env

    @property
    def environment(self) -> EnvBase:
        """Return the TorchRL environment used for stepping."""
        return self._env

    def initial_state(  # -------------------------------------------------------------------------
        self, batch_sample: Batch,
    ) -> RLRolloutState[ModelState]:  # fmt: skip
        """Build an initial rollout state from a batch sample.

        This performs an environment reset using the provided batch contents.

        Args:
            batch_sample: Batch dict containing at least ``"inputs"`` and
                ``"labels"`` of shape ``(B, S)``.

        Returns:
            Initialized :class:`RLRolloutState` with fresh backbone state and env state.
        """
        B = batch_sample["inputs"].shape[0]
        device = batch_sample["inputs"].device

        # Reset env with initial data
        reset_td = TensorDict(
            {"inputs": batch_sample["inputs"], "labels": batch_sample["labels"]},
            batch_size=[B], device=device,
        )  # fmt: skip
        env_td = self._env.reset(reset_td)

        slots = self.initial_slots(batch_sample)
        return RLRolloutState(
            model_state=slots.model_state, steps=slots.steps, halted=slots.halted,
            data=slots.data, env_td=env_td,
        )  # fmt: skip

    def step(  # ----------------------------------------------------------------------------------
        self, state: RLRolloutState[ModelState], batch: Batch, *,
        allow_halt: bool = True, explore: bool = True, **_: Any,
    ) -> Tuple[RLRolloutState[ModelState], RLOutput]:  # fmt: skip
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
        model_state, logits, theta_cls = self.backbone(data, model_state)

        steps = self.advance_steps(state)
        action, done, env_td = self._select_action_and_done(logits, steps, data, state.env_td, allow_halt, explore)  # fmt: skip
        reward = env_td["reward"]

        state = RLRolloutState(model_state=model_state, steps=steps, halted=done, data=data, env_td=env_td)
        output = RLOutput(logits=logits, theta_cls=theta_cls, action=action, reward=reward)

        return state, output

    def _select_action_and_done(  # ---------------------------------------------------------------
        self, logits: list[Tensor], steps: Tensor, data: Batch, env_td: TensorDictBase, 
        allow_halt: bool, explore: bool,
    ) -> Tuple[Tensor, Tensor, TensorDictBase]:  # fmt: skip
        """Sample action, step env, and apply exploration gating."""
        logits_lm, logits_q, logits_r, *_ = logits  # Unpack list of logits multiple heads
        n_actions = self._env.action_spec["action"].shape[-1]

        # 1. Sample action from Q-logits (controller is action-agnostic)
        action = Categorical(logits=logits_q.detach()).sample()  # (B,)

        # 2. Auxiliary exploration controlled by exploration_prob
        if self.config.exploration_prob is not None and explore:
            explore_flag = torch.rand(steps.shape, device=steps.device) < self.config.exploration_prob
            action = torch.where(explore_flag, torch.randint_like(action, low=0, high=n_actions), action)

        # 3. Step the environment — env owns reward + termination semantics
        env_td = env_td.clone()
        env_td["action"] = action.unsqueeze(-1)  # (B, 1)
        env_td["logits"] = logits_lm.detach().to(torch.float32)  # (B, S, V)
        env_td["labels"] = data["labels"]  # (B, S)
        env_td = self._env.step(env_td)["next"]  # TorchRL convention

        terminated = env_td["terminated"].squeeze(-1)  # (B,)
        truncated = env_td["truncated"].squeeze(-1)  # (B,)
        done = terminated | truncated  # (B,)

        # 4. Auxiliary control over termination by controller max_steps
        if self.config.max_steps is not None and allow_halt:
            done = done | (steps >= self.config.max_steps)

        return action, done, env_td


# =================================================================================================
__all__ = ["RLController", "RLRolloutState", "RLOutput", "RLRolloutBackbone", "RLControllerConfig"]
