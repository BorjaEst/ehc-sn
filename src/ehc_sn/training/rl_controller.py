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

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import torch
from pydantic import BaseModel, Field
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torch.distributions import Categorical
from torchrl.envs import EnvBase

from ehc_sn.types import Batch, Device
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class RLControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`RLController`.

    Intentionally minimal: the controller is responsible only for exploration
    probability.  Environment semantics (``max_steps``, ``halt_action``) live
    in the env config; temporal discounting (γ) lives in STR config.
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
class RLBackbone[BkState](Protocol):
    """Backbone protocol expected by :class:`RLController`.

    The backbone is responsible for the actual neural forward pass and recurrent
    state management. The controller treats the backbone as a black box: it feeds
    token inputs, receives logits and a CLS summary, and resets state for halted
    slots.
    """

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int,
    ) -> BkState:  # fmt: skip
        ...  # fmt: skip

    def reset_state(  # ---------------------------------------------------------------------------
        self, reset_flag: Tensor, state: BkState,
    ) -> BkState:   # fmt: skip
        ...  # fmt: skip

    def __call__(  # ------------------------------------------------------------------------------
        self, inputs: Tensor, state: BkState | None = None,
    ) -> Tuple[BkState, Tuple[Tensor, ...], Tensor]:  # fmt: skip
        ...  # fmt: skip


# =================================================================================================
@dataclass
class RLState[ModelState](DetachMixin):
    """Controller carry/state for RL rollouts.

    This state is carried across controller steps and supports partial resets.

    Attributes:
        model_state: Recurrent model state (backbone-specific).
        steps: Per-slot step counter of shape ``(B,)``.
        halted: Per-slot reset/done flag of shape ``(B,)``.
        data: Per-slot buffers (e.g. inputs/labels) that persist until reset.
        env_td: TorchRL TensorDict representing the current environment state.
    """

    model_state: ModelState  # Recurrent state of the model (e.g. LSTM hidden states)
    steps: Tensor  # Per-slot step counter, shape: (B,)
    halted: Tensor  # Per-slot reset/done flag, shape: (B,)
    data: Dict[str, Tensor]  # Per-slot buffers that persist across steps until reset
    env_td: TensorDictBase  # Per-slot TensorDict for interacting with the environment


# =================================================================================================
@dataclass
class RLOutput(DetachMixin):
    """Outputs produced by one controller step.

    Attributes:
        logits: Backbone outputs (a tuple of tensors; interpretation depends on
            the model/controller contract).
        theta_cls: CLS feature vector of shape ``(B, D)`` used for tracing and
            downstream heads.
        action: Sampled action indices of shape ``(B,)``.
        reward: Reward returned by the environment of shape ``(B, 1)``.
    """

    logits: Tuple[Tensor, ...]  # Tuple of (B, S, V) LM logits for supervised loss
    theta_cls: Tensor  # (B, D) — theta CLS features
    action: Tensor  # (B,) selected action indices for this step
    reward: Tensor  # (B, 1) reward from the environment for this step


# =================================================================================================
class RLController:
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
        self, backbone: RLBackbone, env: EnvBase, config: RLControllerConfig,
    ) -> None:  # fmt: skip
        """Create a controller.

        Args:
            backbone: Model implementing :class:`RLBackbone`.
            env: TorchRL environment used to compute rewards/termination.
            config: Controller-specific configuration.
        """
        self._backbone = backbone
        self._env = env
        self._config = config

    @property
    def backbone(self) -> RLBackbone:
        """Return the wrapped backbone."""
        return self._backbone

    @property
    def environment(self) -> EnvBase:
        """Return the TorchRL environment used for stepping."""
        return self._env

    @property
    def config(self) -> RLControllerConfig:
        """Return the controller configuration."""
        return self._config

    def initial_state(  # -------------------------------------------------------------------------
        self, batch_sample: Batch,
    ) -> RLState:  # fmt: skip
        """Build an initial rollout state from a batch sample.

        This performs an environment reset using the provided batch contents.

        Args:
            batch_sample: Batch dict containing at least ``"inputs"`` and
                ``"labels"`` of shape ``(B, S)``.

        Returns:
            Initialized :class:`RLState` with fresh backbone state and env state.
        """
        B = batch_sample["inputs"].shape[0]
        device = batch_sample["inputs"].device

        # Reset env with initial data
        reset_td = TensorDict(
            {"inputs": batch_sample["inputs"], "labels": batch_sample["labels"]},
            batch_size=[B], device=device,
        )  # fmt: skip
        env_td = self._env.reset(reset_td)

        return RLState(
            model_state=self._backbone.init_state(B),
            steps=torch.zeros((B,), dtype=torch.int32, device=device),
            halted=torch.ones((B,), dtype=torch.bool, device=device),
            data={k: torch.empty_like(v) for k, v in batch_sample.items()},
            env_td=env_td,
        )

    def step(  # ----------------------------------------------------------------------------------
        self, state: RLState, batch: Batch, *,
        allow_halt: bool = True, explore: bool = True,
    ) -> Tuple[RLState, RLOutput]:  # fmt: skip
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
        model_state = self._backbone.reset_state(state.halted, state.model_state)
        model_state, logits, theta_cls = self._backbone(data["inputs"], model_state)

        steps = torch.where(state.halted, torch.zeros_like(state.steps), state.steps) + 1
        action, done, env_td = self._select_action_and_done(logits, steps, data["labels"], state.env_td, allow_halt, explore)  # fmt: skip
        reward = env_td["reward"]

        state = RLState(model_state=model_state, steps=steps, halted=state.halted, data=data, env_td=env_td)
        output = RLOutput(logits=logits, theta_cls=theta_cls, action=action, reward=reward)

        return state, output

    def refresh_slot_data(  # --------------------------------------------------------------------
        self, batch: Batch, state: RLState,
    ) -> Dict[str, Tensor]:  # fmt: skip
        """Replace data for halted slots with incoming batch data.

        For slots where ``state.halted`` is True, the new batch row is used.
        For still-active slots, the existing slot buffer is preserved.
        """
        halted, data = state.halted, state.data
        return {
            k: torch.where(halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], data[k])
            for k in batch
        }  # fmt: skip

    def _select_action_and_done(  # ---------------------------------------------------------------
        self, logits: list[Tensor], steps: Tensor, labels: Tensor, env_td: TensorDictBase, 
        allow_halt: bool, explore: bool,
    ) -> Tuple[Tensor, Tensor, TensorDictBase]:  # fmt: skip
        """Sample action, step env, apply exploration gating. Returns (action, done, env_td)."""
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
        env_td["logits"] = logits_lm.detach()  # (B, S, V)
        env_td["labels"] = labels  # (B, S)
        env_td = self._env.step(env_td)["next"]  # TorchRL convention

        terminated = env_td["terminated"].squeeze(-1)  # (B,)
        truncated = env_td["truncated"].squeeze(-1)  # (B,)
        done = terminated | truncated  # (B,)

        # 4. Auxiliary control over termination by controller max_steps
        if self.config.max_steps is not None and allow_halt:
            done = done | (steps >= self.config.max_steps)

        return action, done, env_td


# =================================================================================================
__all__ = ["RLController", "RLState", "RLOutput", "RLBackbone", "RLControllerConfig"]
