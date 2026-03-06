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
from torch import Tensor
from torch.distributions import Categorical

from ehc_sn.types import Batch, Device
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class RLControllerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`RLController`.

    Intentionally minimal: the controller is responsible only for exploration
    probability.  Environment semantics (``max_steps``, ``halt_action``) live
    in the env config; temporal discounting (γ) lives in STR config.
    """

    exploration_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of suppressing an early halt during exploration.",
    )
    max_steps: int = Field(
        ...,
        ge=1,
        description="Maximum deliberation steps per slot before forced termination.",
    )


# =================================================================================================
class RLBackbone[BkState](Protocol):
    """ """

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
    ) -> Tuple[BkState, Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
        ...  # fmt: skip


# =================================================================================================
@dataclass
class RLState[ModelState](DetachMixin):
    """ """

    model_state: ModelState  # Recurrent state of the model (e.g. LSTM hidden states)
    steps: Tensor  # Per-slot step counter, shape: (B,)
    halted: Tensor  # Per-slot reset/done flag, shape: (B,)
    data: Dict[str, Tensor]  # Per-slot buffers that persist across steps until reset


# =================================================================================================
@dataclass
class RLOutput(DetachMixin):
    """ """

    logits: Tensor  # (B, S, V) — LM logits for supervised loss
    q_logits: Tensor  # (B, n_actions) — vmPFC Q-estimate logits for policy and value loss
    action: Tensor  # (B,) — sampled action index
    theta_cls: Tensor  # (B, D) — theta CLS features
    reward_hat: Tensor  # (B, 1) — STR reward prediction


# =================================================================================================
class RLController:
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, backbone: RLBackbone, config: RLControllerConfig,
    ) -> None:  # fmt: skip
        """ """
        self._backbone = backbone
        self._config = config

    @property
    def backbone(self) -> RLBackbone:
        """ """
        return self._backbone

    @property
    def config(self) -> RLControllerConfig:
        """ """
        return self._config

    def initial_state(  # -------------------------------------------------------------------------
        self, batch_sample: Batch,
    ) -> RLState:  # fmt: skip
        """ """
        B = batch_sample["inputs"].shape[0]
        device = batch_sample["inputs"].device
        return RLState(
            model_state=self._backbone.init_state(B),
            steps=torch.zeros((B,), dtype=torch.int32, device=device),
            halted=torch.ones((B,), dtype=torch.bool, device=device),
            data={k: torch.empty_like(v) for k, v in batch_sample.items()},
        )

    def step(  # ----------------------------------------------------------------------------------
        self, state: RLState, batch: Batch, *,
        allow_halt: bool = True, explore: bool = True,
    ) -> Tuple[RLState, RLOutput]:  # fmt: skip
        """ """
        data = self.refresh_slot_data(batch, state)
        model_state = self._backbone.reset_state(state.halted, state.model_state)
        model_state, logits, theta_cls, q_logits, r_hat = self._backbone(data["inputs"], model_state)

        steps = torch.where(state.halted, torch.zeros_like(state.steps), state.steps) + 1
        action, done = self._select_action_and_done(q_logits, steps, allow_halt, explore)

        state = RLState(model_state=model_state, steps=steps, halted=state.halted, data=data)
        output = RLOutput( logits=logits, q_logits=q_logits, action=action, theta_cls=theta_cls, reward_hat=r_hat)  # fmt: skip

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
        self, q_values: Tensor, steps: Tensor, allow_halt: bool, explore: bool,
    ) -> Tuple[Tensor, Tensor]:  # fmt: skip
        """ """
        action = Categorical(logits=q_values.detach()).sample()  # (B,)
        done = ...  # FIXME: Probably call the env and use termination signal

        return action, done


# =================================================================================================
__all__ = ["RLController", "RLState", "RLOutput", "RLBackbone", "RLControllerConfig"]
