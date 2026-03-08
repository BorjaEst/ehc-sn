""" """

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# ==================================================================================================
class ACTControllerConfig(BaseModel, extra="forbid"):
    """ """

    exploration_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Exploration probability for deliberation.",
    )
    max_steps: int = Field(
        ...,
        ge=1,
        description="Maximum deliberation steps per slot.",
    )
    done_action: int = Field(
        default=0,
        ge=0,
        description="Action index that terminates a slot when selected.",
    )


# =================================================================================================
class ACTBackbone[BkState](Protocol):
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
    ) -> Tuple[BkState, Tuple[Tensor, ...], Tensor]:  # fmt: skip
        ...  # fmt: skip


# =================================================================================================
@dataclass
class ACTState[ModelState](DetachMixin):
    """ """

    model_state: ModelState  # Recurrent state of the model (e.g. LSTM hidden states)
    steps: Tensor  # Per-slot step counter, shape: (B,)
    halted: Tensor  # Per-slot reset/done flag, shape: (B,)
    data: Dict[str, Tensor]  # Per-slot buffers that persist across steps until reset


# =================================================================================================
@dataclass
class ACTOutput(DetachMixin):
    """ """

    logits: Tuple[Tensor, ...]  # Tuple of (B, S, V) LM logits for supervised loss
    theta_cls: Tensor  # (B, D) — theta CLS features
    action: Tensor  # (B,) selected action indices for this step
    target_q: Tensor | None = None  # TD(0) bootstrap Q-target, shape: (B,). None outside training.


# =================================================================================================
class ACTController:
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, backbone: ACTBackbone,  config: ACTControllerConfig,
    ) -> None:  # fmt: skip
        """ """
        self._backbone = backbone
        self._config = config

    @property
    def backbone(self) -> ACTBackbone:
        """ """
        return self._backbone

    @property
    def config(self) -> ACTControllerConfig:
        """ """
        return self._config

    def initial_state(  # -------------------------------------------------------------------------
        self, batch_sample: Batch
    ) -> ACTState:  # fmt: skip
        """ """
        batch_size, device = batch_sample["inputs"].shape[0], batch_sample["inputs"].device
        return ACTState(  # FIXME: We need to replace batch_dict by observations and labels
            model_state=self.backbone.init_state(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            data={k: torch.empty_like(v) for k, v in batch_sample.items()},
        )

    def step(  # ----------------------------------------------------------------------------------
        self, state: ACTState, batch: Batch,
        allow_halt: bool = True, explore: bool = True, td_target: bool = True,
    ) -> Tuple[ACTState, ACTOutput]:  # fmt: skip
        """ """
        data = self.refresh_slot_data(batch, state)
        model_state = self.backbone.reset_state(state.halted, state.model_state)
        model_state, logits, theta_cls = self.backbone(data["inputs"], model_state)

        steps = torch.where(state.halted, 0, state.steps) + 1
        action, done = self._select_action_and_done(logits, steps, allow_halt, explore)

        state = ACTState(model_state=model_state, steps=steps, halted=done, data=data)
        output = ACTOutput(logits=logits, theta_cls=theta_cls, action=action)

        # TD(0) bootstrap target for the Q-head.
        if td_target and self._config.max_steps > 1:
            output.target_q = self.compute_td_target(data, model_state, steps)

        return state, output

    def compute_td_target(  # ---------------------------------------------------------------------
        self, data: Dict[str, Tensor], model_state: Any, steps: Tensor,
    ) -> Tensor:  # fmt: skip
        """ """
        with torch.no_grad():
            _, _, next_q = self.backbone(data["inputs"], model_state)
        is_last_step = steps >= self.config.max_steps

        # At last step: forced done → target is Q(done_action). Otherwise: max over all actions.
        done_action = self._config.done_action
        target = torch.where(is_last_step, next_q[..., done_action], next_q.max(dim=-1).values)

        return torch.sigmoid(target)

    def refresh_slot_data(  # ---------------------------------------------------------------------
        self, batch: Batch, state: ACTState
    ) -> Dict[str, Tensor]:  # fmt: skip
        """ """
        halted, data = state.halted, state.data
        return {
            k: torch.where(halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], data[k])
            for k in batch
        }  # fmt: skip

    def _select_action_and_done(  # ---------------------------------------------------------------
        self, logits: list[Tensor], steps: Tensor, allow_halt: bool, explore: bool,
    ) -> Tuple[Tensor, Tensor]:  # fmt: skip
        """ """
        _logits_lm, logits_q, *_ = logits  # Unpack list of logits multiple heads
        config = self.config
        action = logits_q.detach().argmax(dim=-1)  # greedy over all actions (B,)
        done = steps >= config.max_steps

        if allow_halt:
            done = done | (action == config.done_action)

        if explore and (config.max_steps > 1):
            exploration_flag = torch.rand(steps.shape, device=steps.device) < config.exploration_prob
            min_steps = exploration_flag * torch.randint_like(steps, low=2, high=config.max_steps + 1)
            done = done & (steps >= min_steps)

        return action, done


# =================================================================================================
__all__ = ["ACTBackbone", "ACTControllerConfig", "ACTController", "ACTState", "ACTOutput"]
