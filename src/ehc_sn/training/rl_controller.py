""" """

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeAlias

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.distributions import Categorical

import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.types import Batch, Device, Dtype
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class RLControllerConfig(BaseModel, extra="forbid"):
    """ """

    exploration_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Exploration probability for ACT halting.",
    )
    max_steps: int = Field(
        default=10,
        ge=1,
        description="Maximum number of deliberation steps before forced halt.",
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
    ) -> Tuple[BkState, Tensor, Tensor, Tensor]:  # fmt: skip
        ...  # fmt: skip


# =================================================================================================
class RPEController[RPEState](Protocol):
    """ """

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> RPEState:  # fmt: skip
        ...  # fmt: skip

    def reset_state(  # ---------------------------------------------------------------------------
        self, state: RPEState, reset_flag: Tensor,
    ) -> RPEState:  # fmt: skip
        ...  # fmt: skip

    def __call__(  # ------------------------------------------------------------------------------
        self, features: Tensor, state: RPEState | None = None,
    ) -> Tuple[Tensor, Tensor, RPEState]:  # fmt: skip
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
    q_values: Tensor  # (B, n_actions) — vmPFC Q-estimates
    action: Tensor  # (B,) — sampled action index
    theta_cls: Tensor  # (B, D) — theta CLS features (tracing/diagnostics)
    policy_logits: Tensor  # (B, A) — STR policy logits
    value: Tensor  # (B,) — V(s_t) from STR critic


# =================================================================================================
class RLController:
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, backbone: RLBackbone, rpe_controller: RPEController, config: RLControllerConfig,
    ) -> None:  # fmt: skip
        """ """
        self._backbone = backbone
        self._controller = rpe_controller
        self._config = config

    @property
    def backbone(self) -> RLBackbone:
        return self._backbone

    @property
    def rpe_controller(self) -> RPEController:
        return self._controller

    @property
    def config(self) -> RLControllerConfig:
        return self._config

    def initial_state(  # -------------------------------------------------------------------------
        self, batch_sample: Batch,
    ) -> RLState:  # fmt: skip
        """Create initial per-slot state (all slots marked halted = fresh start)."""
        B = batch_sample["inputs"].shape[0]
        device = batch_sample["inputs"].device
        return RLState(
            model_state=self._backbone.init_state(B),
            steps=torch.zeros((B,), dtype=torch.int32, device=device),
            halted=torch.ones((B,), dtype=torch.bool, device=device),  # force refresh on step 0
            prev_outcome=torch.zeros((B,), dtype=torch.float32, device=device),
            data={k: torch.empty_like(v) for k, v in batch_sample.items()},
        )

    def step(  # ----------------------------------------------------------------------------------
        self, state: RLState, batch: Batch, *,
        allow_halt: bool = True, explore: bool = True,
    ) -> Tuple[RLState, RLOutput]:  # fmt: skip
        """ """
        data = self.refresh_slot_data(batch, state)
        model_state = self._backbone.reset_state(state.halted, state.model_state)
        model_state, logits, theta_cls, q_values = self._backbone(data["inputs"], model_state)

        # STR: detach theta_cls for gradient isolation (REQ-006)
        features = theta_cls.detach()
        policy_logits, value, str_state = self._controller(features, model_state.str)
        model_state = dataclasses.replace(model_state, str=str_state)

        steps = torch.where(state.halted, 0, state.steps) + 1
        action, done = self._select_action_and_done(policy_logits, steps, allow_halt, explore)

        new_state = RLState(model_state=model_state, steps=steps, halted=done, data=data)
        output = RLOutput(
            logits=logits, policy_logits=policy_logits, value=value, action=action,
            theta_cls=theta_cls, q_values=q_values,
        )  # fmt: skip

        return new_state, output

    def refresh_slot_data(  # --------------------------------------------------------------------
        self, batch: Batch, state: RLState,
    ) -> Dict[str, Tensor]:  # fmt: skip
        """ """
        halted, data = state.halted, state.data
        return {
            k: torch.where(halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], data[k])
            for k in batch
        }  # fmt: skip

    def _select_action_and_done(  # ---------------------------------------------------------------
        self, policy_logits: Tensor, steps: Tensor, allow_halt: bool, explore: bool,
    ) -> Tuple[Tensor, Tensor]:  # fmt: skip
        dist = Categorical(logits=policy_logits.detach())
        action = dist.sample()  # (B,)
        done = steps >= self._config.max_steps  # forced truncation

        if allow_halt:
            done = done | (action == self._config.done_action)  # agent-chosen termination

        if explore and self._config.max_steps > 1:
            flag = torch.rand_like(steps.float()) < self._config.exploration_prob
            min_s = flag * torch.randint_like(steps, low=2, high=self._config.max_steps + 1)
            done = done & (steps >= min_s)

        return action, done


# =================================================================================================
__all__ = ["RLController", "RLState", "RLOutput", "RLBackbone", "RPEController", "RLControllerConfig"]
