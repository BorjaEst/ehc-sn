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
    halt_max_steps: int = Field(
        default=10,
        ge=1,
        description="Maximum number of deliberation steps before forced halt.",
    )
    gamma: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Discount factor for TD learning. Default 1.0 (no discount) for optimal stopping "
            "with additive ponder costs. Reduce if training oscillates."
        ),
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

    model_state: ModelState  # Recurrent backbone state
    steps: Tensor  # (B,) int32 — per-slot deliberation step counter
    halted: Tensor  # (B,) bool  — per-slot done / reset flag
    prev_outcome: Tensor  # (B,) float32 — outcome from previous step (improvement baseline)
    data: Dict[str, Tensor]  # Per-slot buffered "inputs" and "labels"


# =================================================================================================
@dataclass
class RLOutput(DetachMixin):
    """ """

    logits: Tensor  # LM logits (B, S, vocab) — for supervised loss
    policy_logits: Tensor  # STR policy logits (B, 2) — for actor and entropy losses
    value: Tensor  # V(s_t) (B,) — for critic loss and TD error
    next_value: Tensor  # V(s_{t+1}) (B,), produced under no_grad — TD bootstrap
    action: Tensor  # Sampled action (B,): 0=halt, 1=continue
    theta_cls: Tensor  # Raw (non-detached) z_H[:,0] — tracing only, not used in loss
    q_values: Tensor  # vmPFC Q-estimates (B, n_actions) — for vmPFC TD-Q loss
    next_q_max: Tensor  # max Q for next state (B,), under no_grad — vmPFC TD target


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
        """Run one RLController step.

        Steps performed (in order):
            1. Refresh per-slot data for halted slots; reset prev_outcome for new episodes.
            2. Reset backbone recurrent state for halted slots.
            3. Backbone forward: new state, LM logits, theta_cls, vmPFC q_values.
            4. STR forward on **detached** theta_cls: policy_logits, value.
            5. Action selection: Categorical sample (explore=True) or argmax (False).
            6. Compute done mask (max steps OR halt action when allow_halt=True).
            7. Bootstrap V(s_{t+1}) and max Q(s_{t+1}) under ``torch.no_grad()``.

        Args:
            state: Current per-slot state.
            batch: Current batch of data (inputs and labels).
            allow_halt: If True, halt actions can terminate slots; otherwise, only max steps.
            explore: If True, sample stochastically from the policy; otherwise, take argmax.

        Returns:
            new_state: Updated per-slot state after this step.
            output: RLOutput containing all relevant tensors for loss computation and tracing.
        """

        # TODO: integrate and use exploration_prob
        # TODO:

        # 1. Refresh slot data; reset prev_outcome to 0.0 for newly refreshed slots
        data = self.refresh_slot_data(batch, state)
        prev_outcome = torch.where(state.halted, torch.zeros_like(state.prev_outcome), state.prev_outcome)

        # 2. Reset backbone recurrent state for halted slots
        model_state = self._backbone.reset_state(state.halted, state.model_state)

        # 3. Backbone forward (gradients flow for supervised loss via logits; q_values → vmPFC)
        new_model_state, logits, theta_cls, q_values = self._backbone(data["inputs"], model_state)
        q_values = q_values.detach()  # detach q_values to prevent vmPFC gradients flowing into PFC

        # 4. STR forward — features MUST be detached (REQ-006: no RL grads into PFC)
        features = theta_cls.detach()  # (B, D)
        policy_logits, value, str_state = self._controller(features, new_model_state.str)
        new_model_state = dataclasses.replace(new_model_state, str=str_state)

        # 5. Action selection
        if explore:
            action = Categorical(logits=policy_logits).sample()  # stochastic (B,)
        else:
            action = policy_logits.argmax(dim=-1)  # deterministic (B,)

        # 6. Step counter + done mask
        steps = torch.where(state.halted, torch.zeros_like(state.steps), state.steps) + 1
        done = steps >= self.config.halt_max_steps
        if allow_halt:
            done = done | (action == HALT_ACTION)

        # 7. Bootstrap V(s_{t+1}) and max Q(s_{t+1}) — no_grad; zero for done slots
        with torch.no_grad():
            _, _, next_theta_cls, next_q_values = self._backbone(data["inputs"], new_model_state)
            _, next_value, _ = self._controller(next_theta_cls.detach(), new_model_state.str)
            next_value = torch.where(done, torch.zeros_like(next_value), next_value)
            next_q_max = next_q_values.max(dim=-1).values
            next_q_max = torch.where(done, torch.zeros_like(next_q_max), next_q_max)

        new_state = RLState(
            model_state=new_model_state,
            steps=steps,
            halted=done,
            prev_outcome=prev_outcome,
            data=data,
        )
        output = RLOutput(
            logits=logits,
            policy_logits=policy_logits,
            value=value,
            next_value=next_value,
            action=action,
            theta_cls=theta_cls,
            q_values=q_values,
            next_q_max=next_q_max,
        )
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
