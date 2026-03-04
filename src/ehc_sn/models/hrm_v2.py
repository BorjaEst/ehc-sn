""" """

import math
from dataclasses import dataclass
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.optim import Optimizer

import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.data.schema import CHANNEL_SOLUTION
from ehc_sn.data.transforms import channels_to_grid
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.modules.pfc import PFCModel, PFCSettings, PFCState
from ehc_sn.modules.str import STRModel, STRSettings
from ehc_sn.rollouts.collect import TraceCollector, TraceField, TraceSpec, TraceValue
from ehc_sn.rollouts.trace_tree import TraceTree
from ehc_sn.training.buffers import FifoBuffer
from ehc_sn.training.optim import AdamATan2, AdamATan2Config
from ehc_sn.training.partial_reset import PartialResetBatchAssembler
from ehc_sn.training.schedules import CosineAnnealingLRWithWarmup, SchedulerConfig, SequentialLR
from ehc_sn.training.step_loop import StepContext, StepLoop
from ehc_sn.types import Device, Dtype
from ehc_sn.utils import trunc_normal_init_

# Community-standard map-style batch: plain dict returned by MazeDataset / DataLoader.
Batch: TypeAlias = Dict[str, Tensor]

# HRM-private: solution-path token, not part of the canonical SEM vocabulary.
O_ID: int = 5

# Token label ignored by supervised loss (padding / non-supervised positions).
IGNORE_LABEL_ID: int = -100

# STR action indices (must match STRModel output dim; 0=halt, 1=continue).
HALT_ACTION: int = 0
CONTINUE_ACTION: int = 1


# =================================================================================================
class ModelSettings_V2(BaseModel, extra="forbid"):
    """ """

    pfc: PFCSettings = Field(..., description="Settings for the core PFC model architecture.")
    str: STRSettings = Field(..., description="Settings for the STR actor-critic architecture.")

    vocab_size: int = Field(
        ...,
        ge=1,
        description="Vocabulary size for token embeddings and LM head.",
    )

    @property
    def seq_length(self) -> int:
        """Convenience property to access sequence length from the PFC settings."""
        return self.pfc.seq_length

    @property
    def hidden_size(self) -> int:
        """Convenience property to access hidden size from the PFC settings."""
        return self.pfc.reasoning_h.cortex.embedding_dim

    @property
    def embedding_scale(self) -> float:
        """Convenience property for scaling embeddings to maintain variance."""
        # scale by 1/sqrt(2) to maintain forward variance
        return 0.707106781 * math.sqrt(self.hidden_size)

    @property
    def init_std(self) -> float:
        """Convenience property for standard deviation of truncated normal initialization."""
        return 1.0 / math.sqrt(self.hidden_size)


# =================================================================================================
class ModelConfig_HRM_V2(BaseModel, extra="forbid"):
    """ """

    model: ModelSettings_V2 = Field(
        ...,
        description="PFC backbone configuration.",
    )

    # ~~ RL parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    gamma: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "TD discount factor.  Default 1.0 (no discount) is correct for optimal stopping "
            "with additive ponder costs.  Reduce to 0.99 only if training oscillates."
        ),
    )
    rl_warmup_steps: int = Field(
        default=5000,
        ge=0,
        description=(
            "Number of optimiser steps during which only the supervised optimizer trains. "
            "STR and vmPFC are frozen; allow_halt=False forces full deliberation. "
            "Prevents 'halt immediately' collapse before PFC representations are informative."
        ),
    )
    halt_max_steps: int = Field(
        default=10,
        ge=1,
        description="Hard cap on per-slot deliberation steps.",
    )

    # ~~ RL loss coefficients ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    c_actor: float = Field(default=1.0, ge=0.0, description="Actor loss coefficient.")
    c_critic: float = Field(default=0.5, ge=0.0, description="Critic loss coefficient.")
    c_entropy: float = Field(default=0.01, ge=0.0, description="Entropy regularisation coefficient.")
    c_vmPFC: float = Field(default=0.5, ge=0.0, description="vmPFC auxiliary Q-predictor loss coefficient.")

    # ~~ Supervised loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_function: LossType = Field(
        default="stablemax_cross_entropy",
        description="Token-level supervised loss function name.",
    )

    # ~~ Optimizers & scheduling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    optimizer_supervised: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="Optimiser for supervised parameters (PFC + embeddings + LM head).",
    )
    optimizer_rl: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="Optimiser for RL parameters (STR actor-critic only).",
    )
    optimizer_vmPFC: AdamATan2Config = Field(
        default_factory=AdamATan2Config,
        description="Optimiser for vmPFC parameters (pfc.estimator only).",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="LR scheduler config applied to both optimisers.",
    )

    global_batch_size: int = Field(
        ...,
        description=(
            "Global batch size across all devices. "
            "The per-device batch size is computed as `global_batch_size // world_size`."
        ),
    )  # TODO: consider moving to BufferSettings or similar


# =================================================================================================
class TraceFields:
    """Accessors for v2 trace values from :class:`RLStepOutput` / :class:`RLState`."""

    @staticmethod
    def get_model_loss(ctx: StepContext) -> TraceValue:
        return ctx.outputs.loss.detach()

    @staticmethod
    def get_steps(ctx: StepContext) -> TraceValue:
        steps: Tensor = ctx.carry.steps
        return steps.detach()

    @staticmethod
    def get_halted(ctx: StepContext) -> TraceValue:
        halted: Tensor = ctx.carry.halted
        return halted.detach()

    @staticmethod
    def get_action(ctx: StepContext) -> TraceValue:
        return ctx.outputs.action.detach()

    @staticmethod
    def get_value(ctx: StepContext) -> TraceValue:
        return ctx.outputs.value.detach()

    @staticmethod
    def get_policy_logits(ctx: StepContext) -> TraceValue:
        return ctx.outputs.policy_logits.detach()

    @staticmethod
    def get_pred_is_o(ctx: StepContext) -> TraceValue:
        logits: Tensor = ctx.outputs.logits
        pred = torch.argmax(logits.detach(), dim=-1)
        return (pred == O_ID).to(torch.uint8)

    @staticmethod
    def get_outcome(ctx: StepContext) -> TraceValue:
        return ctx.outputs.outcome_mean.detach()


# =================================================================================================
def trace_fields() -> List[TraceField[StepContext]]:
    """ """
    return [
        TraceField(name="loss", get=TraceFields.get_model_loss),
        TraceField(name="steps", get=TraceFields.get_steps),
        TraceField(name="act/halted", get=TraceFields.get_halted),  # compat alias
        TraceField(name="str/action", get=TraceFields.get_action),
        TraceField(name="str/value", get=TraceFields.get_value),
        TraceField(name="str/policy_logits", get=TraceFields.get_policy_logits),
        TraceField(name="pred/is_o", get=TraceFields.get_pred_is_o),
        TraceField(name="outcome", get=TraceFields.get_outcome),
    ]


# =================================================================================================
@dataclass
class HRMState:
    """ """

    pfc: PFCState
    str: STRState

    def detach(self) -> "HRMState":
        """Return a copy with PFC state detached from the computation graph."""
        return HRMState(pfc=self.pfc.detach(), str=self.str.detach())


# =================================================================================================
class HRModelV2(nn.Module):
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelSettings_V2, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.embed_pos = nn.Embedding(config.seq_length, config.hidden_size, device=device, dtype=dtype)
        self.pfc = PFCModel(config.pfc, device=device, dtype=dtype)
        self.str = STRModel(config.str, device=device, dtype=dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)  # fmt: skip
        self.reset_parameters()

    @property
    def config(self) -> ModelSettings_V2:
        """ """
        return self._config

    def reset_parameters(self) -> None:  # -------------------------------------------------------
        """ """
        init_std = self.config.init_std
        trunc_normal_init_(self.embed_tokens.weight, std=init_std)
        trunc_normal_init_(self.embed_pos.weight, std=init_std)
        trunc_normal_init_(self.lm_head.weight, std=init_std)

    def init_state(  # ---------------------------------------------------------------------------
        self, batch_size: int,
    ) -> HRMState:  # fmt: skip
        """ """
        return HRMState(pfc=self.pfc.init_state(batch_size))

    def reset_state(  # --------------------------------------------------------------------------
        self, reset_flag: Tensor, state: HRMState,
    ) -> HRMState:  # fmt: skip
        """ """
        return HRMState(pfc=self.pfc.reset_state(state.pfc, reset_flag))

    def forward(  # -------------------------------------------------------------------------------
        self, inputs: Tensor, state: Optional[HRMState] = None,
    ) -> Tuple[HRMState, Tensor, Tensor, Tensor]:  # fmt: skip
        """ """
        state = state or self.init_state(batch_size=inputs.shape[0])
        x = self.embed_inputs(inputs)  # (B, S, D)
        state_pfc, z_H, q_values = self.pfc(x, state=state.pfc)  # z_H: (B, S+1, D)
        logits = self.lm_head(z_H[:, 1:])  # strip CLS → (B, S, vocab)
        theta_cls = z_H[:, 0]  # (B, D) — theta / CLS summary
        return HRMState(pfc=state_pfc), logits, theta_cls, q_values

    def embed_inputs(  # --------------------------------------------------------------------------
        self, input: Tensor,
    ) -> Tensor:  # fmt: skip
        """ """
        token_embeddings = self.embed_tokens(input.to(torch.int32))
        positions = torch.arange(self.config.seq_length, device=input.device)
        pos_embeddings = self.embed_pos(positions).unsqueeze(0)

        # Scale embeddings to keep activations in a reasonable range.
        return self.config.embedding_scale * (token_embeddings + pos_embeddings)


# =================================================================================================
@dataclass
class RLState:
    """Per-slot state for the RL training loop.

    ``halted``: per-slot "episode done / needs refresh" flag.  When ``True``, that
    slot is replaced with a fresh sample on the next ``RLController.step()`` call.
    """

    model_state: HRMState  # Recurrent backbone state
    steps: Tensor  # (B,) int32 — per-slot deliberation step counter
    halted: Tensor  # (B,) bool  — per-slot done / reset flag
    prev_outcome: Tensor  # (B,) float32 — outcome from previous step (improvement baseline)
    data: Dict[str, Tensor]  # Per-slot buffered "inputs" and "labels"

    def detach(self) -> "RLState":
        return RLState(
            model_state=self.model_state.detach(),
            steps=self.steps,
            halted=self.halted,
            prev_outcome=self.prev_outcome,
            data=self.data,
        )


# =================================================================================================
@dataclass
class RLOutput:
    """Outputs from a single :class:`RLController` step.

    All fields are kept *live* (not detached) inside the controller; the loss head
    detaches selectively.  ``next_value`` is always produced under ``torch.no_grad()``.
    """

    logits: Tensor  # LM logits (B, S, vocab) — for supervised loss
    policy_logits: Tensor  # STR policy logits (B, 2) — for actor and entropy losses
    value: Tensor  # V(s_t) (B,) — for critic loss and TD error
    next_value: Tensor  # V(s_{t+1}) (B,), produced under no_grad — TD bootstrap
    action: Tensor  # Sampled action (B,): 0=halt, 1=continue
    theta_cls: Tensor  # Raw (non-detached) z_H[:,0] — tracing only, not used in loss
    q_values: Tensor  # vmPFC Q-estimates (B, n_actions) — for vmPFC TD-Q loss
    next_q_max: Tensor  # max Q for next state (B,), under no_grad — vmPFC TD target

    def detach(self) -> "RLOutput":
        return RLOutput(
            logits=self.logits.detach(),
            policy_logits=self.policy_logits.detach(),
            value=self.value.detach(),
            next_value=self.next_value.detach(),
            action=self.action.detach(),
            theta_cls=self.theta_cls.detach(),
            q_values=self.q_values.detach(),
            next_q_max=self.next_q_max.detach(),
        )


# =================================================================================================
class RLController:
    """Manages per-slot resets, backbone + STR forward passes, and action selection.

    Responsibilities:
        - Refresh per-slot data for finished episodes (same as ACTController slot refresh).
        - Reset backbone recurrent state for finished slots.
        - Run backbone forward to get LM logits + theta-level CLS features.
        - Feed *detached* CLS features to STR → policy logits + value.
        - Sample (train) or argmax (eval) the action from the STR policy.
        - Compute the done mask and bootstrap ``V_{t+1}`` under ``no_grad``.

    Does NOT compute any loss.  Does NOT use correctness as a gate supervision signal.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, backbone: HRModelV2, str_module: STRModel, halt_max_steps: int,
    ) -> None:  # fmt: skip
        self._backbone = backbone
        self._str = str_module
        self._halt_max_steps = halt_max_steps

    @property
    def backbone(self) -> HRModelV2:
        return self._backbone

    @property
    def str_module(self) -> STRModel:
        return self._str

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

    def refresh_slot_data(  # --------------------------------------------------------------------
        self, batch: Batch, state: RLState,
    ) -> Dict[str, Tensor]:  # fmt: skip
        """Replace halted slots with fresh incoming batch data."""
        halted = state.halted
        return {
            k: torch.where(
                halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                state.data[k],
            )
            for k in batch
        }

    def step(  # ----------------------------------------------------------------------------------
        self, state: RLState, batch: Batch,
        *, explore: bool, allow_halt: bool, gamma: float,
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
        """
        # 1. Refresh slot data; reset prev_outcome to 0.0 for newly refreshed slots
        data = self.refresh_slot_data(batch, state)
        prev_outcome = torch.where(state.halted, torch.zeros_like(state.prev_outcome), state.prev_outcome)

        # 2. Reset backbone recurrent state for halted slots
        model_state = self._backbone.reset_state(state.halted, state.model_state)

        # 3. Backbone forward (gradients flow for supervised loss via logits; q_values → vmPFC)
        new_model_state, logits, theta_cls, q_values = self._backbone(data["inputs"], model_state)

        # 4. STR forward — features MUST be detached (REQ-006: no RL grads into PFC)
        features = theta_cls.detach()  # (B, D)
        policy_logits, value = self._str(features)  # (B, 2), (B,)

        # 5. Action selection
        if explore:
            action = Categorical(logits=policy_logits).sample()  # stochastic (B,)
        else:
            action = policy_logits.argmax(dim=-1)  # deterministic (B,)

        # 6. Step counter + done mask
        steps = torch.where(state.halted, torch.zeros_like(state.steps), state.steps) + 1
        done = steps >= self._halt_max_steps
        if allow_halt:
            done = done | (action == HALT_ACTION)

        # 7. Bootstrap V(s_{t+1}) and max Q(s_{t+1}) — no_grad; zero for done slots
        with torch.no_grad():
            _, _, next_theta_cls, next_q_values = self._backbone(data["inputs"], new_model_state)
            _, next_value = self._str(next_theta_cls.detach())
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


# =================================================================================================
@dataclass(frozen=True)
class RLStepOutput:
    """Aggregated outputs of one :class:`RLLossHead` forward step.

    ``loss`` is the *live* (non-detached) combined loss for ``manual_backward``.
    All other tensor fields are detached for logging/tracing.
    """

    loss: Tensor  # combined total loss — kept live for backward()
    loss_supervised: Tensor  # detached — for logging
    loss_actor: Tensor  # detached — for logging
    loss_critic: Tensor  # detached — for logging
    loss_entropy: Tensor  # detached — for logging
    loss_vmPFC: Tensor  # detached — for logging
    reward_mean: Tensor  # detached — for logging
    delta_mean: Tensor  # detached — for logging
    outcome_mean: Tensor  # detached — for logging
    action: Tensor  # detached (B,) — for tracing
    value: Tensor  # detached (B,) — for tracing
    logits: Tensor  # detached (B, S, V) — for tracing + val accuracy
    policy_logits: Tensor  # detached (B, 2) — for tracing
    q_values: Tensor  # detached (B, n_actions) — for tracing


# =================================================================================================
class RLLossHead(nn.Module):
    """Step module for the HRM v2 STR actor-critic training loop.

    Implements the ``StepModule`` protocol for use with ``StepLoop``.

    Loss composition::

        loss_total = loss_supervised
                   + c_actor   * loss_actor
                   + c_critic  * loss_critic
                   + c_entropy * loss_entropy
                   + c_vmPFC   * loss_vmPFC

    Reward is improvement-based: ``r_t = O_t - O_{t-1}`` where ``O_t`` is the
    fraction of correct non-ignored tokens at step ``t`` (REQ-003).  This produces
    a continuous dopamine-RPE signal fed to the STR TD(0) error.

    During warmup (``is_warmup=True``) only ``loss_supervised`` contributes;
    RL and vmPFC losses are zeroed out (REQ-011).
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, controller: RLController, config: ModelConfig_HRM_V2,
    ) -> None:  # fmt: skip
        super().__init__()
        self._controller = controller
        self._config = config

    @property
    def controller(self) -> RLController:
        return self._controller

    @property
    def config(self) -> ModelConfig_HRM_V2:
        return self._config

    @property
    def loss_fn(self) -> Any:
        """Resolved token-level supervised loss function."""
        return getattr(cross_entropy_module, self._config.loss_function)

    def initial_carry(  # -------------------------------------------------------------------------
        self, batch_sample: Batch,
    ) -> RLState:  # fmt: skip
        return self._controller.initial_state(batch_sample)

    def forward(  # -------------------------------------------------------------------------------
        self, batch: Batch, carry: RLState, **options: Any,
    ) -> Tuple[RLStepOutput, RLState, bool]:  # fmt: skip
        """Run one RL step (``StepModule`` protocol).

        Expected options:
            explore (bool): stochastic action sampling during training.
            allow_halt (bool): whether halt actions can terminate a slot.
            is_warmup (bool): when True, zero RL + vmPFC losses (supervised-only phase).

        Returns ``(step_output, new_carry, done)`` where ``done`` signals that all
        slots have halted (used by ``StepLoop`` to stop iterating).
        """
        explore = bool(options.get("explore", True))
        allow_halt = bool(options.get("allow_halt", True))
        is_warmup = bool(options.get("is_warmup", False))
        gamma = self._config.gamma

        carry, outputs = self._controller.step(
            carry,
            batch,
            explore=explore,
            allow_halt=allow_halt,
            gamma=gamma,
        )
        labels = carry.data["labels"]

        # --- Supervised loss (shapes PFC; STR has no gradient path here) ---------
        loss_sup = self._compute_supervised_loss(outputs.logits, labels)

        # --- Improvement-based reward (no_grad) ----------------------------------
        # r_t = O_t - O_{t-1}: continuous RPE signal (dopamine analogue).
        # carry.prev_outcome = O_{t-1} (or 0.0 at episode start).
        with torch.no_grad():
            outcome = self._compute_outcome(outputs.logits, labels)  # (B,) in [0, 1]
        reward = outcome - carry.prev_outcome  # (B,)

        # Update prev_outcome so the next step uses O_t as its baseline.
        carry.prev_outcome = outcome.detach()

        # --- TD(0) error: delta = r + gamma * V(s') - V(s) -----------------------
        delta = reward + gamma * outputs.next_value - outputs.value  # (B,)

        # --- Actor loss: -log pi(a|s) * stop_gradient(delta) ---------------------
        log_probs = F.log_softmax(outputs.policy_logits, dim=-1)  # (B, 2)
        log_prob_a = log_probs.gather(1, outputs.action.unsqueeze(-1)).squeeze(-1)  # (B,)
        loss_actor = -(log_prob_a * delta.detach()).sum()

        # --- Critic loss: 0.5 * delta^2 ------------------------------------------
        loss_critic = 0.5 * (delta**2).sum()

        # --- Entropy bonus -------------------------------------------------------
        loss_entropy = -Categorical(logits=outputs.policy_logits).entropy().sum()

        # --- vmPFC auxiliary Q-loss (TD Q-learning, off-policy) ------------------
        q_a = outputs.q_values.gather(1, outputs.action.unsqueeze(-1)).squeeze(-1)  # (B,)
        q_target = (reward + gamma * outputs.next_q_max).detach()  # (B,)
        loss_vmPFC = 0.5 * ((q_a - q_target) ** 2).sum()

        # --- Warmup gating: suppress RL + vmPFC losses during supervised warmup --
        if is_warmup:
            loss_actor = torch.zeros_like(loss_actor)
            loss_critic = torch.zeros_like(loss_critic)
            loss_entropy = torch.zeros_like(loss_entropy)
            loss_vmPFC = torch.zeros_like(loss_vmPFC)

        # --- Total ---------------------------------------------------------------
        loss_total = (
            loss_sup
            + self._config.c_actor * loss_actor
            + self._config.c_critic * loss_critic
            + self._config.c_entropy * loss_entropy
            + self._config.c_vmPFC * loss_vmPFC
        )

        step_output = RLStepOutput(
            loss=loss_total,
            loss_supervised=loss_sup.detach(),
            loss_actor=loss_actor.detach(),
            loss_critic=loss_critic.detach(),
            loss_entropy=loss_entropy.detach(),
            loss_vmPFC=loss_vmPFC.detach(),
            reward_mean=reward.mean().detach(),
            delta_mean=delta.mean().detach(),
            outcome_mean=outcome.mean().detach(),
            action=outputs.action.detach(),
            value=outputs.value.detach(),
            logits=outputs.logits.detach(),
            policy_logits=outputs.policy_logits.detach(),
            q_values=outputs.q_values.detach(),
        )
        done = bool(carry.halted.all())
        return step_output, carry, done

    # -- Helpers ----------------------------------------------------------------------------------

    @staticmethod
    def _compute_outcome(logits: Tensor, labels: Tensor) -> Tensor:
        """Per-sequence fraction of correct non-ignored tokens (call inside ``torch.no_grad()``).

        Returns float tensor ``(B,)`` in ``[0, 1]``.
        Sequences where all labels are ignored return 0.0.
        """
        mask = labels != IGNORE_LABEL_ID  # (B, S)
        counts = mask.sum(-1).clamp_min(1).float()  # (B,)
        correct = mask & (logits.argmax(-1) == labels)  # (B, S)
        return correct.sum(-1).float() / counts  # (B,)

    # -- Helpers ----------------------------------------------------------------------------------

    @staticmethod
    def _seq_is_correct(logits: Tensor, labels: Tensor) -> Tensor:
        """Per-sequence exact correctness (call inside ``torch.no_grad()``)."""
        mask = labels != IGNORE_LABEL_ID  # (B, S)
        counts = mask.sum(-1)  # (B,)
        correct = mask & (logits.argmax(-1) == labels)
        return (correct.sum(-1) == counts) & (counts > 0)  # (B,) bool

    def _compute_supervised_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Per-seq-normalised token cross-entropy, summed over the batch."""
        loss_per_token = self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID)  # (B, S)
        counts = (labels != IGNORE_LABEL_ID).sum(-1).clamp_min(1).float()  # (B,)
        return (loss_per_token.sum(-1) / counts).sum()


# =================================================================================================
class TrainingModel(L.LightningModule):
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelConfig_HRM_V2,
    ) -> None:  # fmt: skip
        super().__init__()
        self.model = HRModelV2(config.model)
        self.controller = RLController(self.model, self.model.str, config.halt_max_steps)
        self.step_module = RLLossHead(self.controller, config)
        self._config = config

        # Manual optimisation: explicit backward + opt step (legacy parity + dual-opt clarity).
        self.automatic_optimization = False
        self._train_carry = None

        # Metrics are cloned for train/val to allow separate logging and state management.
        base_metrics = build_metrics()
        self.train_metrics = base_metrics.clone(prefix="train/")
        self.val_metrics = base_metrics.clone(prefix="val/")
        self.trace_specs = TraceSpec(fields=trace_fields())

        # Buffer + assembler implement partial-reset batching for ACT runs.
        self._train_buffer = FifoBuffer(
            capacity_rows=4 * config.global_batch_size,  # or local batch size if you prefer
            keys=("inputs", "labels"),
            pin_memory=True,
        )
        self._train_batch_assembler = PartialResetBatchAssembler(
            buffer=self._train_buffer,
            keys=("inputs", "labels"),
        )

    @property
    def config(self) -> ModelConfig_HRM_V2:
        """ """
        return self._config

    def configure_optimizers(  # ------------------------------------------------------------------
        self,
    ) -> Tuple[List[Optimizer], List[SequentialLR]]:  # fmt: skip
        """ """
        total_steps = int(self.trainer.estimated_stepping_batches)
        sch_cfg = self._config.scheduler

        # Optimizer A: supervised — all model params EXCEPT pfc.estimator (vmPFC)
        vmPFC_ids = {id(p) for p in self.model.pfc.estimator.parameters()}
        sup_params = [p for p in self.model.parameters() if id(p) not in vmPFC_ids]
        opt_sup = AdamATan2(sup_params, self._config.optimizer_supervised)

        # Optimizer B: RL — STR actor-critic only (strictly isolated)
        opt_rl = AdamATan2(list(self.model.str.parameters()), self._config.optimizer_rl)

        # Optimizer C: vmPFC — pfc.estimator only (auxiliary Q-predictor)
        opt_vmPFC = AdamATan2(list(self.model.pfc.estimator.parameters()), self._config.optimizer_vmPFC)

        sch_sup = CosineAnnealingLRWithWarmup(opt_sup, total_steps, sch_cfg)
        sch_rl = CosineAnnealingLRWithWarmup(opt_rl, total_steps, sch_cfg)
        sch_vmPFC = CosineAnnealingLRWithWarmup(opt_vmPFC, total_steps, sch_cfg)

        return [opt_sup, opt_rl, opt_vmPFC], [sch_sup, sch_rl, sch_vmPFC]

    # -- Lifecycle --------------------------------------------------------------------------------

    def on_train_epoch_start(  # ------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Reset training buffer, carry, and metrics at the start of each epoch."""
        self._train_carry = None
        self._train_buffer.clear()
        self.train_metrics.reset()

    def on_validation_epoch_start(  # ------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Reset validation metrics at the start of each epoch."""
        self.val_metrics.reset()

    # -- Training ----------------------------------------------------------------------------------

    def training_step(  # -------------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> Dict[str, Any]:  # fmt: skip
        """ """
        batch_dict = batch

        # Initialize carry/state on the first batch
        if self._train_carry is None:
            self._train_carry = self.step_module.initial_carry(batch_dict)

        # Assemble partial-reset step batch
        step_batch = self._train_batch_assembler.make_step_batch(
            incoming=batch_dict,
            reset_mask=self._train_carry.halted,
        )

        # Horizon=1 step loop: run one step of the controller
        step_batches = repeat(step_batch, 1)
        is_warmup = self.global_step < self._config.rl_warmup_steps
        step_opts = {"explore": True, "allow_halt": not is_warmup, "is_warmup": is_warmup}
        carry0 = self._train_carry

        step = None
        for _t, step in StepLoop(self.step_module, step_batches, carry0, options=step_opts):
            pass  # horizon = 1; loop runs exactly once
        if step is None:
            raise ValueError("StepLoop did not yield any steps.")
        self._train_carry = step.carry.detach()

        # Normalize by local batch size; DDP averages gradients across ranks.
        local_bs = int(batch_dict["inputs"].shape[0])
        out = step.outputs
        loss = _normalize_loss_for_backward(out.loss, local_bs)
        self.manual_backward(loss)

        # Optimizer step and reset gradient for all optimizers
        opt_sup, opt_rl, opt_vmPFC = self.optimizers()  # type: ignore[misc]
        sch_sup, sch_rl, sch_vmPFC = self.lr_schedulers()  # type: ignore[misc]

        opt_sup.step(); opt_sup.zero_grad(set_to_none=True); sch_sup.step()  # fmt: skip
        opt_rl.step(); opt_rl.zero_grad(set_to_none=True); sch_rl.step()  # fmt: skip
        opt_vmPFC.step(); opt_vmPFC.zero_grad(set_to_none=True); sch_vmPFC.step()  # fmt: skip

        scheduler = self.lr_schedulers()
        for sch in scheduler if isinstance(scheduler, list) else [scheduler]:
            sch.step()  # type: ignore

        update_metrics_from_step(self.train_metrics, step.outputs.metrics)
        loss_gm = step.outputs.loss / float(local_bs)
        if (self.global_step + 1) % self.trainer.log_every_n_steps == 0:  # type: ignore
            self.log_dict(self.train_metrics.compute(), on_step=True, on_epoch=False, logger=True)
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train/loss_gm", loss_gm.detach(), on_step=True, on_epoch=False, logger=True)
        self.log("train/warmup", torch.tensor(float(is_warmup)), on_step=True, on_epoch=False, logger=True)

        return {"loss": loss.detach()}

    # -- Validation --------------------------------------------------------------------------------

    def validation_step(  # -----------------------------------------------------------------------
        self, batch: Batch, batch_idx: int,
    ) -> Dict[str, Any]:  # fmt: skip
        """ """
        batch_dict = batch

        # Run a full rollout until all slots halt, collecting traces for logging/analysis.
        step_batches = repeat(batch_dict)  # Run until all examples halt
        step_opts = {"explore": False, "allow_halt": True, "is_warmup": False}
        carry0 = self.step_module.initial_carry(batch_dict)
        collector = TraceCollector(TraceTree(), self.trace_specs)

        step = None
        for t, step in StepLoop(self.step_module, step_batches, carry0, options=step_opts):
            collector.append(t, step)
        if step is None:
            raise ValueError("Evaluation loop did not yield any steps.")

        update_metrics_from_step(self.val_metrics, step.outputs.metrics)
        vals = self.val_metrics.compute()  # Compute metrics based on accumulated state
        self.log_dict(vals, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/accuracy", vals["val/all/accuracy"], prog_bar=True, logger=True, sync_dist=True)

        return {"trace": collector.tree}


# =================================================================================================
def _normalize_loss_for_backward(  # --------------------------------------------------------------
    total_loss: Tensor, local_bs: int,
) -> Tensor:  # fmt: skip
    """Normalize the total loss by the local batch size for distributed training.

    In distributed training (e.g. DDP), each rank computes gradients on its local mini-batch.
    To ensure that the overall gradient magnitudes are consistent regardless of the number of
    devices, we normalize the loss by the local batch size (the number of examples processed
    by this rank). DDP will then average the gradients across ranks, effectively normalizing by
    the global batch size.

    Args:
        total_loss: The unnormalized loss computed for the current mini-batch (scalar tensor).
        local_bs: The effective batch size for this mini-batch on the current rank (number of examples).

    Returns:
        The loss normalized by the local batch size, ready for backward().
    """
    if local_bs <= 0:
        raise ValueError(f"local_bs must be positive, got {local_bs}.")
    return total_loss / float(local_bs)


# =================================================================================================
def supervised_maze_tokenize(  # ------------------------------------------------------------------
    channels: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:  # fmt: skip
    """Convert raw maze channels into flattened input/label token sequences.

    Uses :func:`~ehc_sn.data.transforms.channels_to_grid` to merge topology,
    start, and goals into a canonical ``int32`` grid, then flattens to a 1-D
    token sequence.  The label sequence overwrites solution-path cells with
    :data:`O_ID` (HRM-private supervision token).

    Args:
        channels: Raw NPZ channel dict (as returned by ``MazeDataset``).

    Returns:
        ``{"inputs": int32 (H*W,), "labels": int32 (H*W,)}``.
    """
    grid = channels_to_grid(channels)["grid"]  # (H, W) int32
    inputs = grid.ravel()
    labels = inputs.copy()
    if CHANNEL_SOLUTION in channels:
        labels[channels[CHANNEL_SOLUTION].ravel() > 0] = O_ID
    return {"inputs": inputs, "labels": labels}
