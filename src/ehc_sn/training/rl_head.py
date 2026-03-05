""" """

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeAlias

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.distributions import Categorical

import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.metrics import HaltedAgg, LossAgg, StepMetrics, TokenAgg
from ehc_sn.training.rl_controller import RLController, RLOutput, RLState
from ehc_sn.types import Batch, Device, Dtype
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class RLLossConfig(BaseModel, extra="forbid"):
    """ """

    function: LossType = Field(
        default="stablemax_cross_entropy",
        description="The loss function to use for the modeling loss.",
    )


# =================================================================================================
@dataclass(frozen=True)
class RLLossStep:
    """ """

    loss: Tensor  # combined total loss — kept live for backward()
    metrics: StepMetrics  # Aggregated metrics for this step, used for logging
    outputs: Optional[RLOutput] = None  # Raw controller outputs


# =================================================================================================
class RLLossHead(nn.Module):
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, controller: RLController, config: RLLossConfig,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self._controller = controller
        self._config = config

    @property
    def controller(self) -> RLController:
        """ """
        return self._controller

    @property
    def config(self) -> RLLossConfig:
        """ """
        return self._config

    @property
    def loss_fn(self) -> Any:
        """Resolved token-level supervised loss function."""
        return getattr(cross_entropy_module, self._config.function)

    def initial_carry(  # -------------------------------------------------------------------------
        self, batch_sample: Batch,
    ) -> RLState:  # fmt: skip
        """ """
        return self._controller.initial_state(batch_sample)

    def forward(  # -------------------------------------------------------------------------------
        self, batch: Batch, carry: RLState, *, 
        is_warmup:bool=False, **options: Any,
    ) -> Tuple[RLLossStep, RLState, bool]:  # fmt: skip
        """ """
        # TODO: The RLHeadLoss should receive targets computed using gamma, but it must not store or define gamma itself.
        gamma = self._config.gamma

        carry, outputs = self._controller.step(carry, batch, **options)
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
        loss_total = ...

        step_output = RLLossStep(...)
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
