"""Concrete aggregate latent-consistency loss head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor

import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.controllers.var import VARController, VAROutput, VARState
from ehc_sn.heads._base import BaseLossHead
from ehc_sn.loss.consistency import LatentCode, mean_latent_norm, mse_consistency, sum_latent_terms
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.loss.regularization import RegularizationNorm, sum_regularization_terms
from ehc_sn.metrics import signals as S
from ehc_sn.metrics.keys import VAR_LOSS_LATENT, VAR_LOSS_OBS_NLL, VAR_LOSS_REG
from ehc_sn.training.types import RatioStat, RolloutAgg, StepMetrics, TokenAgg, TransitionAgg
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class VARLossConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`VARLossHead`."""

    observation_loss: LossType = Field(
        default="softmax_cross_entropy",
        description="Observation negative log-likelihood primitive from ehc_sn.loss.cross_entropy.",
    )
    c_obs: float = Field(
        default=1.0,
        ge=0.0,
        description="Observation loss coefficient.",
    )
    c_latent: float = Field(
        default=1.0,
        ge=0.0,
        description="Latent consistency coefficient.",
    )
    c_reg: float = Field(
        default=0.0,
        ge=0.0,
        description="Regularization loss coefficient.",
    )
    reg_norm: RegularizationNorm = Field(
        default="none",
        description="Regularization norm applied to reg_latent when provided.",
    )


# =================================================================================================
@dataclass(frozen=True)
class VARLosses(DetachMixin):
    """Aggregate latent-consistency loss bundle for VAR."""

    loss_obs_nll_sum: Tensor  # Sum of negative log-likelihood losses over the batch for the current step
    loss_latent_sum: Tensor  # Sum of latent-consistency losses over the batch for the current step
    loss_reg_sum: Tensor  # Sum of regularization losses over the batch for the current step

    @property
    def total(self) -> Tensor:
        """Total scalar loss for the step."""
        return self.loss_obs_nll_sum + self.loss_latent_sum + self.loss_reg_sum


# =================================================================================================
@dataclass(frozen=True)
class VARLossStep:
    """A single rollout/loss step produced by :class:`VARLossHead`."""

    losses: VARLosses  # Combined losses for this step, kept live for backward()
    metrics: StepMetrics  # Aggregated metrics for this step, used for logging
    outputs: Optional[VAROutput] = None  # Raw controller outputs
    signals: Dict[str, Any] | None = None  # Diagnostic signals (T2/T3); plain dict

    def __post_init__(self) -> None:
        if self.signals is None:
            object.__setattr__(self, "signals", {})

    @property
    def loss(self) -> Tensor:
        """Scalar loss for this step, used for back-propagation."""
        return self.losses.total


# =================================================================================================
class VARLossHead(BaseLossHead[VARController, VARLossConfig]):
    """Loss head wrapping :class:`~ehc_sn.controllers.var.VARController`."""

    def __init__(  # ------------------------------------------------------------------------------
        self, controller: VARController, config: VARLossConfig,
    ) -> None:  # fmt: skip
        """Create a loss head.

        Args:
            controller: VAR controller managing halting and state.
            config: Loss configuration.
        """
        super().__init__(controller=controller, config=config)

    @property
    def loss_fn(self) -> Any:
        """Return the configured observation loss primitive."""
        return getattr(cross_entropy_module, self.config.observation_loss)

    def forward(  # -------------------------------------------------------------------------------
        self, batch: Batch, carry: Any, **options: Any,
    ) -> tuple[VARLossStep, Any, bool]:  # fmt: skip
        """Run one controller step and compute aggregate latent-consistency losses."""
        carry, outputs = self.controller.step(carry, batch, **options)
        losses = self.compute_losses(outputs, carry.data["labels"], **options)
        metrics = build_var_step_metrics(
            self._build_metric_ratios(losses, batch_size=int(carry.halted.shape[0])),
            batch_size=int(carry.halted.shape[0]),
            like=losses.total.detach(),
        )
        signals = self.compute_signals(carry, outputs, losses)
        return (
            VARLossStep(losses=losses, metrics=metrics, outputs=outputs, signals=signals),
            carry,
            bool(carry.halted.all()),
        )

    def compute_losses(  # -----------------------------------------------------------------------
        self, outputs: VAROutput, labels: Tensor, **_: Any,
    ) -> VARLosses:  # fmt: skip
        """Compute aggregate observation, latent-consistency, and regularization losses."""
        if labels.ndim == outputs.obs_logits.ndim:
            labels = torch.argmax(labels, dim=-1)

        loss_obs_nll_sum = self.config.c_obs * self.loss_fn(outputs.obs_logits, labels).sum()
        loss_latent_sum = self.config.c_latent * sum_latent_terms( mse_consistency, outputs.latent_post, outputs.latent_prior).sum()  # fmt: skip

        if outputs.reg_latent is None or self.config.c_reg == 0.0 or self.config.reg_norm == "none":
            loss_reg_sum = loss_obs_nll_sum.new_zeros(())
        else:
            loss_reg_sum = self.config.c_reg * sum_regularization_terms( outputs.reg_latent, self.config.reg_norm).sum()  # fmt: skip

        return VARLosses(loss_obs_nll_sum, loss_latent_sum, loss_reg_sum)

    def _build_metric_ratios(  # -----------------------------------------------------------------
        self, losses: VARLosses, *, batch_size: int,
    ) -> dict[str, RatioStat]:  # fmt: skip
        """Build detached ratio metrics for logging."""
        batch_count = losses.total.new_tensor(batch_size, dtype=torch.float32)
        return {
            VAR_LOSS_OBS_NLL: RatioStat(losses.loss_obs_nll_sum.detach(), batch_count),
            VAR_LOSS_LATENT: RatioStat(losses.loss_latent_sum.detach(), batch_count),
            VAR_LOSS_REG: RatioStat(losses.loss_reg_sum.detach(), batch_count),
        }

    def compute_signals(  # -----------------------------------------------------------------------
        self, state: VARState, outputs: VAROutput, losses: VARLosses,
    ) -> Dict[str, Tensor]:  # fmt: skip
        """Compute lightweight diagnostic signals for logging."""
        signals = {
            S.STEPS_MEAN: state.steps.float().mean().detach(),
            S.LOSS_TOTAL: losses.total.detach(),
            S.LOSS_OBS_NLL: losses.loss_obs_nll_sum.detach(),
            S.LOSS_LATENT: losses.loss_latent_sum.detach(),
            S.LOSS_REG: losses.loss_reg_sum.detach(),
            S.LATENT_POST_NORM: mean_latent_norm(outputs.latent_post),
            S.LATENT_PRIOR_NORM: mean_latent_norm(outputs.latent_prior),
        }
        if outputs.theta_cls is not None:
            signals[S.THETA_CLS_NORM] = outputs.theta_cls.detach().norm(dim=-1).mean()
        return signals


# =================================================================================================
def build_var_step_metrics(  # --------------------------------------------------------------------
    extras: Dict[str, RatioStat], *, batch_size: int, like: Tensor,
) -> StepMetrics:  # fmt: skip
    """Build generic metrics for a VAR step.

    VAR heads do not currently report token or rollout accuracy, so those
    aggregates are zero-filled while loss ratios live in ``extras``.
    """
    zero = like.new_zeros(())
    batch_count = like.new_tensor(batch_size, dtype=torch.float32)
    return StepMetrics(
        episode=RolloutAgg(completed_count=zero, eligible_count=zero, accuracy_sum=zero, exact_sum=zero, steps_sum=zero),
        episode_tokens=TokenAgg(token_correct_sum=zero, token_count_sum=zero),
        step=TransitionAgg(evaluated_count=batch_count, eligible_count=batch_count, accuracy_sum=zero, exact_sum=zero, steps_sum=zero),
        step_tokens=TokenAgg(token_correct_sum=zero, token_count_sum=zero),
        extras=extras,
    )  # fmt: skip


# =================================================================================================
__all__ = ["LatentCode", "VARLossConfig", "VARLossHead", "VARLosses", "VARLossStep"]
