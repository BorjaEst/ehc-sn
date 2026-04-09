"""Concrete aggregate latent-consistency loss head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers.var import MAIN_LATENT_RELATION, VAROutput, VARRolloutState
from ehc_sn.heads._variational import VariationalLosses, VariationalLossHeadBase, VariationalLossStep, get_reg_term, require_latent_relation
from ehc_sn.loss.consistency import LatentCode, mean_latent_norm, mse_consistency, sum_latent_terms
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.loss.regularization import RegularizationNorm, sum_regularization_terms
from ehc_sn.metrics import signals as S
from ehc_sn.metrics.keys import VAR_LOSS_LATENT, VAR_LOSS_OBS_NLL, VAR_LOSS_REG
from ehc_sn.training.types import RatioStat, StepMetrics
from ehc_sn.types import Batch


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
class VARLosses(VariationalLosses):
    """Aggregate latent-consistency loss bundle for VAR."""

    loss_latent_core_sum: Tensor  # Sum of latent-consistency losses over the batch for the current step

    @property
    def loss_latent_sum(self) -> Tensor:
        """Return the aggregate latent-consistency loss for the current step."""
        return self.loss_latent_core_sum


# =================================================================================================
@dataclass(frozen=True)
class VARLossStep(VariationalLossStep):
    """A single rollout/loss step produced by :class:`VARLossHead`."""

    losses: VARLosses  # Combined losses for this step, kept live for backward()
    metrics: StepMetrics  # Aggregated metrics for this step, used for logging
    outputs: Optional[VAROutput] = None  # Raw controller outputs
    signals: Dict[str, Any] | None = None  # Diagnostic signals (T2/T3); plain dict


# =================================================================================================
class VARLossHead(VariationalLossHeadBase[VARLossConfig]):
    """Pure VAR objective scored over executed rollout chunks."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: VARLossConfig,
    ) -> None:  # fmt: skip
        """Create a VAR objective from its loss configuration."""
        super().__init__(config=config)

    def compute_losses(  # -----------------------------------------------------------------------
        self, outputs: VAROutput, carry: Any, **_: Any,
    ) -> VARLosses:  # fmt: skip
        """Compute aggregate observation, latent-consistency, and regularization losses."""
        labels = carry.data["labels"]
        main_relation = require_latent_relation(outputs.latent_relations, MAIN_LATENT_RELATION)
        if labels.ndim == outputs.obs_logits.ndim:
            labels = torch.argmax(labels, dim=-1)

        loss_obs_nll_sum = self.config.c_obs * self.loss_fn(outputs.obs_logits, labels).sum()
        loss_latent_sum = self.config.c_latent * sum_latent_terms(mse_consistency, main_relation.lhs, main_relation.rhs).sum()  # fmt: skip

        reg_latent = get_reg_term(outputs.reg_terms, MAIN_LATENT_RELATION)
        if reg_latent is None or self.config.c_reg == 0.0 or self.config.reg_norm == "none":
            loss_reg_sum = loss_obs_nll_sum.new_zeros(())
        else:
            loss_reg_sum = self.config.c_reg * sum_regularization_terms(reg_latent, self.config.reg_norm).sum()  # fmt: skip

        return VARLosses(
            loss_obs_nll_sum=loss_obs_nll_sum,
            loss_latent_core_sum=loss_latent_sum,
            loss_reg_sum=loss_reg_sum,
        )

    def _build_step_output(  # --------------------------------------------------------------------
        self, losses: VARLosses, metrics: StepMetrics, signals: Dict[str, Any], outputs: VAROutput,
    ) -> VARLossStep:  # fmt: skip
        """Wrap losses, metrics, and signals into a :class:`VARLossStep`."""
        return VARLossStep(losses=losses, metrics=metrics, outputs=outputs, signals=signals)

    def _build_metric_ratios(  # -----------------------------------------------------------------
        self, losses: VARLosses, *, carry: Any, outputs: Any, batch_size: int,
    ) -> dict[str, RatioStat]:  # fmt: skip
        """Build detached ratio metrics for logging."""
        batch_count = losses.total.new_tensor(batch_size, dtype=torch.float32)
        return {
            VAR_LOSS_OBS_NLL: RatioStat(losses.loss_obs_nll_sum.detach(), batch_count),
            VAR_LOSS_LATENT: RatioStat(losses.loss_latent_sum.detach(), batch_count),
            VAR_LOSS_REG: RatioStat(losses.loss_reg_sum.detach(), batch_count),
        }

    def compute_signals(  # -----------------------------------------------------------------------
        self, batch: Batch, state: VARRolloutState, outputs: VAROutput, losses: VARLosses,
    ) -> Dict[str, Tensor]:  # fmt: skip
        """Compute lightweight diagnostic signals for logging."""
        main_relation = require_latent_relation(outputs.latent_relations, MAIN_LATENT_RELATION)
        signals = super().compute_signals(batch, state, outputs, losses)
        signals.update(
            {
                S.LATENT_POST_NORM: mean_latent_norm(main_relation.lhs),
                S.LATENT_PRIOR_NORM: mean_latent_norm(main_relation.rhs),
            }
        )
        if outputs.theta_cls is not None:
            signals[S.THETA_CLS_NORM] = outputs.theta_cls.detach().norm(dim=-1).mean()
        return signals


# =================================================================================================
__all__ = ["LatentCode", "VARLossConfig", "VARLossHead", "VARLosses", "VARLossStep"]
