"""Concrete aggregate variational loss head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from pydantic import Field
from torch import Tensor

from ehc_sn.controllers.var import VARController, VAROutput, VARState
from ehc_sn.heads._variational import (
    VariationalLossConfig,
    VariationalLosses,
    VariationalLossHeadBase,
    VariationalLossStep,
    mean_latent_norm,
    sum_latent_terms,
    sum_regularization_terms,
)
from ehc_sn.loss.consistency import mse_consistency
from ehc_sn.metrics import signals as S
from ehc_sn.metrics.keys import VAR_LOSS_LATENT, VAR_LOSS_OBS_NLL, VAR_LOSS_REG
from ehc_sn.training.types import RatioStat, StepMetrics
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class VARLossConfig(VariationalLossConfig):
    """Configuration for :class:`VARLossHead`."""

    c_latent: float = Field(default=1.0, ge=0.0, description="Aggregate latent loss coefficient.")


# =================================================================================================
@dataclass(frozen=True)
class VARLosses(VariationalLosses):
    """Aggregate variational loss bundle."""

    loss_latent_sum_value: Tensor

    @property
    def loss_latent_sum(self) -> Tensor:
        """Aggregate latent loss term for the step."""
        return self.loss_latent_sum_value


# =================================================================================================
@dataclass(frozen=True)
class VARLossStep(VariationalLossStep):
    """A single rollout/loss step produced by :class:`VARLossHead`."""


# =================================================================================================
class VARLossHead(VariationalLossHeadBase[VARController, VARLossConfig]):
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

    def compute_losses(  # -----------------------------------------------------------------------
        self, outputs: VAROutput, batch: Batch, **_: Any,
    ) -> VARLosses:  # fmt: skip
        """Compute aggregate observation, latent, and regularization losses."""
        labels = batch["labels"]
        if labels.ndim == outputs.obs_logits.ndim:
            labels = torch.argmax(labels, dim=-1)

        loss_obs_nll_sum = self.config.c_obs * self.loss_fn(outputs.obs_logits, labels).sum()
        loss_latent_sum = (
            self.config.c_latent
            * sum_latent_terms(
                mse_consistency,
                outputs.latent_post,
                outputs.latent_prior,
            ).sum()
        )

        if outputs.reg_latent is None or self.config.c_reg == 0.0 or self.config.reg_norm == "none":
            loss_reg_sum = loss_obs_nll_sum.new_zeros(())
        else:
            loss_reg_sum = (
                self.config.c_reg
                * sum_regularization_terms(
                    outputs.reg_latent,
                    self.config.reg_norm,
                ).sum()
            )

        return VARLosses(
            loss_obs_nll_sum=loss_obs_nll_sum,
            loss_reg_sum=loss_reg_sum,
            loss_latent_sum_value=loss_latent_sum,
        )

    def _build_step_output(  # -------------------------------------------------------------------
        self, losses: VARLosses, metrics: Any, signals: Dict[str, Any], outputs: VAROutput,
    ) -> VARLossStep:  # fmt: skip
        """Wrap losses, metrics, and signals into an :class:`VARLossStep`."""
        return VARLossStep(losses=losses, metrics=metrics, outputs=outputs, signals=signals)

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
            "loss_total": losses.total.detach(),
            "loss_obs_nll": losses.loss_obs_nll_sum.detach(),
            "loss_latent": losses.loss_latent_sum.detach(),
            "loss_reg": losses.loss_reg_sum.detach(),
            "latent_post_norm": mean_latent_norm(outputs.latent_post),
            "latent_prior_norm": mean_latent_norm(outputs.latent_prior),
        }
        if outputs.theta_cls is not None:
            signals[S.THETA_CLS_NORM] = outputs.theta_cls.detach().norm(dim=-1).mean()
        return signals


# =================================================================================================
__all__ = ["VARLossConfig", "VARLossHead", "VARLosses", "VARLossStep"]
