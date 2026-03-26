"""TEM variational loss head.

This module adapts the legacy TEM loss decomposition to the variational-family
head contract. The public API exposes ELBO-style top-level losses, while
TEM-specific pathway detail remains in detached diagnostics and metric extras.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers.tem import (
    GRID_REG_TERM,
    GRID_TRANSITION_RELATION,
    PLACE_REG_TERM,
    PLACE_SENSORY_RELATION,
    PLACE_TRANSITION_RELATION,
    TEMController,
    TEMOutput,
)
from ehc_sn.heads._variational import VariationalLosses, VariationalLossHeadBase, VariationalLossStep, get_reg_term, require_latent_relation
from ehc_sn.loss.consistency import LatentCode, mean_latent_norm, mse_consistency, sum_latent_terms
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.loss.regularization import RegularizationNorm, sum_regularization_terms
from ehc_sn.metrics import signals as S
from ehc_sn.metrics.keys import (
    TEM_ACC_OBS_ANCESTRAL,
    TEM_ACC_OBS_INFERENCE,
    TEM_ACC_OBS_RETRIEVED,
    TEM_LOSS_GRID_KL,
    TEM_LOSS_OBS_NLL,
    TEM_LOSS_PLACE_CONSISTENCY,
    TEM_LOSS_REG,
)
from ehc_sn.training.types import RatioStat, StepMetrics
from ehc_sn.types import Batch


# =================================================================================================
class TEMLossConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`TEMLossHead`."""

    observation_loss: LossType = Field(
        default="softmax_cross_entropy",
        description="Observation negative log-likelihood primitive from ehc_sn.loss.cross_entropy.",
    )
    c_obs: float = Field(
        default=1.0,
        ge=0.0,
        description="Observation loss coefficient.",
    )
    c_grid: float = Field(
        default=1.0,
        ge=0.0,
        description="Grid consistency coefficient.",
    )
    c_place: float = Field(
        default=1.0,
        ge=0.0,
        description="Place consistency coefficient.",
    )
    c_grid_reg: float = Field(
        default=0.01,
        ge=0.0,
        description="Grid regularization coefficient.",
    )
    c_place_reg: float = Field(
        default=0.02,
        ge=0.0,
        description="Place regularization coefficient.",
    )
    grid_reg_norm: RegularizationNorm = Field(
        default="l2",
        description="Regularization norm for grid codes.",
    )
    place_reg_norm: RegularizationNorm = Field(
        default="l1",
        description="Regularization norm for place codes.",
    )


# =================================================================================================
@dataclass(frozen=True)
class TEMLosses(VariationalLosses):
    """ELBO-style TEM loss bundle with explicit latent groups."""

    loss_grid_kl_sum: Tensor
    loss_place_consistency_sum: Tensor

    @property
    def loss_latent_sum(self) -> Tensor:
        """Return the aggregate latent loss for the current TEM step."""
        return self.loss_grid_kl_sum + self.loss_place_consistency_sum


# =================================================================================================
@dataclass(frozen=True)
class TEMLossStep(VariationalLossStep):
    """A single rollout/loss step produced by :class:`TEMLossHead`."""

    losses: TEMLosses
    metrics: StepMetrics
    outputs: Optional[TEMOutput] = None
    signals: Dict[str, Any] | None = None


# =================================================================================================
class TEMLossHead(VariationalLossHeadBase[TEMController, TEMLossConfig]):
    """Loss head wrapping a TEM-compatible controller or output adapter."""

    def __init__(  # ------------------------------------------------------------------------------
        self, controller: TEMController, config: TEMLossConfig,
    ) -> None:  # fmt: skip
        """Create a TEM loss head.

        Args:
            controller: TEM controller managing rollout state.
            config: Loss configuration.
        """
        super().__init__(controller=controller, config=config)

    def compute_losses(  # -----------------------------------------------------------------------
        self, outputs: TEMOutput, carry: Any, **_: Any,
    ) -> TEMLosses:  # fmt: skip
        """Compute ELBO-style TEM losses for a single step."""
        labels = self._observation_target(carry)
        grid_relation = require_latent_relation(outputs.latent_relations, GRID_TRANSITION_RELATION)
        place_transition_relation = require_latent_relation(outputs.latent_relations, PLACE_TRANSITION_RELATION)  # fmt: skip

        loss_obs_inference = self.loss_fn(outputs.logits_inference, labels).sum()
        loss_obs_retrieved = self.loss_fn(outputs.logits_retrieved, labels).sum()
        loss_obs_ancestral = self.loss_fn(outputs.logits_ancestral, labels).sum()
        loss_obs_nll_sum = self.config.c_obs * (loss_obs_inference + loss_obs_retrieved + loss_obs_ancestral)
        loss_grid_kl_sum = self.config.c_grid * sum_latent_terms(mse_consistency, grid_relation.lhs, grid_relation.rhs).sum()  # fmt: skip

        place_transition = sum_latent_terms(mse_consistency, place_transition_relation.lhs, place_transition_relation.rhs)  # fmt: skip
        place_sensory_relation = outputs.latent_relations.get(PLACE_SENSORY_RELATION)
        if place_sensory_relation is not None:
            place_sensory = sum_latent_terms(mse_consistency, place_sensory_relation.lhs, place_sensory_relation.rhs)  # fmt: skip
        else:
            place_sensory = place_transition.new_zeros(place_transition.shape)
        loss_place_consistency_sum = self.config.c_place * (place_transition + place_sensory).sum()

        grid_reg_code = get_reg_term(outputs.reg_terms, GRID_REG_TERM)
        if grid_reg_code is None:
            grid_reg_code = grid_relation.lhs

        place_reg_code = get_reg_term(outputs.reg_terms, PLACE_REG_TERM)
        if place_reg_code is None:
            place_reg_code = place_transition_relation.lhs
        grid_reg = self._regularization_sum(grid_reg_code, self.config.grid_reg_norm, self.config.c_grid_reg)
        place_reg = self._regularization_sum(place_reg_code, self.config.place_reg_norm, self.config.c_place_reg)  # fmt: skip
        loss_reg_sum = grid_reg + place_reg

        return TEMLosses(
            loss_obs_nll_sum=loss_obs_nll_sum,
            loss_reg_sum=loss_reg_sum,
            loss_grid_kl_sum=loss_grid_kl_sum,
            loss_place_consistency_sum=loss_place_consistency_sum,
        )

    def _build_metric_ratios(  # -----------------------------------------------------------------
        self, losses: TEMLosses, *, carry: Any, outputs: TEMOutput, batch_size: int,
    ) -> dict[str, RatioStat]:  # fmt: skip
        """Build detached TEM ratio metrics for logging."""
        labels = self._observation_target(carry)
        batch_count = losses.total.new_tensor(batch_size, dtype=losses.total.dtype)
        return {
            TEM_ACC_OBS_INFERENCE: RatioStat(_correct_prediction_count(outputs.logits_inference, labels), batch_count),
            TEM_ACC_OBS_RETRIEVED: RatioStat(_correct_prediction_count(outputs.logits_retrieved, labels), batch_count),
            TEM_ACC_OBS_ANCESTRAL: RatioStat(_correct_prediction_count(outputs.logits_ancestral, labels), batch_count),
            TEM_LOSS_OBS_NLL: RatioStat(losses.loss_obs_nll_sum.detach(), batch_count),
            TEM_LOSS_GRID_KL: RatioStat(losses.loss_grid_kl_sum.detach(), batch_count),
            TEM_LOSS_PLACE_CONSISTENCY: RatioStat(losses.loss_place_consistency_sum.detach(), batch_count),
            TEM_LOSS_REG: RatioStat(losses.loss_reg_sum.detach(), batch_count),
        }  # fmt: skip

    def _build_step_output(  # -------------------------------------------------------------------
        self, losses: TEMLosses, metrics: StepMetrics, signals: Dict[str, Any], outputs: Any,
    ) -> TEMLossStep:  # fmt: skip
        """Wrap losses, metrics, and signals into a :class:`TEMLossStep`."""
        return TEMLossStep(losses=losses, metrics=metrics, outputs=outputs, signals=signals)

    def compute_signals(  # -----------------------------------------------------------------------
        self, batch: Batch, carry: Any, outputs: TEMOutput, losses: TEMLosses,
    ) -> Dict[str, Tensor]:  # fmt: skip
        """Compute detached TEM diagnostics and ELBO-style scalar signals."""
        labels = self._observation_target(carry)
        grid_relation = require_latent_relation(outputs.latent_relations, GRID_TRANSITION_RELATION)
        place_transition_relation = require_latent_relation(outputs.latent_relations, PLACE_TRANSITION_RELATION)  # fmt: skip
        place_sensory_relation = outputs.latent_relations.get(PLACE_SENSORY_RELATION)
        signals = super().compute_signals(batch, carry, outputs, losses)

        loss_obs_inference = self.loss_fn(outputs.logits_inference, labels).sum().detach()
        loss_obs_retrieved = self.loss_fn(outputs.logits_retrieved, labels).sum().detach()
        loss_obs_ancestral = self.loss_fn(outputs.logits_ancestral, labels).sum().detach()
        place_transition = sum_latent_terms(mse_consistency, place_transition_relation.lhs, place_transition_relation.rhs).sum().detach()  # fmt: skip
        if place_sensory_relation is not None:
            place_sensory = sum_latent_terms(mse_consistency, place_sensory_relation.lhs, place_sensory_relation.rhs).sum().detach()  # fmt: skip
        else:
            place_sensory = losses.loss_place_consistency_sum.new_zeros(())

        signals.update(
            {
                S.LOSS_GRID_KL: losses.loss_grid_kl_sum.detach(),
                S.LOSS_PLACE_CONSISTENCY: losses.loss_place_consistency_sum.detach(),
                S.LOSS_OBS_INFER: loss_obs_inference,
                S.LOSS_OBS_RETRIEVED: loss_obs_retrieved,
                S.LOSS_OBS_ANCESTRAL: loss_obs_ancestral,
                S.LOSS_PLACE_TRANSITION: place_transition,
                S.LOSS_PLACE_SENSORY: place_sensory,
                S.GRID_POST_NORM: mean_latent_norm(grid_relation.lhs),
                S.GRID_PRIOR_NORM: mean_latent_norm(grid_relation.rhs),
                S.PLACE_POST_NORM: mean_latent_norm(place_transition_relation.lhs),
                S.PLACE_PRIOR_NORM: mean_latent_norm(place_transition_relation.rhs),
            }
        )
        if outputs.theta_cls is not None:
            signals[S.THETA_CLS_NORM] = outputs.theta_cls.detach().norm(dim=-1).mean()
        return signals

    def _observation_target(  # -------------------------------------------------------------------
        self, carry: Any,
    ) -> Tensor:  # fmt: skip
        """Return observation targets from TEM carry data.

        The preferred carry-data key is ``observation_target``. A fallback to
        ``labels`` is retained temporarily for compatibility with the current
        rollout wiring.
        """
        labels = carry.data.get("observation_target", carry.data.get("labels"))
        if labels is None:
            raise KeyError("TEM carry data must provide 'observation_target' or legacy 'labels'.")
        if labels.ndim > 1:
            return labels.argmax(dim=-1)
        return labels

    def _regularization_sum(  # ------------------------------------------------------------------
        self, code: LatentCode, norm: RegularizationNorm, coefficient: float,
    ) -> Tensor:  # fmt: skip
        """Return the weighted regularization sum for one latent group."""
        if coefficient == 0.0 or norm == "none":
            first_block = code if isinstance(code, Tensor) else next(iter(code))
            return first_block.new_zeros(())
        return coefficient * sum_regularization_terms(code, norm).sum()


# =================================================================================================
def _correct_prediction_count(  # -----------------------------------------------------------------
    logits: Tensor, labels: Tensor, 
) -> Tensor:  # fmt: skip
    """Return the detached count of correct observation predictions for one pathway."""
    return logits.argmax(dim=-1).eq(labels).sum().to(dtype=logits.dtype).detach()


# =================================================================================================
__all__ = ["TEMLossConfig", "TEMLossHead", "TEMLosses", "TEMLossStep"]
