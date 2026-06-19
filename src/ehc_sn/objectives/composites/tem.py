"""TEM objective composite.

This module defines :class:`TEMObjective`, the TEM regime step scorer.
Supervision is supplied via a typed :class:`TEMScoringInput` dataclass;
dynamic algorithm parameters arrive via :class:`TEMScoringContext`.
Latent-relation and regularization-code fields are read from the bridge
output through the structural :class:`TEMPrediction` protocol.

The module owns no task imports and no visible :func:`getattr` probing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

import ehc_sn.loss.consistency as consistency_module
import ehc_sn.loss.cross_entropy as cross_entropy_module
import ehc_sn.metrics.signals as S
from ehc_sn.loss.consistency import (
    LatentCode,
    LatentRelation,
    mean_latent_norm,
    sum_latent_terms,
)
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.loss.regularization import (
    RegularizationNorm,
    sum_regularization_terms,
)
from ehc_sn.metrics.keys import (
    TEM_LOSS_GRID_KL,
    TEM_LOSS_OBS_NLL,
    TEM_LOSS_OBS_PATH,
    TEM_LOSS_OBS_POST,
    TEM_LOSS_OBS_RECALL,
    TEM_LOSS_PLACE_CONSISTENCY,
    TEM_LOSS_PLACE_SENSORY,
    TEM_LOSS_PLACE_TRANSITION,
    TEM_LOSS_REG,
)
from ehc_sn.metrics.step_metrics import (
    RatioStat,
    RolloutAgg,
    StepMetrics,
    TokenAgg,
    TransitionAgg,
)
from ehc_sn.metrics.tem import compute_tem_observation_accuracy
from ehc_sn.rollouts.runtime import StepRecord
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class TEMObjectiveConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`TEMObjective`."""

    observation_loss: LossType = Field(default="softmax_cross_entropy")
    c_obs: float = Field(default=1.0, ge=0.0)
    latent_loss: str = Field(default="mse_consistency")
    c_grid: float = Field(default=1.0, ge=0.0)
    c_place: float = Field(default=1.0, ge=0.0)
    grid_reg_norm: RegularizationNorm = Field(default="l2")
    c_grid_reg: float = Field(default=0.01, ge=0.0)
    place_reg_norm: RegularizationNorm = Field(default="l1")
    c_place_reg: float = Field(default=0.02, ge=0.0)
    temp_it: int = Field(default=2000, ge=1)
    p2g_use_it: int = Field(default=0, ge=0)
    p2g_scale: float = Field(default=200.0, gt=0.0)
    g_reg_it: int = Field(default=40000000, ge=1)
    p_reg_it: int = Field(default=4000, ge=1)


# =============================================================================
# Per-step typed contracts
# =============================================================================


@dataclass(frozen=True)
class TEMSupervision:
    """Task-owned targets and masks for one executed TEM step.

    Built by the task binding from ``record.executed_frame``, not from
    an outer batch.  Contains only target/mask data — no computed
    evaluation outputs.

    Attributes:
        observation_labels: Observation id labels, shape ``(B,)`` int64.
        protocol_mask: Revisit eligibility mask, shape ``(B,)`` bool.
    """

    observation_labels: Tensor
    protocol_mask: Tensor


@dataclass(frozen=True)
class TEMScoringContext:
    """Per-step algorithm parameters that vary during execution.

    Invariant mathematical coefficients belong in :class:`TEMObjectiveConfig`.
    Values here are scheduled and may change across rollout chunks.

    Attributes:
        temperature: Annealed temperature for consistency losses.
        p2g_use: Place-to-grid gate strength.
        g_cell_reg: Grid-cell regularization weight.
        p_cell_reg: Place-cell regularization weight.
    """

    temperature: float
    p2g_use: float
    g_cell_reg: float
    p_cell_reg: float


@dataclass(frozen=True)
class TEMScoringInput:
    """Per-step typed input consumed by :meth:`TEMObjective.evaluate_step`."""

    supervision: TEMSupervision
    context: TEMScoringContext


class TEMPrediction(Protocol):
    """Objective-facing prediction contract satisfied by bridge outputs.

    Adapter bridge output types satisfy this structurally.  No adapter
    imports are required inside ``objectives/``.
    """

    grid_transition: LatentRelation
    place_transition: LatentRelation
    place_sensory: LatentRelation | None
    grid_reg_code: LatentCode
    place_reg_code: LatentCode

    @property
    def logits_post(self) -> Tensor: ...
    @property
    def logits_recall(self) -> Tensor: ...
    @property
    def logits_path(self) -> Tensor: ...


# =============================================================================
class TEMStepOutput(TEMPrediction, Protocol):
    """Backward-compatible protocol — extends :class:`TEMPrediction`."""

    ...


# =============================================================================
@dataclass(frozen=True)
class VariationalLosses(DetachMixin, ABC):
    """Shared parent loss contract for variational-family objectives."""

    @property
    @abstractmethod
    def loss_obs_nll_sum(self) -> Tensor: ...
    @property
    @abstractmethod
    def loss_latent_sum(self) -> Tensor: ...
    @property
    @abstractmethod
    def loss_reg_sum(self) -> Tensor: ...

    @property
    def total(self) -> Tensor:
        return self.loss_obs_nll_sum + self.loss_latent_sum + self.loss_reg_sum


# =============================================================================
@dataclass(frozen=True)
class TEMLosses(VariationalLosses):
    """TEM loss bundle with explicit latent groups and per-pathway observation sums."""

    loss_obs_post_sum: Tensor
    loss_obs_recall_sum: Tensor
    loss_obs_path_sum: Tensor

    @property
    def loss_obs_nll_sum(self) -> Tensor:
        return (
            self.loss_obs_post_sum
            + self.loss_obs_recall_sum
            + self.loss_obs_path_sum
        )

    loss_place_transition_sum: Tensor
    loss_place_sensory_sum: Tensor

    @property
    def loss_place_consistency_sum(self) -> Tensor:
        return self.loss_place_transition_sum + self.loss_place_sensory_sum

    loss_grid_kl_sum: Tensor

    @property
    def loss_latent_sum(self) -> Tensor:
        return self.loss_place_consistency_sum + self.loss_grid_kl_sum

    loss_grid_reg_sum: Tensor
    loss_place_reg_sum: Tensor

    @property
    def loss_reg_sum(self) -> Tensor:
        return self.loss_grid_reg_sum + self.loss_place_reg_sum


# =============================================================================
@dataclass(frozen=True)
class TEMTerms:
    """Scored per-example TEM loss terms shared across projections."""

    obs_post: Tensor
    obs_recall: Tensor
    obs_path: Tensor
    place_transition: Tensor
    place_sensory: Tensor
    grid_kl: Tensor
    grid_reg: Tensor
    place_reg: Tensor

    @property
    def obs_nll(self) -> Tensor:
        return self.obs_post + self.obs_recall + self.obs_path

    @property
    def place_consistency(self) -> Tensor:
        return self.place_transition + self.place_sensory

    @property
    def latent(self) -> Tensor:
        return self.place_consistency + self.grid_kl

    @property
    def reg(self) -> Tensor:
        return self.grid_reg + self.place_reg


# =============================================================================
@dataclass(frozen=True)
class TEMObjectiveStep:
    """A single rollout/loss step produced by :class:`TEMObjective`."""

    losses: TEMLosses
    metrics: StepMetrics
    outputs: Optional[TEMStepOutput] = None
    signals: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.signals is None:
            object.__setattr__(self, "signals", {})

    @property
    def loss(self) -> Tensor:
        return self.losses.total


# =============================================================================
class TEMObjective(nn.Module):
    """TEM regime step scorer.

    Scores one executed ``StepRecord`` at a time via :meth:`evaluate_step`.
    Supervision and dynamic algorithm parameters are supplied through a
    typed :class:`TEMScoringInput`.  Rollout traversal is external — this
    module does not initiate it.
    """

    def __init__(self, config: TEMObjectiveConfig) -> None:
        super().__init__()
        self._config = config

    @property
    def config(self) -> TEMObjectiveConfig:
        """Return the objective configuration."""
        return self._config

    @property
    def obs_loss_fn(self) -> Any:
        return getattr(cross_entropy_module, self.config.observation_loss)

    @property
    def latent_term_fn(self) -> Any:
        return getattr(consistency_module, self.config.latent_loss)

    def evaluate_step(
        self, record: StepRecord, *, inputs: TEMScoringInput
    ) -> TEMObjectiveStep:
        """Score one executed TEM step.

        Args:
            record: Executed step record.  The bridge output must satisfy
                :class:`TEMPrediction` structurally.
            inputs: Typed per-step scoring input.
        """
        supervision = inputs.supervision
        ctx = inputs.context
        labels = supervision.observation_labels
        protocol_mask = supervision.protocol_mask

        # Typed prediction (structural protocol, no adapter imports).
        step_output = record.outputs
        pred: TEMPrediction = getattr(
            step_output,
            "backbone_output",
            getattr(step_output, "task", step_output),
        )

        grid_rel = pred.grid_transition
        place_trans_rel = pred.place_transition
        place_sens_rel = pred.place_sensory
        grid_reg_code = pred.grid_reg_code
        place_reg_code = pred.place_reg_code

        temp = ctx.temperature
        p2g_use = ctx.p2g_use
        g_cell_reg = ctx.g_cell_reg
        p_cell_reg = ctx.p_cell_reg

        # Observation accuracy (computed internally, no task imports).
        observation_metrics = compute_tem_observation_accuracy(
            logits_post=pred.logits_post,
            logits_recall=pred.logits_recall,
            logits_path=pred.logits_path,
            labels=labels,
            protocol_mask=protocol_mask,
        )

        # --- Terms ---
        def _latent_loss_fn(
            relation: LatentRelation | None, *, zeros_like: Tensor | None = None
        ) -> Tensor:
            if relation is None:
                if zeros_like is None:
                    raise ValueError(
                        "zeros_like is required when relation is None"
                    )
                return zeros_like.new_zeros(zeros_like.shape[0])
            return sum_latent_terms(
                self.latent_term_fn, relation.lhs, relation.rhs
            )

        terms = TEMTerms(
            obs_post=self.obs_loss_fn(logits=pred.logits_post, labels=labels)
            * self.config.c_obs,
            obs_recall=self.obs_loss_fn(
                logits=pred.logits_recall, labels=labels
            )
            * self.config.c_obs,
            obs_path=self.obs_loss_fn(logits=pred.logits_path, labels=labels)
            * self.config.c_obs,
            grid_kl=_latent_loss_fn(relation=grid_rel)
            * self.config.c_grid
            * temp,
            place_transition=_latent_loss_fn(relation=place_trans_rel)
            * self.config.c_place
            * temp,
            place_sensory=_latent_loss_fn(
                relation=place_sens_rel, zeros_like=labels
            )
            * self.config.c_place
            * temp
            * p2g_use,
            grid_reg=_regularization_terms(
                grid_reg_code, self.config.grid_reg_norm
            )
            * self.config.c_grid_reg
            * g_cell_reg,
            place_reg=_regularization_terms(
                place_reg_code, self.config.place_reg_norm
            )
            * self.config.c_place_reg
            * p_cell_reg,
        )

        # --- Losses ---
        losses = TEMLosses(
            loss_obs_post_sum=_masked_mean(
                values=terms.obs_post, mask=protocol_mask
            ),
            loss_obs_recall_sum=_masked_mean(
                values=terms.obs_recall, mask=protocol_mask
            ),
            loss_obs_path_sum=_masked_mean(
                values=terms.obs_path, mask=protocol_mask
            ),
            loss_place_transition_sum=_masked_mean(
                values=terms.place_transition, mask=protocol_mask
            ),
            loss_place_sensory_sum=_masked_mean(
                values=terms.place_sensory, mask=protocol_mask
            ),
            loss_grid_kl_sum=_masked_mean(
                values=terms.grid_kl, mask=protocol_mask
            ),
            loss_grid_reg_sum=_masked_mean(
                values=terms.grid_reg, mask=protocol_mask
            ),
            loss_place_reg_sum=_masked_mean(
                values=terms.place_reg, mask=protocol_mask
            ),
        )

        # --- Metrics ---
        acc_extras = observation_metrics
        revisit = protocol_mask.float()
        revisit_count = revisit.sum().detach()

        def _rev_sum(t: Tensor) -> Tensor:
            return (t * revisit).sum().detach()

        loss_extras = {
            TEM_LOSS_OBS_NLL: RatioStat(
                numerator_sum=_rev_sum(terms.obs_nll),
                denominator_sum=revisit_count,
            ),
            TEM_LOSS_OBS_POST: RatioStat(
                numerator_sum=_rev_sum(terms.obs_post),
                denominator_sum=revisit_count,
            ),
            TEM_LOSS_OBS_RECALL: RatioStat(
                numerator_sum=_rev_sum(terms.obs_recall),
                denominator_sum=revisit_count,
            ),
            TEM_LOSS_OBS_PATH: RatioStat(
                numerator_sum=_rev_sum(terms.obs_path),
                denominator_sum=revisit_count,
            ),
            TEM_LOSS_GRID_KL: RatioStat(
                numerator_sum=_rev_sum(terms.grid_kl),
                denominator_sum=revisit_count,
            ),
            TEM_LOSS_PLACE_SENSORY: RatioStat(
                numerator_sum=_rev_sum(terms.place_sensory),
                denominator_sum=revisit_count,
            ),
            TEM_LOSS_PLACE_TRANSITION: RatioStat(
                numerator_sum=_rev_sum(terms.place_transition),
                denominator_sum=revisit_count,
            ),
            TEM_LOSS_PLACE_CONSISTENCY: RatioStat(
                numerator_sum=_rev_sum(terms.place_consistency),
                denominator_sum=revisit_count,
            ),
            TEM_LOSS_REG: RatioStat(
                numerator_sum=_rev_sum(terms.reg), denominator_sum=revisit_count
            ),
        }
        metrics = build_variational_step_metrics({**acc_extras, **loss_extras})

        # --- Signals ---
        _zero = losses.loss_grid_kl_sum.new_zeros(())
        signals = {
            S.LOSS_TOTAL: losses.total.detach(),
            S.LOSS_OBS_NLL: losses.loss_obs_nll_sum.detach(),
            S.LOSS_LATENT: losses.loss_latent_sum.detach(),
            S.LOSS_REG: losses.loss_reg_sum.detach(),
            S.LATENT_POST_NORM: mean_latent_norm(grid_rel.lhs).detach(),
            S.LATENT_PRIOR_NORM: mean_latent_norm(grid_rel.rhs).detach(),
            S.LOSS_GRID_KL: losses.loss_grid_kl_sum.detach(),
            S.LOSS_PLACE_CONSISTENCY: losses.loss_place_consistency_sum.detach(),
            S.LOSS_OBS_INFER: losses.loss_obs_post_sum.detach(),
            S.LOSS_OBS_RECALL: losses.loss_obs_recall_sum.detach(),
            S.LOSS_OBS_PATH: losses.loss_obs_path_sum.detach(),
            S.LOSS_PLACE_TRANSITION: losses.loss_place_transition_sum.detach(),
            S.LOSS_PLACE_SENSORY: losses.loss_place_sensory_sum.detach(),
            S.GRID_POST_NORM: mean_latent_norm(grid_rel.lhs).detach(),
            S.GRID_PRIOR_NORM: mean_latent_norm(grid_rel.rhs).detach(),
            S.PLACE_POST_NORM: mean_latent_norm(place_trans_rel.lhs).detach(),
            S.PLACE_PRIOR_NORM: mean_latent_norm(place_trans_rel.rhs).detach(),
        }

        return TEMObjectiveStep(
            losses=losses, metrics=metrics, outputs=pred, signals=signals
        )

    def obs_loss_fn(self, logits: Tensor, labels: Tensor) -> Tensor:
        return getattr(cross_entropy_module, self.config.observation_loss)(
            logits, labels
        )


# =============================================================================
def build_variational_step_metrics(
    extras: Mapping[str, RatioStat],
) -> StepMetrics:
    """Build a :class:`StepMetrics` from a variational-family extras mapping."""
    first = next(iter(extras.values()), None)
    zero = (
        first.numerator_sum.new_zeros(())
        if first is not None
        else torch.zeros(())
    )
    return StepMetrics(
        episode=RolloutAgg(zero, zero, zero, zero, zero),
        episode_tokens=TokenAgg(zero, zero),
        step=TransitionAgg(zero, zero, zero, zero, zero),
        step_tokens=TokenAgg(zero, zero),
        extras=extras,
    )


# =============================================================================
def _regularization_terms(code: LatentCode, norm: RegularizationNorm) -> Tensor:
    """Return weighted per-example regularization for one latent group."""
    if norm == "none":
        first_block = code if isinstance(code, Tensor) else next(iter(code))
        return first_block.new_zeros((first_block.shape[0],))
    return sum_regularization_terms(code, norm)


# =============================================================================
def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Return the scalar mean over values selected by a boolean batch mask."""
    weights = mask.to(dtype=values.dtype)
    denom = weights.sum().clamp_min(1)
    return (values * weights).sum() / denom


# =============================================================================
__all__ = [
    "TEMObjectiveConfig",
    "TEMObjective",
    "TEMObjectiveStep",
    "TEMStepOutput",
    "TEMLosses",
    "VariationalLosses",
    "TEMSupervision",
    "TEMScoringContext",
    "TEMScoringInput",
    "TEMPrediction",
    "build_variational_step_metrics",
]
