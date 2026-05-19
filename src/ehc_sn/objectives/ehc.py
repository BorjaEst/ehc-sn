"""EHC variational objective.

This module implements the ELBO-style EHC rollout-scoring objective. The
canonical public surface is :class:`EHCObjectiveBinding` (protocol),
:class:`EHCObjective` (implementation), and :class:`EHCObjectiveConfig`.

Task-specific supervision extraction and correctness evaluation are fully
delegated to the injected :class:`EHCObjectiveBinding`, so this module remains
task-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.loss.consistency import (
    LatentCode,
    LatentRelation,
    mean_latent_norm,
    mse_consistency,
    sum_latent_terms,
)
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.loss.regularization import (
    RegularizationNorm,
    sum_regularization_terms,
)
from ehc_sn.metrics import signals as S
from ehc_sn.metrics.keys import (
    EHC_ACC_OBS_ANCESTRAL_ALL,
    EHC_ACC_OBS_ANCESTRAL_REVISIT,
    EHC_ACC_OBS_INFERENCE_ALL,
    EHC_ACC_OBS_INFERENCE_REVISIT,
    EHC_ACC_OBS_RETRIEVED_ALL,
    EHC_ACC_OBS_RETRIEVED_REVISIT,
    EHC_LOSS_GRID_KL_ALL,
    EHC_LOSS_GRID_KL_REVISIT,
    EHC_LOSS_OBS_NLL_ALL,
    EHC_LOSS_OBS_NLL_REVISIT,
    EHC_LOSS_PLACE_CONSISTENCY_ALL,
    EHC_LOSS_PLACE_CONSISTENCY_REVISIT,
    EHC_LOSS_REG_ALL,
    EHC_LOSS_REG_REVISIT,
)
from ehc_sn.objectives._variational import (
    VariationalLosses,
    VariationalObjectiveBase,
    VariationalObjectiveStep,
    get_reg_term,
    require_latent_relation,
)
from ehc_sn.rollouts import CarrySnapshot, StepRecord
from ehc_sn.training.types import RatioStat, StepMetrics
from ehc_sn.types import Batch

GRID_REG_TERM = "grid_reg_term"
GRID_TRANSITION_RELATION = "grid_transition_relation"
PLACE_REG_TERM = "place_reg_term"
PLACE_SENSORY_RELATION = "place_sensory_relation"
PLACE_TRANSITION_RELATION = "place_transition_relation"


# =============================================================================
class EHCStepOutput(Protocol):
    """Objective-facing output contract for EHC-family rollout steps."""

    @property
    def logits_inference(self) -> Tensor:
        """Return posterior-path observation logits of shape ``(B, V)``."""
        ...

    @property
    def logits_retrieved(self) -> Tensor:
        """Return sensory-recall-path observation logits of shape ``(B, V)``."""
        ...

    @property
    def logits_ancestral(self) -> Tensor:
        """Return structural-prior-path observation logits of shape ``(B, V)``."""
        ...

    @property
    def latent_relations(self) -> dict[str, LatentRelation]:
        """Return named EHC latent-consistency relations."""
        ...

    @property
    def reg_terms(self) -> dict[str, LatentCode] | None:
        """Return optional named regularization-code overrides."""
        ...

    @property
    def theta_cls(self) -> Tensor | None:
        """Return optional theta-classifier state used for diagnostics."""
        ...


# =============================================================================
class EHCObjectiveBinding[TargetsT](Protocol):
    """Canonical task-binding protocol for the EHC objective.

    The ``executed_batch`` input is the executed-step payload (``record.batch``)
    and is authoritative for current-step supervision. The ``snapshot`` input
    is a frozen post-step snapshot that should only provide continuity facts or
    lightweight post-step projections.
    """

    def extract_targets(  # ---------------------------------------------------
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: Any,
    ) -> TargetsT:
        """Return the task-owned supervision targets for the current step."""

    def extract_observation_id(  # --------------------------------------------
        self,
        targets: TargetsT,
    ) -> Tensor:
        """Return the integer observation-id tensor ``(B,)`` from ``targets``."""

    def extract_protocol_mask(  # ---------------------------------------------
        self,
        targets: TargetsT,
    ) -> Tensor:
        """Return the boolean protocol-eligibility mask ``(B,)`` from ``targets``."""

    def evaluate_observation_metrics(  # --------------------------------------
        self,
        step_output: Any,
        targets: TargetsT,
    ) -> dict[str, RatioStat]:
        """Return task-owned count-bearing accuracy metrics for one step."""


# =============================================================================
class EHCObjectiveConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`EHCObjective`."""

    observation_loss: LossType = Field(
        default="softmax_cross_entropy",
        description="Observation negative log-likelihood primitive from "
        "ehc_sn.loss.cross_entropy.",
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
    # Schedule fields (mirroring TEM training dynamics)
    temp_it: int = Field(
        default=2000,
        ge=1,
        description="Steps over which the temperature ramp reaches 1.0.",
    )
    p2g_use_it: int = Field(
        default=0,
        ge=0,
        description="Sigmoid midpoint (steps) for the p2g-use schedule.",
    )
    p2g_scale: float = Field(
        default=200.0,
        gt=0.0,
        description="Sigmoid scale (steps) for the p2g-use schedule.",
    )
    g_reg_it: int = Field(
        default=40000000,
        ge=1,
        description="Steps over which the grid regularizer decays to zero.",
    )
    p_reg_it: int = Field(
        default=4000,
        ge=1,
        description="Steps over which the place regularizer decays to zero.",
    )


# =============================================================================
@dataclass(frozen=True)
class EHCLosses(VariationalLosses):
    """ELBO-style EHC loss bundle with explicit latent groups."""

    loss_place_transition_sum: Tensor
    loss_place_sensory_sum: Tensor
    loss_grid_kl_sum: Tensor
    loss_place_consistency_sum: Tensor
    protocol_count: Tensor

    @property
    def loss_latent_sum(self) -> Tensor:
        """Return the aggregate latent loss for the current EHC step."""
        return self.loss_grid_kl_sum + self.loss_place_consistency_sum

    @property
    def total(self) -> Tensor:
        """Return the total loss normalized by active protocol count."""
        denom = self.protocol_count.clamp_min(1.0)
        return (
            self.loss_obs_nll_sum
            + self.loss_grid_kl_sum
            + self.loss_place_consistency_sum
            + self.loss_reg_sum
        ) / denom


# =============================================================================
@dataclass(frozen=True)
class EHCObjectiveStep(VariationalObjectiveStep):
    """A single rollout/loss step produced by :class:`EHCObjective`."""

    losses: EHCLosses
    metrics: StepMetrics
    outputs: Optional[EHCStepOutput] = None
    signals: dict[str, Any] | None = None


# =============================================================================
class EHCObjective(VariationalObjectiveBase[EHCObjectiveConfig]):
    """EHC objective scored over executed rollout chunks."""

    def __init__(  # ----------------------------------------------------------
        self,
        config: EHCObjectiveConfig,
        *,
        task_binding: EHCObjectiveBinding[Any],
    ) -> None:
        """Create an EHC objective from its loss configuration."""
        super().__init__(config=config)
        self._task_binding = task_binding

    def compute_losses(  # ----------------------------------------------------
        self,
        outputs: EHCStepOutput,
        carry: Any,
        batch: Any = None,
        step_output: Any = None,
        temp: float = 1.0,
        p2g_use: float = 1.0,
        g_cell_reg: float = 1.0,
        p_cell_reg: float = 1.0,
        **_: Any,
    ) -> EHCLosses:
        """Compute ELBO-style EHC losses for a single step."""
        targets = self._task_binding.extract_targets(
            executed_batch=batch, snapshot=carry, step_output=step_output
        )
        labels = self._task_binding.extract_observation_id(targets)
        protocol_mask = self._task_binding.extract_protocol_mask(targets)
        grid_relation = require_latent_relation(
            outputs.latent_relations, GRID_TRANSITION_RELATION
        )
        place_transition_relation = require_latent_relation(
            outputs.latent_relations, PLACE_TRANSITION_RELATION
        )

        loss_obs_inference = self.loss_fn(outputs.logits_inference, labels)
        loss_obs_retrieved = self.loss_fn(outputs.logits_retrieved, labels)
        loss_obs_ancestral = self.loss_fn(outputs.logits_ancestral, labels)
        loss_obs_nll_sum = self.config.c_obs * self._masked_sum(
            loss_obs_inference + loss_obs_retrieved + loss_obs_ancestral,
            protocol_mask,
        )
        grid_mse = sum_latent_terms(
            mse_consistency, grid_relation.lhs, grid_relation.rhs
        )
        loss_grid_kl_sum = (
            temp
            * self.config.c_grid
            * self._masked_sum(grid_mse, protocol_mask)
        )

        place_transition = sum_latent_terms(
            mse_consistency,
            place_transition_relation.lhs,
            place_transition_relation.rhs,
        )
        place_sensory_relation = outputs.latent_relations.get(
            PLACE_SENSORY_RELATION
        )
        if place_sensory_relation is not None:
            place_sensory = sum_latent_terms(
                mse_consistency,
                place_sensory_relation.lhs,
                place_sensory_relation.rhs,
            )
        else:
            place_sensory = place_transition.new_zeros(place_transition.shape)
        loss_place_transition_sum = (
            temp
            * self.config.c_place
            * self._masked_sum(place_transition, protocol_mask)
        )
        loss_place_sensory_sum = (
            temp
            * p2g_use
            * self.config.c_place
            * self._masked_sum(place_sensory, protocol_mask)
        )
        loss_place_consistency_sum = (
            loss_place_transition_sum + loss_place_sensory_sum
        )

        grid_reg_code = get_reg_term(outputs.reg_terms, GRID_REG_TERM)
        if grid_reg_code is None:
            grid_reg_code = grid_relation.lhs

        place_reg_code = get_reg_term(outputs.reg_terms, PLACE_REG_TERM)
        if place_reg_code is None:
            place_reg_code = place_transition_relation.lhs

        grid_reg = self._regularization_terms(
            grid_reg_code,
            self.config.grid_reg_norm,
            self.config.c_grid_reg * g_cell_reg,
        )
        place_reg = self._regularization_terms(
            place_reg_code,
            self.config.place_reg_norm,
            self.config.c_place_reg * p_cell_reg,
        )
        loss_reg_sum = self._masked_sum(grid_reg + place_reg, protocol_mask)

        protocol_count = protocol_mask.to(dtype=loss_obs_nll_sum.dtype).sum()
        return EHCLosses(
            loss_obs_nll_sum=loss_obs_nll_sum,
            loss_reg_sum=loss_reg_sum,
            loss_place_transition_sum=loss_place_transition_sum,
            loss_place_sensory_sum=loss_place_sensory_sum,
            loss_grid_kl_sum=loss_grid_kl_sum,
            loss_place_consistency_sum=loss_place_consistency_sum,
            protocol_count=protocol_count,
        )

    def _build_metric_ratios(  # ----------------------------------------------
        self,
        losses: EHCLosses,
        *,
        carry: Any,
        outputs: EHCStepOutput,
        batch_size: int,
        batch: Any = None,
        step_output: Any = None,
        temp: float = 1.0,
        p2g_use: float = 1.0,
        g_cell_reg: float = 1.0,
        p_cell_reg: float = 1.0,
        **_: Any,
    ) -> dict[str, RatioStat]:
        """Build detached EHC ratio metrics for logging."""
        targets = self._task_binding.extract_targets(
            executed_batch=batch, snapshot=carry, step_output=step_output
        )
        labels = self._task_binding.extract_observation_id(targets)
        protocol_mask = self._task_binding.extract_protocol_mask(targets)
        protocol_count = protocol_mask.to(dtype=losses.total.dtype).sum()
        batch_count = losses.total.new_tensor(
            batch_size, dtype=losses.total.dtype
        )
        all_loss_sums = self._all_step_loss_sums(
            outputs,
            labels,
            temp=temp,
            p2g_use=p2g_use,
            g_cell_reg=g_cell_reg,
            p_cell_reg=p_cell_reg,
        )
        acc_metrics = self._task_binding.evaluate_observation_metrics(
            outputs, targets
        )
        return {
            **acc_metrics,
            EHC_LOSS_OBS_NLL_REVISIT: RatioStat(
                losses.loss_obs_nll_sum.detach(), protocol_count
            ),
            EHC_LOSS_GRID_KL_REVISIT: RatioStat(
                losses.loss_grid_kl_sum.detach(), protocol_count
            ),
            EHC_LOSS_PLACE_CONSISTENCY_REVISIT: RatioStat(
                losses.loss_place_consistency_sum.detach(), protocol_count
            ),
            EHC_LOSS_REG_REVISIT: RatioStat(
                losses.loss_reg_sum.detach(), protocol_count
            ),
            EHC_LOSS_OBS_NLL_ALL: RatioStat(
                all_loss_sums["loss_obs_nll_sum"], batch_count
            ),
            EHC_LOSS_GRID_KL_ALL: RatioStat(
                all_loss_sums["loss_grid_kl_sum"], batch_count
            ),
            EHC_LOSS_PLACE_CONSISTENCY_ALL: RatioStat(
                all_loss_sums["loss_place_consistency_sum"], batch_count
            ),
            EHC_LOSS_REG_ALL: RatioStat(
                all_loss_sums["loss_reg_sum"], batch_count
            ),
        }

    def build_output(  # ------------------------------------------------
        self,
        losses: EHCLosses,
        metrics: StepMetrics,
        signals: dict[str, Any],
        outputs: Any,
    ) -> EHCObjectiveStep:
        """Wrap losses, metrics, and signals into an :class:`EHCObjectiveStep`."""
        return EHCObjectiveStep(
            losses=losses, metrics=metrics, outputs=outputs, signals=signals
        )

    def compute_signals(  # ---------------------------------------------------
        self,
        record: StepRecord,
        outputs: EHCStepOutput,
        context: Any,
        terms: Any,
        losses: EHCLosses,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Compute detached EHC diagnostics and ELBO-style scalar signals."""
        _ = context, terms
        step_output = record.outputs
        targets = self._task_binding.extract_targets(
            executed_batch=record.batch,
            snapshot=record.carry,
            step_output=step_output,
        )
        labels = self._task_binding.extract_observation_id(targets)
        grid_relation = require_latent_relation(
            outputs.latent_relations, GRID_TRANSITION_RELATION
        )
        place_transition_relation = require_latent_relation(
            outputs.latent_relations, PLACE_TRANSITION_RELATION
        )
        place_sensory_relation = outputs.latent_relations.get(
            PLACE_SENSORY_RELATION
        )
        denom = losses.protocol_count.detach().clamp_min(1.0)
        signals = super().compute_signals(
            record, outputs, context, terms, losses
        )

        loss_obs_inference = (
            self.loss_fn(outputs.logits_inference, labels).sum().detach()
        )
        loss_obs_retrieved = (
            self.loss_fn(outputs.logits_retrieved, labels).sum().detach()
        )
        loss_obs_ancestral = (
            self.loss_fn(outputs.logits_ancestral, labels).sum().detach()
        )
        place_transition = (
            sum_latent_terms(
                mse_consistency,
                place_transition_relation.lhs,
                place_transition_relation.rhs,
            )
            .sum()
            .detach()
        )
        if place_sensory_relation is not None:
            place_sensory = (
                sum_latent_terms(
                    mse_consistency,
                    place_sensory_relation.lhs,
                    place_sensory_relation.rhs,
                )
                .sum()
                .detach()
            )
        else:
            place_sensory = losses.loss_place_consistency_sum.new_zeros(())

        signals.update(
            {
                S.LOSS_GRID_KL: losses.loss_grid_kl_sum.detach() / denom,
                S.LOSS_PLACE_CONSISTENCY: losses.loss_place_consistency_sum.detach()
                / denom,
                S.LOSS_OBS_INFER: loss_obs_inference / denom,
                S.LOSS_OBS_RETRIEVED: loss_obs_retrieved / denom,
                S.LOSS_OBS_ANCESTRAL: loss_obs_ancestral / denom,
                S.LOSS_PLACE_TRANSITION: place_transition / denom,
                S.LOSS_PLACE_SENSORY: place_sensory / denom,
                S.GRID_POST_NORM: mean_latent_norm(grid_relation.lhs),
                S.GRID_PRIOR_NORM: mean_latent_norm(grid_relation.rhs),
                S.PLACE_POST_NORM: mean_latent_norm(
                    place_transition_relation.lhs
                ),
                S.PLACE_PRIOR_NORM: mean_latent_norm(
                    place_transition_relation.rhs
                ),
            }
        )
        theta_cls = getattr(outputs, "theta_cls", None)
        if theta_cls is not None:
            signals[S.THETA_CLS_NORM] = theta_cls.detach().norm(dim=-1).mean()
        return signals

    @staticmethod
    def _masked_sum(  # -------------------------------------------------------
        values: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Return the scalar sum over values selected by a boolean batch mask."""
        return (values * mask.to(dtype=values.dtype)).sum()

    def _all_step_loss_sums(  # -----------------------------------------------
        self,
        outputs: EHCStepOutput,
        labels: Tensor,
        temp: float = 1.0,
        p2g_use: float = 1.0,
        g_cell_reg: float = 1.0,
        p_cell_reg: float = 1.0,
    ) -> dict[str, Tensor]:
        """Return detached all-step EHC loss sums for diagnostics and metric logging."""
        grid_relation = require_latent_relation(
            outputs.latent_relations, GRID_TRANSITION_RELATION
        )
        place_transition_relation = require_latent_relation(
            outputs.latent_relations, PLACE_TRANSITION_RELATION
        )
        place_sensory_relation = outputs.latent_relations.get(
            PLACE_SENSORY_RELATION
        )

        loss_obs_inference = self.loss_fn(outputs.logits_inference, labels)
        loss_obs_retrieved = self.loss_fn(outputs.logits_retrieved, labels)
        loss_obs_ancestral = self.loss_fn(outputs.logits_ancestral, labels)
        loss_obs_nll_sum = (
            self.config.c_obs
            * (
                loss_obs_inference + loss_obs_retrieved + loss_obs_ancestral
            ).sum()
        )

        place_transition = sum_latent_terms(
            mse_consistency,
            place_transition_relation.lhs,
            place_transition_relation.rhs,
        )
        if place_sensory_relation is not None:
            place_sensory = sum_latent_terms(
                mse_consistency,
                place_sensory_relation.lhs,
                place_sensory_relation.rhs,
            )
        else:
            place_sensory = place_transition.new_zeros(place_transition.shape)

        grid_reg_code = get_reg_term(outputs.reg_terms, GRID_REG_TERM)
        if grid_reg_code is None:
            grid_reg_code = grid_relation.lhs

        place_reg_code = get_reg_term(outputs.reg_terms, PLACE_REG_TERM)
        if place_reg_code is None:
            place_reg_code = place_transition_relation.lhs

        grid_reg = self._regularization_terms(
            grid_reg_code,
            self.config.grid_reg_norm,
            self.config.c_grid_reg * g_cell_reg,
        )
        place_reg = self._regularization_terms(
            place_reg_code,
            self.config.place_reg_norm,
            self.config.c_place_reg * p_cell_reg,
        )

        return {
            "loss_obs_nll_sum": loss_obs_nll_sum.detach(),
            "loss_grid_kl_sum": (
                temp
                * self.config.c_grid
                * sum_latent_terms(
                    mse_consistency, grid_relation.lhs, grid_relation.rhs
                ).sum()
            ).detach(),
            "loss_place_consistency_sum": (
                self.config.c_place
                * (
                    temp * place_transition + temp * p2g_use * place_sensory
                ).sum()
            ).detach(),
            "loss_reg_sum": (grid_reg.sum() + place_reg.sum()).detach(),
        }

    def _regularization_terms(  # ---------------------------------------------
        self,
        code: LatentCode,
        norm: RegularizationNorm,
        coefficient: float,
    ) -> Tensor:
        """Return weighted per-example regularization for one latent group."""
        if coefficient == 0.0 or norm == "none":
            first_block = code if isinstance(code, Tensor) else next(iter(code))
            return first_block.new_zeros((first_block.shape[0],))
        return coefficient * sum_regularization_terms(code, norm)


# =============================================================================
def resolve_objective_schedule(  # --------------------------------------------
    step: int,
    config: EHCObjectiveConfig,
) -> dict[str, float]:
    """Derive per-step schedule scalars from global_step and objective config."""
    temp = min((step + 1) / config.temp_it, 1.0)
    p2g_use = 1.0 / (
        1.0 + math.exp(-(step - config.p2g_use_it) / config.p2g_scale)
    )
    g_cell_reg = 1.0 - min((step + 1) / config.g_reg_it, 1.0)
    p_cell_reg = 1.0 - min((step + 1) / config.p_reg_it, 1.0)
    return {
        "temp": temp,
        "p2g_use": p2g_use,
        "g_cell_reg": g_cell_reg,
        "p_cell_reg": p_cell_reg,
    }


# =============================================================================
__all__ = [
    "GRID_REG_TERM",
    "GRID_TRANSITION_RELATION",
    "PLACE_REG_TERM",
    "PLACE_SENSORY_RELATION",
    "PLACE_TRANSITION_RELATION",
    "EHCStepOutput",
    "EHCObjectiveBinding",
    "EHCObjectiveConfig",
    "EHCObjective",
    "EHCObjectiveStep",
    "EHCLosses",
    "resolve_objective_schedule",
]
