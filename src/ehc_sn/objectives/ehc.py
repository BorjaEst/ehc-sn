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

import ehc_sn.metrics.signals as S
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
from ehc_sn.metrics.step_metrics import RatioStat, StepMetrics
from ehc_sn.objectives._variational import (
    VariationalLosses,
    VariationalObjectiveBase,
    VariationalObjectiveStep,
    build_variational_step_metrics,
    get_reg_term,
    require_latent_relation,
)
from ehc_sn.rollouts.runtime import CarrySnapshot, StepRecord
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

    loss_obs_inference_sum: Tensor
    loss_obs_retrieved_sum: Tensor
    loss_obs_ancestral_sum: Tensor
    loss_place_transition_sum: Tensor
    loss_place_sensory_sum: Tensor
    loss_grid_kl_sum: Tensor
    loss_grid_reg_sum: Tensor
    loss_place_reg_sum: Tensor
    # Detached all-batch component sums retained for diagnostics parity.
    loss_obs_inference_all_sum: Tensor
    loss_obs_retrieved_all_sum: Tensor
    loss_obs_ancestral_all_sum: Tensor
    loss_place_transition_all_sum: Tensor
    loss_place_sensory_all_sum: Tensor
    protocol_count: Tensor

    @property
    def loss_obs_nll_sum(self) -> Tensor:
        """Return the aggregate observation negative log-likelihood sum."""
        return (
            self.loss_obs_inference_sum
            + self.loss_obs_retrieved_sum
            + self.loss_obs_ancestral_sum
        )

    @property
    def loss_reg_sum(self) -> Tensor:
        """Return the aggregate regularization sum for the current step."""
        return self.loss_grid_reg_sum + self.loss_place_reg_sum

    @property
    def loss_place_consistency_sum(self) -> Tensor:
        """Return the aggregate place-consistency sum for the current step."""
        return self.loss_place_transition_sum + self.loss_place_sensory_sum

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
@dataclass(frozen=True)
class EHCTerms:
    """Scored per-example EHC loss terms shared across consumers."""

    obs_inference: Tensor
    obs_retrieved: Tensor
    obs_ancestral: Tensor
    place_transition: Tensor
    place_sensory: Tensor
    grid_kl: Tensor
    grid_reg: Tensor
    place_reg: Tensor

    @property
    def obs_nll(self) -> Tensor:
        """Return the aggregate per-example observation loss."""
        return self.obs_inference + self.obs_retrieved + self.obs_ancestral

    @property
    def place_consistency(self) -> Tensor:
        """Return the aggregate per-example place-consistency term."""
        return self.place_transition + self.place_sensory

    @property
    def latent(self) -> Tensor:
        """Return the aggregate per-example latent loss across relations."""
        return self.place_consistency + self.grid_kl

    @property
    def reg(self) -> Tensor:
        """Return the aggregate per-example regularization term."""
        return self.grid_reg + self.place_reg


# =============================================================================
@dataclass(frozen=True)
class EHCContext:
    """Shared objective-scoring context resolved once per EHC consumer."""

    targets: Any
    labels: Tensor
    protocol_mask: Tensor
    grid_relation: LatentRelation
    place_transition_relation: LatentRelation
    place_sensory_relation: LatentRelation | None
    grid_reg_code: LatentCode
    place_reg_code: LatentCode


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

    def runtime_loss_options(  # ----------------------------------------------
        self,
        step: int,
        *,
        p2g_use: float | None = None,
    ) -> dict[str, float]:
        """Derive per-step schedule scalars from the current step."""
        options = resolve_objective_schedule(step=step, config=self.config)
        if p2g_use is not None:
            options["p2g_use"] = p2g_use
        return options

    def build_context(  # -----------------------------------------------------
        self,
        record: StepRecord,
        outputs: EHCStepOutput,
        **_: Any,
    ) -> EHCContext:
        """Resolve task targets, protocol masks, relations, and reg fallbacks."""
        targets = self._task_binding.extract_targets(
            executed_batch=record.batch,
            snapshot=record.carry,
            step_output=record.outputs,
        )
        return EHCContext(
            targets=targets,
            labels=self._task_binding.extract_observation_id(targets),
            protocol_mask=self._task_binding.extract_protocol_mask(targets),
            grid_relation=require_latent_relation(
                outputs.latent_relations,
                GRID_TRANSITION_RELATION,
            ),
            place_transition_relation=require_latent_relation(
                outputs.latent_relations,
                PLACE_TRANSITION_RELATION,
            ),
            place_sensory_relation=outputs.latent_relations.get(
                PLACE_SENSORY_RELATION,
            ),
            grid_reg_code=get_reg_term(
                outputs.reg_terms,
                outputs.latent_relations,
                key=GRID_REG_TERM,
                fallback=GRID_TRANSITION_RELATION,
            ),
            place_reg_code=get_reg_term(
                outputs.reg_terms,
                outputs.latent_relations,
                key=PLACE_REG_TERM,
                fallback=PLACE_TRANSITION_RELATION,
            ),
        )

    def obs_loss_fn(  # -------------------------------------------------------
        self,
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Compute per-example observation NLL under the configured loss."""
        return super().obs_loss_fn(logits, labels)

    def latent_loss_fn(  # ----------------------------------------------------
        self,
        relation: LatentRelation | None,
        *,
        zeros_like: Tensor | None = None,
    ) -> Tensor:
        """Compute per-example latent loss for one relation."""
        if relation is None:
            if zeros_like is None:
                raise ValueError("zeros_like is required when relation is None")
            return zeros_like.new_zeros(zeros_like.shape[0])
        return sum_latent_terms(mse_consistency, relation.lhs, relation.rhs)

    def greg_loss_fn(  # ------------------------------------------------------
        self,
        code: LatentCode,
    ) -> Tensor:
        """Compute per-example grid-code regularization under the config."""
        return _regularization_terms(code, self.config.grid_reg_norm)

    def preg_loss_fn(  # ------------------------------------------------------
        self,
        code: LatentCode,
    ) -> Tensor:
        """Compute per-example place-code regularization under the config."""
        return _regularization_terms(code, self.config.place_reg_norm)

    def compute_terms(  # -----------------------------------------------------
        self,
        outputs: EHCStepOutput,
        context: EHCContext,
        *,
        temp: float = 1.0,
        p2g_use: float = 1.0,
        g_cell_reg: float = 1.0,
        p_cell_reg: float = 1.0,
        **_: Any,
    ) -> EHCTerms:
        """Return scored per-example EHC loss terms for one step."""
        return EHCTerms(
            obs_inference=self.obs_loss_fn(
                logits=outputs.logits_inference,
                labels=context.labels,
            )
            * self.config.c_obs,
            obs_retrieved=self.obs_loss_fn(
                logits=outputs.logits_retrieved,
                labels=context.labels,
            )
            * self.config.c_obs,
            obs_ancestral=self.obs_loss_fn(
                logits=outputs.logits_ancestral,
                labels=context.labels,
            )
            * self.config.c_obs,
            place_transition=self.latent_loss_fn(
                relation=context.place_transition_relation,
            )
            * self.config.c_place
            * temp,
            place_sensory=self.latent_loss_fn(
                relation=context.place_sensory_relation,
                zeros_like=context.labels,
            )
            * self.config.c_place
            * temp
            * p2g_use,
            grid_kl=self.latent_loss_fn(
                relation=context.grid_relation,
            )
            * self.config.c_grid
            * temp,
            grid_reg=self.greg_loss_fn(context.grid_reg_code)
            * self.config.c_grid_reg
            * g_cell_reg,
            place_reg=self.preg_loss_fn(context.place_reg_code)
            * self.config.c_place_reg
            * p_cell_reg,
        )

    def compute_losses(  # ----------------------------------------------------
        self,
        terms: EHCTerms,
        context: EHCContext,
        **_: Any,
    ) -> EHCLosses:
        """Compute ELBO-style EHC losses for a single step."""
        protocol_count = context.protocol_mask.to(
            dtype=terms.obs_nll.dtype
        ).sum()
        return EHCLosses(
            loss_obs_inference_sum=self._masked_sum(
                terms.obs_inference,
                context.protocol_mask,
            ),
            loss_obs_retrieved_sum=self._masked_sum(
                terms.obs_retrieved,
                context.protocol_mask,
            ),
            loss_obs_ancestral_sum=self._masked_sum(
                terms.obs_ancestral,
                context.protocol_mask,
            ),
            loss_place_transition_sum=self._masked_sum(
                terms.place_transition,
                context.protocol_mask,
            ),
            loss_place_sensory_sum=self._masked_sum(
                terms.place_sensory,
                context.protocol_mask,
            ),
            loss_grid_kl_sum=self._masked_sum(
                terms.grid_kl,
                context.protocol_mask,
            ),
            loss_grid_reg_sum=self._masked_sum(
                terms.grid_reg,
                context.protocol_mask,
            ),
            loss_place_reg_sum=self._masked_sum(
                terms.place_reg,
                context.protocol_mask,
            ),
            loss_obs_inference_all_sum=terms.obs_inference.sum().detach(),
            loss_obs_retrieved_all_sum=terms.obs_retrieved.sum().detach(),
            loss_obs_ancestral_all_sum=terms.obs_ancestral.sum().detach(),
            loss_place_transition_all_sum=terms.place_transition.sum().detach(),
            loss_place_sensory_all_sum=terms.place_sensory.sum().detach(),
            protocol_count=protocol_count,
        )

    def evaluate_metrics(  # --------------------------------------------------
        self,
        record: StepRecord,
        outputs: EHCStepOutput,
        context: EHCContext,
        terms: EHCTerms,
        losses: EHCLosses,
        **_: Any,
    ) -> StepMetrics:
        """Evaluate EHC metrics for one step from precomputed losses and terms."""
        _ = record, losses
        acc_extras = self._task_binding.evaluate_observation_metrics(
            outputs,
            context.targets,
        )
        revisit = context.protocol_mask.float()
        revisit_count = revisit.sum().detach()
        batch_count = revisit.new_tensor(float(revisit.shape[0]))

        def _rev_sum(tensor: Tensor) -> Tensor:
            return (tensor * revisit).sum().detach()

        def _all_sum(tensor: Tensor) -> Tensor:
            return tensor.sum().detach()

        loss_extras = {
            EHC_LOSS_OBS_NLL_REVISIT: RatioStat(
                numerator_sum=_rev_sum(terms.obs_nll),
                denominator_sum=revisit_count,
            ),
            EHC_LOSS_OBS_NLL_ALL: RatioStat(
                numerator_sum=_all_sum(terms.obs_nll),
                denominator_sum=batch_count,
            ),
            EHC_LOSS_GRID_KL_REVISIT: RatioStat(
                numerator_sum=_rev_sum(terms.grid_kl),
                denominator_sum=revisit_count,
            ),
            EHC_LOSS_GRID_KL_ALL: RatioStat(
                numerator_sum=_all_sum(terms.grid_kl),
                denominator_sum=batch_count,
            ),
            EHC_LOSS_PLACE_CONSISTENCY_REVISIT: RatioStat(
                numerator_sum=_rev_sum(terms.place_consistency),
                denominator_sum=revisit_count,
            ),
            EHC_LOSS_PLACE_CONSISTENCY_ALL: RatioStat(
                numerator_sum=_all_sum(terms.place_consistency),
                denominator_sum=batch_count,
            ),
            EHC_LOSS_REG_REVISIT: RatioStat(
                numerator_sum=_rev_sum(terms.reg),
                denominator_sum=revisit_count,
            ),
            EHC_LOSS_REG_ALL: RatioStat(
                numerator_sum=_all_sum(terms.reg),
                denominator_sum=batch_count,
            ),
        }
        return build_variational_step_metrics({**acc_extras, **loss_extras})

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
        context: EHCContext,
        terms: EHCTerms,
        losses: EHCLosses,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Compute detached EHC diagnostics and ELBO-style scalar signals."""
        _ = record
        denom = losses.protocol_count.detach().clamp_min(1.0)
        grid_post_norm = mean_latent_norm(context.grid_relation.lhs).detach()
        grid_prior_norm = mean_latent_norm(context.grid_relation.rhs).detach()
        place_post_norm = mean_latent_norm(
            context.place_transition_relation.lhs
        ).detach()
        place_prior_norm = mean_latent_norm(
            context.place_transition_relation.rhs
        ).detach()

        signals = {
            S.LOSS_TOTAL: losses.total.detach(),
            S.LOSS_OBS_NLL: losses.loss_obs_nll_sum.detach(),
            S.LOSS_LATENT: losses.loss_latent_sum.detach(),
            S.LOSS_REG: losses.loss_reg_sum.detach(),
            S.LATENT_POST_NORM: grid_post_norm,
            S.LATENT_PRIOR_NORM: grid_prior_norm,
            S.LOSS_GRID_KL: losses.loss_grid_kl_sum.detach() / denom,
            S.LOSS_PLACE_CONSISTENCY: losses.loss_place_consistency_sum.detach()
            / denom,
            S.LOSS_OBS_INFER: losses.loss_obs_inference_all_sum / denom,
            S.LOSS_OBS_RETRIEVED: losses.loss_obs_retrieved_all_sum / denom,
            S.LOSS_OBS_ANCESTRAL: losses.loss_obs_ancestral_all_sum / denom,
            S.LOSS_PLACE_TRANSITION: losses.loss_place_transition_all_sum
            / denom,
            S.LOSS_PLACE_SENSORY: losses.loss_place_sensory_all_sum / denom,
            S.GRID_POST_NORM: grid_post_norm,
            S.GRID_PRIOR_NORM: grid_prior_norm,
            S.PLACE_POST_NORM: place_post_norm,
            S.PLACE_PRIOR_NORM: place_prior_norm,
        }
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
def _regularization_terms(  # -------------------------------------------------
    code: LatentCode,
    norm: RegularizationNorm,
) -> Tensor:
    """Return per-example regularization for one latent group."""
    if norm == "none":
        first_block = code if isinstance(code, Tensor) else next(iter(code))
        return first_block.new_zeros((first_block.shape[0],))
    return sum_regularization_terms(code, norm)


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
