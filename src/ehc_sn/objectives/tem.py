"""TEM objective.

This module implements the original TEM rollout-scoring objective. The
canonical public surface is :class:`TEMObjectiveBinding` (protocol),
:class:`TEMObjective` (implementation, also exported as ``TEMObjective``), and
:class:`TEMObjectiveConfig` (also exported as ``TEMObjectiveConfig``).

Task-specific supervision extraction and correctness evaluation are fully
delegated to the injected :class:`TEMObjectiveBinding`, so this module remains
task-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch
from pydantic import BaseModel, Field
from torch import Tensor

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
from ehc_sn.metrics import signals as S
from ehc_sn.metrics.keys import (
    TEM_LOSS_GRID_KL_ALL,
    TEM_LOSS_GRID_KL_REVISIT,
    TEM_LOSS_OBS_ANCESTRAL_ALL,
    TEM_LOSS_OBS_ANCESTRAL_REVISIT,
    TEM_LOSS_OBS_INFERENCE_ALL,
    TEM_LOSS_OBS_INFERENCE_REVISIT,
    TEM_LOSS_OBS_NLL_ALL,
    TEM_LOSS_OBS_NLL_REVISIT,
    TEM_LOSS_OBS_RETRIEVED_ALL,
    TEM_LOSS_OBS_RETRIEVED_REVISIT,
    TEM_LOSS_PLACE_CONSISTENCY_ALL,
    TEM_LOSS_PLACE_CONSISTENCY_REVISIT,
    TEM_LOSS_REG_ALL,
    TEM_LOSS_REG_REVISIT,
)
from ehc_sn.objectives._variational import (
    VariationalLosses,
    VariationalObjectiveBase,
    VariationalObjectiveStep,
    build_variational_step_metrics,
    get_reg_term,
    require_latent_relation,
)
from ehc_sn.rollouts import CarrySnapshot, StepRecord
from ehc_sn.training.types import RatioStat, StepMetrics
from ehc_sn.types import Batch

# Canonical string keys for the latent-relation dictionaries produced by the
# Arena TEM bridge adapters and consumed by TEMObjective.compute_losses.
GRID_TRANSITION_RELATION: str = "grid_transition"
PLACE_TRANSITION_RELATION: str = "place_transition"
PLACE_SENSORY_RELATION: str = "place_sensory"
# Canonical string keys for optional reg-term overrides from bridge adapters.
GRID_REG_TERM: str = "grid_reg"
PLACE_REG_TERM: str = "place_reg"


# =============================================================================
class TEMObjectiveConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`TEMObjective`."""

    observation_loss: LossType = Field(
        default="softmax_cross_entropy",
        description="Observation negative log-likelihood primitive from ehc_sn.loss.cross_entropy.",
    )
    c_obs: float = Field(
        default=1.0,
        ge=0.0,
        description="Observation loss coefficient.",
    )

    latent_loss: str = Field(
        default="mse_consistency",
        description="Latent consistency primitive from ehc_sn.loss.consistency.",
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

    grid_reg_norm: RegularizationNorm = Field(
        default="l2",
        description="Regularization norm for grid codes.",
    )
    c_grid_reg: float = Field(
        default=0.01,
        ge=0.0,
        description="Grid regularization coefficient.",
    )

    place_reg_norm: RegularizationNorm = Field(
        default="l1",
        description="Regularization norm for place codes.",
    )
    c_place_reg: float = Field(
        default=0.02,
        ge=0.0,
        description="Place regularization coefficient.",
    )
    # Schedule fields (original TEM training dynamics)
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
class TEMStepOutput(Protocol):
    """Objective-facing output contract for TEM-family rollout steps."""

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
        """Named latent consistency relations keyed by the relation constants above."""
        ...

    @property
    def reg_terms(self) -> dict[str, LatentCode] | None:
        """Optional named regularization-code overrides; `None` falls back to relation codes."""
        ...


# =============================================================================
class TEMObjectiveBinding[TargetsT](Protocol):
    """Canonical task-binding protocol for the TEM objective.

    Implemented in the adapter layer so that :class:`TEMObjective` stays
    task-agnostic.  The binding owns all task-specific target extraction and
    observation-correctness evaluation; the objective owns only loss math and
    metric assembly.

    Type parameter ``TargetsT`` is the task-owned supervision-target dataclass
    (e.g. :class:`~ehc_sn.tasks.arena.contracts.ArenaTargets`).

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
        ...

    def extract_observation_id(  # --------------------------------------------
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: Any,
    ) -> Tensor:
        """Return the integer observation-id tensor ``(B,)`` for the current step."""
        ...

    def extract_protocol_mask(  # ---------------------------------------------
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: Any,
    ) -> Tensor:
        """Return the boolean protocol-eligibility mask ``(B,)`` for the current step."""
        ...

    def evaluate_observation_metrics(  # --------------------------------------
        self,
        step_output: Any,
        targets: TargetsT,
    ) -> dict[str, RatioStat]:
        """Return task-owned count-bearing accuracy metrics for one step.

        The values are :class:`~ehc_sn.training.types.RatioStat` numerator/
        denominator pairs (counts, not yet reduced to ratios).  Keys must align
        with the TEM metric-key constants in :mod:`ehc_sn.metrics.keys`.
        """
        ...


# =============================================================================
@dataclass(frozen=True)
class TEMLosses(VariationalLosses):
    """TEM loss bundle with explicit latent groups and per-pathway observation sums.

    Observation sums are masked weighted sums under the protocol mask.
    ``loss_obs_nll_sum`` equals the sum of the three per-pathway sums.
    ``loss_place_consistency_sum`` equals the sum of the two place sub-losses.
    """

    # Per-pathway observation sums (masked, weighted by c_obs)
    loss_obs_inference_sum: Tensor
    loss_obs_retrieved_sum: Tensor
    loss_obs_ancestral_sum: Tensor

    @property
    def loss_obs_nll_sum(self) -> Tensor:
        """Return the aggregate observation negative for the current TEM step."""
        return (
            self.loss_obs_inference_sum
            + self.loss_obs_retrieved_sum
            + self.loss_obs_ancestral_sum
        )

    # Per-component place-consistency sums (masked, weighted by c_place * temp)
    loss_place_transition_sum: Tensor
    loss_place_sensory_sum: Tensor

    @property
    def loss_place_consistency_sum(self) -> Tensor:
        """Return the aggregate place-consistency sum for the current TEM step."""
        return self.loss_place_transition_sum + self.loss_place_sensory_sum

    # Per-component grid-consistency sum (masked, weighted by c_grid * temp)
    loss_grid_kl_sum: Tensor

    @property
    def loss_latent_sum(self) -> Tensor:
        """Return the aggregate latent loss for the current TEM step."""
        return self.loss_place_consistency_sum + self.loss_grid_kl_sum

    # Regularization sums (masked, weighted by c_*_reg); the aggregate regularization
    loss_grid_reg_sum: Tensor
    loss_place_reg_sum: Tensor

    @property
    def loss_reg_sum(self) -> Tensor:
        """Return the aggregate regularization sum for the current TEM step."""
        return self.loss_grid_reg_sum + self.loss_place_reg_sum


# =============================================================================
@dataclass(frozen=True)
class TEMTerms:
    """Scored per-example TEM loss terms shared across projections."""

    # Per-pathway observation terms (unmasked, unaggregated, weighted by c_obs)
    obs_inference: Tensor
    obs_retrieved: Tensor
    obs_ancestral: Tensor

    @property
    def obs_nll(self) -> Tensor:
        """Return the aggregate per-example observation loss across all pathways."""
        return self.obs_inference + self.obs_retrieved + self.obs_ancestral

    # Place-consistency terms (unmasked, unaggregated, weighted by c_place * temp)
    place_transition: Tensor
    place_sensory: Tensor

    @property
    def place_consistency(self) -> Tensor:
        """Return the aggregate per-example place-consistency term."""
        return self.place_transition + self.place_sensory

    # Grid-consistency term (unmasked, unaggregated, weighted by c_grid * temp)
    grid_kl: Tensor

    @property
    def latent(self) -> Tensor:
        """Return the aggregate per-example latent loss across all relations."""
        return self.place_consistency + self.grid_kl

    # Regularization terms (unmasked, unaggregated, weighted by c_*_reg)
    grid_reg: Tensor
    place_reg: Tensor

    @property
    def reg(self) -> Tensor:
        """Return the aggregate per-example regularization term."""
        return self.grid_reg + self.place_reg


# =============================================================================
@dataclass(frozen=True)
class TEMObjectiveStep(VariationalObjectiveStep):
    """A single rollout/loss step produced by :class:`TEMObjective`."""

    losses: TEMLosses
    metrics: StepMetrics
    outputs: Optional[TEMStepOutput] = None
    signals: dict[str, Any] | None = None


# =============================================================================
@dataclass(frozen=True)
class TEMContext:
    """Shared objective-scoring context resolved once per TEM consumer."""

    targets: Any
    labels: Tensor
    protocol_mask: Tensor
    grid_relation: LatentRelation
    place_transition_relation: LatentRelation
    place_sensory_relation: LatentRelation | None
    grid_reg_code: LatentCode
    place_reg_code: LatentCode


# =============================================================================
class TEMObjective(VariationalObjectiveBase[TEMObjectiveConfig]):
    """TEM objective scored over executed rollout chunks.

    Loss math lives here. Task-specific target extraction
    and correctness evaluation are fully delegated to the injected
    :class:`TEMObjectiveBinding`.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: TEMObjectiveConfig,
        *,
        task_binding: TEMObjectiveBinding[Any],
    ) -> None:
        """Create a TEM objective from its loss configuration.

        Args:
            config: TEM objective configuration.
            task_binding: Explicit binding for extracting supervision targets and
                evaluating observation correctness.  Use the task-owned adapter
                (e.g. ``ArenaTEMTaskBinding``); no implicit default exists.
        """
        super().__init__(config=config)
        self._task_binding = task_binding

    def runtime_loss_options(  # ----------------------------------------------
        self,
        step: int,
        *,
        p2g_use: float = 1.0,
    ) -> dict[str, float]:
        """Derive per-step schedule scalars from the current step for use in loss computation."""
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}.")

        return {
            "temp": min((step + 1) / float(self.config.temp_it), 1.0),
            "p2g_use": p2g_use,
            "g_cell_reg": 1.0 - min((step + 1) / float(self.config.g_reg_it), 1.0),  # fmt: skip
            "p_cell_reg": 1.0 - min((step + 1) / float(self.config.p_reg_it), 1.0),  # fmt: skip
        }

    def build_context(  # -----------------------------------------------------
        self,
        record: StepRecord,
        outputs: TEMStepOutput,
        **_: Any,
    ) -> TEMContext:
        """Resolve task targets, protocol masks, relations, and reg fallbacks."""
        return TEMContext(
            # Task-specific target extraction and protocol masking are delegated to the binding
            targets=self._task_binding.extract_targets(
                batch=record.batch,
                carry=record.carry,
                step_output=record.outputs,
            ),
            labels=self._task_binding.extract_observation_id(
                batch=record.batch,
                carry=record.carry,
                step_output=record.outputs,
            ),
            protocol_mask=self._task_binding.extract_protocol_mask(
                batch=record.batch,
                carry=record.carry,
                step_output=record.outputs,
            ),
            # Relation extraction is delegated to the binding
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
            # Regularization-code extraction first looks for explicit reg-term overrides
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

    def compute_terms(  # ----------------------------------------------------
        self,
        outputs: TEMStepOutput,
        context: TEMContext,
        *,
        temp: float = 1.0,
        p2g_use: float = 1.0,
        g_cell_reg: float = 1.0,
        p_cell_reg: float = 1.0,
        **_: Any,
    ) -> TEMTerms:
        """Return scored per-example TEM loss terms for one step."""

        return TEMTerms(
            # Compute the per-example observation loss terms for each pathway.
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
            # Grid and place consistency terms under the current config and schedules.
            grid_kl=self.latent_loss_fn(
                relation=context.grid_relation,
            )
            * self.config.c_grid
            * temp,
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
            # Regularization terms for grid and place codes.
            grid_reg=self.greg_loss_fn(
                code=context.grid_reg_code,
            )
            * self.config.c_grid_reg
            * g_cell_reg,
            place_reg=self.preg_loss_fn(
                code=context.place_reg_code,
            )
            * self.config.c_place_reg
            * p_cell_reg,
        )

    def obs_loss_fn(  # -------------------------------------------------------
        self,
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Compute the per-example observation negative log-likelihood under the current config."""
        return super().obs_loss_fn(logits, labels)

    def latent_loss_fn(  # ----------------------------------------------------
        self,
        relation: Optional[LatentRelation],
        *,
        zeros_like: Tensor | None = None,
    ) -> Tensor:
        """Compute the per-example latent loss for one relation under the current config."""
        if relation is None:
            if zeros_like is None:
                raise ValueError("zeros_like is required when relation is None")
            return zeros_like.new_zeros(zeros_like.shape[0])
        return sum_latent_terms(self.latent_term_fn, relation.lhs, relation.rhs)

    def greg_loss_fn(  # ------------------------------------------------------
        self,
        code: LatentCode,
    ) -> Tensor:
        """Compute the per-example grid-code regularization under the current config."""
        return _regularization_terms(code, self.config.grid_reg_norm)

    def preg_loss_fn(  # ------------------------------------------------------
        self,
        code: LatentCode,
    ) -> Tensor:
        """Compute the per-example place-code regularization under the current config."""
        return _regularization_terms(code, self.config.place_reg_norm)

    def compute_losses(  # ----------------------------------------------------
        self,
        terms: TEMTerms,
        context: TEMContext,
        **_: Any,
    ) -> TEMLosses:
        """Compute TEM losses for a single step under the current config and schedules."""
        return TEMLosses(
            loss_obs_inference_sum=_masked_mean(
                values=terms.obs_inference,
                mask=context.protocol_mask,
            ),
            loss_obs_retrieved_sum=_masked_mean(
                values=terms.obs_retrieved,
                mask=context.protocol_mask,
            ),
            loss_obs_ancestral_sum=_masked_mean(
                values=terms.obs_ancestral,
                mask=context.protocol_mask,
            ),
            loss_place_transition_sum=_masked_mean(
                values=terms.place_transition,
                mask=context.protocol_mask,
            ),
            loss_place_sensory_sum=_masked_mean(
                values=terms.place_sensory,
                mask=context.protocol_mask,
            ),
            loss_grid_kl_sum=_masked_mean(
                values=terms.grid_kl,
                mask=context.protocol_mask,
            ),
            loss_grid_reg_sum=_masked_mean(
                values=terms.grid_reg,
                mask=context.protocol_mask,
            ),
            loss_place_reg_sum=_masked_mean(
                values=terms.place_reg,
                mask=context.protocol_mask,
            ),
        )

    def evaluate_metrics(  # ----------------------------------------------------
        self,
        record: StepRecord,
        outputs: TEMStepOutput,
        context: TEMContext,
        terms: TEMTerms,
        losses: TEMLosses,
        **_: Any,
    ) -> StepMetrics:
        """Evaluate TEM metrics for one step from precomputed losses and terms."""
        _ = record, losses
        acc_extras = self._task_binding.evaluate_observation_metrics(
            outputs, context.targets
        )

        revisit = context.protocol_mask.float()
        revisit_count = revisit.sum().detach()
        batch_count = revisit.new_tensor(float(revisit.shape[0]))

        def _rev_sum(t: Tensor) -> Tensor:
            return (t * revisit).sum().detach()

        def _all_sum(t: Tensor) -> Tensor:
            return t.sum().detach()

        loss_extras = {
            TEM_LOSS_OBS_NLL_REVISIT: RatioStat(
                numerator_sum=_rev_sum(terms.obs_nll),
                denominator_sum=revisit_count,
            ),
            TEM_LOSS_OBS_NLL_ALL: RatioStat(
                numerator_sum=_all_sum(terms.obs_nll),
                denominator_sum=batch_count,
            ),
            TEM_LOSS_OBS_INFERENCE_REVISIT: RatioStat(
                numerator_sum=_rev_sum(terms.obs_inference),
                denominator_sum=revisit_count,
            ),
            TEM_LOSS_OBS_INFERENCE_ALL: RatioStat(
                numerator_sum=_all_sum(terms.obs_inference),
                denominator_sum=batch_count,
            ),
            TEM_LOSS_OBS_RETRIEVED_REVISIT: RatioStat(
                numerator_sum=_rev_sum(terms.obs_retrieved),
                denominator_sum=revisit_count,
            ),
            TEM_LOSS_OBS_RETRIEVED_ALL: RatioStat(
                numerator_sum=_all_sum(terms.obs_retrieved),
                denominator_sum=batch_count,
            ),
            TEM_LOSS_OBS_ANCESTRAL_REVISIT: RatioStat(
                numerator_sum=_rev_sum(terms.obs_ancestral),
                denominator_sum=revisit_count,
            ),
            TEM_LOSS_OBS_ANCESTRAL_ALL: RatioStat(
                numerator_sum=_all_sum(terms.obs_ancestral),
                denominator_sum=batch_count,
            ),
            TEM_LOSS_GRID_KL_REVISIT: RatioStat(
                numerator_sum=_rev_sum(terms.grid_kl),
                denominator_sum=revisit_count,
            ),
            TEM_LOSS_GRID_KL_ALL: RatioStat(
                numerator_sum=_all_sum(terms.grid_kl),
                denominator_sum=batch_count,
            ),
            TEM_LOSS_PLACE_CONSISTENCY_REVISIT: RatioStat(
                numerator_sum=_rev_sum(terms.place_consistency),
                denominator_sum=revisit_count,
            ),
            TEM_LOSS_PLACE_CONSISTENCY_ALL: RatioStat(
                numerator_sum=_all_sum(terms.place_consistency),
                denominator_sum=batch_count,
            ),
            TEM_LOSS_REG_REVISIT: RatioStat(
                numerator_sum=_rev_sum(terms.reg),
                denominator_sum=revisit_count,
            ),
            TEM_LOSS_REG_ALL: RatioStat(
                numerator_sum=_all_sum(terms.reg),
                denominator_sum=batch_count,
            ),
        }
        return build_variational_step_metrics({**acc_extras, **loss_extras})

    def compute_signals(  # ---------------------------------------------------
        self,
        record: StepRecord,
        outputs: TEMStepOutput,
        context: TEMContext,
        terms: TEMTerms,
        losses: TEMLosses,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Compute TEM signals for one step from precomputed losses and context."""
        _ = record, context, terms
        grid_rel = outputs.latent_relations.get(GRID_TRANSITION_RELATION)
        place_rel = outputs.latent_relations.get(PLACE_TRANSITION_RELATION)
        _zero = losses.loss_grid_kl_sum.new_zeros(())

        if grid_rel is not None:
            grid_post_norm = mean_latent_norm(grid_rel.lhs).detach()
        else:
            grid_post_norm = _zero
        if grid_rel is not None:
            grid_prior_norm = mean_latent_norm(grid_rel.rhs).detach()
        else:
            grid_prior_norm = _zero
        if place_rel is not None:
            place_post_norm = mean_latent_norm(place_rel.lhs).detach()
        else:
            place_post_norm = _zero
        if place_rel is not None:
            place_prior_norm = mean_latent_norm(place_rel.rhs).detach()
        else:
            place_prior_norm = _zero

        signals = {
            #  Raw loss sums for logging and potential control flow;
            S.LOSS_TOTAL: losses.total.detach(),
            S.LOSS_OBS_NLL: losses.loss_obs_nll_sum.detach(),
            S.LOSS_LATENT: losses.loss_latent_sum.detach(),
            S.LOSS_REG: losses.loss_reg_sum.detach(),
            # Per-relation latent norms for diagnosing collapse or explosion;
            S.LATENT_POST_NORM: grid_post_norm,
            S.LATENT_PRIOR_NORM: grid_prior_norm,
            # Per-pathway and per-component loss sums for fine-grained diagnostics;
            S.LOSS_GRID_KL: losses.loss_grid_kl_sum.detach(),
            S.LOSS_PLACE_CONSISTENCY: losses.loss_place_consistency_sum.detach(),
            # Observation loss components for diagnosing pathway-specific failures;
            S.LOSS_OBS_INFER: losses.loss_obs_inference_sum.detach(),
            S.LOSS_OBS_RETRIEVED: losses.loss_obs_retrieved_sum.detach(),
            S.LOSS_OBS_ANCESTRAL: losses.loss_obs_ancestral_sum.detach(),
            # Place-consistency components for diagnosing relation-specific failures.
            S.LOSS_PLACE_TRANSITION: losses.loss_place_transition_sum.detach(),
            S.LOSS_PLACE_SENSORY: losses.loss_place_sensory_sum.detach(),
            # Regularization components for diagnosing under- or over-regularization.
            S.GRID_POST_NORM: grid_post_norm,
            S.GRID_PRIOR_NORM: grid_prior_norm,
            # Place-consistency components for diagnosing relation-specific failures.
            S.PLACE_POST_NORM: place_post_norm,
            S.PLACE_PRIOR_NORM: place_prior_norm,
        }

        if any(torch.tensor(list(signals.values())) > 1e6):
            print(
                "Large signal values detected: "
                f"{ {k: v.item() for k, v in signals.items()} }",
            )
        if any(torch.isnan(v) for v in signals.values()):
            print(
                "NaN signal values detected: "
                f"{ {k: v.item() for k, v in signals.items()} }",
            )

        return signals

    def build_output(  # ------------------------------------------------------
        self,
        losses: TEMLosses,
        metrics: StepMetrics,
        signals: dict[str, Tensor],
        outputs: TEMStepOutput,
        **_: Any,
    ) -> TEMObjectiveStep:
        """Assemble the final TEMObjectiveStep output for one step."""
        return TEMObjectiveStep(
            losses=losses,
            metrics=metrics,
            signals=signals,
            outputs=outputs,
        )


# =============================================================================
def _regularization_terms(  # -------------------------------------------------
    code: LatentCode,
    norm: RegularizationNorm,
) -> Tensor:
    """Return weighted per-example regularization for one latent group."""
    if norm == "none":
        first_block = code if isinstance(code, Tensor) else next(iter(code))
        return first_block.new_zeros((first_block.shape[0],))
    return sum_regularization_terms(code, norm)


# =============================================================================
def _masked_mean(  # ----------------------------------------------------------
    values: Tensor,
    mask: Tensor,
) -> Tensor:
    """Return the scalar mean over values selected by a boolean batch mask."""
    weights = mask.to(dtype=values.dtype)
    denom = weights.sum().clamp_min(1)
    return (values * weights).sum() / denom


# =============================================================================
__all__ = [
    # canonical names
    "TEMObjectiveBinding",
    "TEMObjectiveConfig",
    "TEMObjective",
    "TEMObjectiveStep",
    "TEMStepOutput",
    "TEMLosses",
    # relation and reg-term key constants
    "GRID_TRANSITION_RELATION",
    "PLACE_TRANSITION_RELATION",
    "PLACE_SENSORY_RELATION",
    "GRID_REG_TERM",
    "PLACE_REG_TERM",
]
