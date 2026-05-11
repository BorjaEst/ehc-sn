"""TEM variational objective.

This module implements the ELBO-style TEM rollout-scoring objective. The
canonical public surface is :class:`TEMObjectiveBinding` (protocol),
:class:`TEMLossHead` (implementation, also exported as ``TEMObjective``), and
:class:`TEMLossConfig` (also exported as ``TEMObjectiveConfig``).

Task-specific supervision extraction and correctness evaluation are fully
delegated to the injected :class:`TEMObjectiveBinding`, so this module remains
task-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers.tem import (
    GRID_REG_TERM,
    GRID_TRANSITION_RELATION,
    PLACE_REG_TERM,
    PLACE_SENSORY_RELATION,
    PLACE_TRANSITION_RELATION,
    TEMStepOutput,
)
from ehc_sn.loss.consistency import LatentCode, mean_latent_norm, mse_consistency, sum_latent_terms
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.loss.regularization import RegularizationNorm, sum_regularization_terms
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
    VariationalLossHeadBase,
    VariationalLossStep,
    get_reg_term,
    require_latent_relation,
)
from ehc_sn.training.types import RatioStat, StepMetrics
from ehc_sn.types import Batch


# =================================================================================================
class TEMObjectiveBinding[TargetsT](Protocol):
    """Canonical task-binding protocol for the TEM objective.

    Implemented in the adapter layer so that :class:`TEMLossHead` stays
    task-agnostic.  The binding owns all task-specific target extraction and
    observation-correctness evaluation; the objective owns only loss math and
    metric assembly.

    Type parameter ``TargetsT`` is the task-owned supervision-target dataclass
    (e.g. :class:`~ehc_sn.tasks.arena.contracts.ArenaTargets`).
    """

    def extract_targets(self, batch: Batch, carry: Any, step_output: Any) -> TargetsT:
        """Return the task-owned supervision targets for the current step."""
        ...

    def extract_observation_id(self, targets: TargetsT) -> Tensor:
        """Return the integer observation-id tensor ``(B,)`` from ``targets``."""
        ...

    def extract_protocol_mask(self, targets: TargetsT) -> Tensor:
        """Return the boolean protocol-eligibility mask ``(B,)`` from ``targets``."""
        ...

    def evaluate_observation_metrics(self, step_output: Any, targets: TargetsT) -> dict[str, RatioStat]:
        """Return task-owned count-bearing accuracy metrics for one step.

        The values are :class:`~ehc_sn.training.types.RatioStat` numerator/
        denominator pairs (counts, not yet reduced to ratios).  Keys must align
        with the TEM metric-key constants in :mod:`ehc_sn.metrics.keys`.
        """
        ...


# Backward-compatible alias — prefer TEMObjectiveBinding in new code.
TEMSupervisionBinding = TEMObjectiveBinding


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
    outputs: Optional[TEMStepOutput] = None
    signals: dict[str, Any] | None = None


# =================================================================================================
class TEMLossHead(VariationalLossHeadBase[TEMLossConfig]):
    """TEM objective scored over executed rollout chunks.

    Loss math (ELBO decomposition) lives here. Task-specific target extraction
    and correctness evaluation are fully delegated to the injected
    :class:`TEMObjectiveBinding`.
    """

    def __init__(self, config: TEMLossConfig, *, task_binding: TEMObjectiveBinding[Any],) -> None:  # fmt: skip  # ------------------------------------------------------------------------------
        """Create a TEM objective from its loss configuration.

        Args:
            config: TEM objective configuration.
            task_binding: Explicit binding for extracting supervision targets and
                evaluating observation correctness.  Use the task-owned adapter
                (e.g. ``ArenaTEMTaskBinding``); no implicit default exists.
        """
        super().__init__(config=config)
        self._task_binding = task_binding

    def compute_losses(self, outputs: TEMStepOutput, carry: Any, batch: Any = None, step_output: Any = None, **_: Any,) -> TEMLosses:  # fmt: skip  # -----------------------------------------------------------------------
        """Compute ELBO-style TEM losses for a single step."""
        targets = self._task_binding.extract_targets(batch, carry, step_output)
        labels = self._task_binding.extract_observation_id(targets)
        protocol_mask = self._task_binding.extract_protocol_mask(targets)
        grid_relation = require_latent_relation(outputs.latent_relations, GRID_TRANSITION_RELATION)
        place_transition_relation = require_latent_relation(outputs.latent_relations, PLACE_TRANSITION_RELATION)  # fmt: skip

        loss_obs_inference = self.loss_fn(outputs.logits_inference, labels)
        loss_obs_retrieved = self.loss_fn(outputs.logits_retrieved, labels)
        loss_obs_ancestral = self.loss_fn(outputs.logits_ancestral, labels)
        loss_obs_nll_sum = self.config.c_obs * self._masked_sum(
            loss_obs_inference + loss_obs_retrieved + loss_obs_ancestral,
            protocol_mask,
        )
        loss_grid_kl_sum = self.config.c_grid * self._masked_sum(
            sum_latent_terms(mse_consistency, grid_relation.lhs, grid_relation.rhs),
            protocol_mask,
        )  # fmt: skip

        place_transition = sum_latent_terms(mse_consistency, place_transition_relation.lhs, place_transition_relation.rhs)  # fmt: skip
        place_sensory_relation = outputs.latent_relations.get(PLACE_SENSORY_RELATION)
        if place_sensory_relation is not None:
            place_sensory = sum_latent_terms(mse_consistency, place_sensory_relation.lhs, place_sensory_relation.rhs)  # fmt: skip
        else:
            place_sensory = place_transition.new_zeros(place_transition.shape)
        loss_place_consistency_sum = self.config.c_place * self._masked_sum(place_transition + place_sensory, protocol_mask)

        grid_reg_code = get_reg_term(outputs.reg_terms, GRID_REG_TERM)
        if grid_reg_code is None:
            grid_reg_code = grid_relation.lhs

        place_reg_code = get_reg_term(outputs.reg_terms, PLACE_REG_TERM)
        if place_reg_code is None:
            place_reg_code = place_transition_relation.lhs
        grid_reg = self._regularization_terms(grid_reg_code, self.config.grid_reg_norm, self.config.c_grid_reg)
        place_reg = self._regularization_terms(place_reg_code, self.config.place_reg_norm, self.config.c_place_reg)  # fmt: skip
        loss_reg_sum = self._masked_sum(grid_reg + place_reg, protocol_mask)

        return TEMLosses(
            loss_obs_nll_sum=loss_obs_nll_sum,
            loss_reg_sum=loss_reg_sum,
            loss_grid_kl_sum=loss_grid_kl_sum,
            loss_place_consistency_sum=loss_place_consistency_sum,
        )

    def _build_metric_ratios(self, losses: TEMLosses, *, carry: Any, outputs: TEMStepOutput, batch_size: int, batch: Any = None, step_output: Any = None, **_: Any,) -> dict[str, RatioStat]:  # fmt: skip  # -----------------------------------------------------------------
        """Build detached TEM ratio metrics for logging.

        Accuracy metrics come from the task-owned binding so this objective
        does not recompute argmax correctness locally.  Loss ratio metrics are
        assembled here from the loss bundle computed in :meth:`compute_losses`.
        """
        targets = self._task_binding.extract_targets(batch, carry, step_output)
        labels = self._task_binding.extract_observation_id(targets)
        protocol_mask = self._task_binding.extract_protocol_mask(targets)
        protocol_count = protocol_mask.to(dtype=losses.total.dtype).sum()
        batch_count = losses.total.new_tensor(batch_size, dtype=losses.total.dtype)
        all_loss_sums = self._all_step_loss_sums(outputs, labels)
        # Per-pathway detached losses for the revisit-split and all-step breakdowns.
        loss_inf = self.loss_fn(outputs.logits_inference, labels).detach()
        loss_ret = self.loss_fn(outputs.logits_retrieved, labels).detach()
        loss_anc = self.loss_fn(outputs.logits_ancestral, labels).detach()
        pmask = protocol_mask.to(dtype=loss_inf.dtype)
        # Accuracy metrics are owned by the task; the binding delegates to the task evaluator.
        acc_metrics = self._task_binding.evaluate_observation_metrics(outputs, targets)
        return {
            **acc_metrics,
            TEM_LOSS_OBS_NLL_REVISIT: RatioStat(losses.loss_obs_nll_sum.detach(), protocol_count),
            TEM_LOSS_OBS_INFERENCE_REVISIT: RatioStat((loss_inf * pmask).sum(), protocol_count),
            TEM_LOSS_OBS_RETRIEVED_REVISIT: RatioStat((loss_ret * pmask).sum(), protocol_count),
            TEM_LOSS_OBS_ANCESTRAL_REVISIT: RatioStat((loss_anc * pmask).sum(), protocol_count),
            TEM_LOSS_GRID_KL_REVISIT: RatioStat(losses.loss_grid_kl_sum.detach(), protocol_count),
            TEM_LOSS_PLACE_CONSISTENCY_REVISIT: RatioStat(losses.loss_place_consistency_sum.detach(), protocol_count),
            TEM_LOSS_REG_REVISIT: RatioStat(losses.loss_reg_sum.detach(), protocol_count),
            TEM_LOSS_OBS_NLL_ALL: RatioStat(all_loss_sums["loss_obs_nll_sum"], batch_count),
            TEM_LOSS_OBS_INFERENCE_ALL: RatioStat(loss_inf.sum(), batch_count),
            TEM_LOSS_OBS_RETRIEVED_ALL: RatioStat(loss_ret.sum(), batch_count),
            TEM_LOSS_OBS_ANCESTRAL_ALL: RatioStat(loss_anc.sum(), batch_count),
            TEM_LOSS_GRID_KL_ALL: RatioStat(all_loss_sums["loss_grid_kl_sum"], batch_count),
            TEM_LOSS_PLACE_CONSISTENCY_ALL: RatioStat(all_loss_sums["loss_place_consistency_sum"], batch_count),
            TEM_LOSS_REG_ALL: RatioStat(all_loss_sums["loss_reg_sum"], batch_count),
        }  # fmt: skip

    def _build_step_output(self, losses: TEMLosses, metrics: StepMetrics, signals: dict[str, Any], outputs: Any,) -> TEMLossStep:  # fmt: skip  # -------------------------------------------------------------------
        """Wrap losses, metrics, and signals into a :class:`TEMLossStep`."""
        return TEMLossStep(losses=losses, metrics=metrics, outputs=outputs, signals=signals)

    def compute_signals(self, batch: Batch, carry: Any, outputs: TEMStepOutput, losses: TEMLosses, step_output: Any = None, **_: Any,) -> dict[str, Tensor]:  # fmt: skip  # -----------------------------------------------------------------------
        """Compute detached TEM diagnostics and ELBO-style scalar signals."""
        targets = self._task_binding.extract_targets(batch, carry, step_output)
        labels = self._task_binding.extract_observation_id(targets)
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
        return signals

    @staticmethod
    def _masked_sum(values: Tensor, mask: Tensor) -> Tensor:
        """Return the scalar sum over values selected by a boolean batch mask."""
        return (values * mask.to(dtype=values.dtype)).sum()

    def _all_step_loss_sums(self, outputs: TEMStepOutput, labels: Tensor) -> dict[str, Tensor]:
        """Return detached all-step TEM loss sums for diagnostics and metric logging."""
        grid_relation = require_latent_relation(outputs.latent_relations, GRID_TRANSITION_RELATION)
        place_transition_relation = require_latent_relation(outputs.latent_relations, PLACE_TRANSITION_RELATION)  # fmt: skip
        place_sensory_relation = outputs.latent_relations.get(PLACE_SENSORY_RELATION)

        loss_obs_inference = self.loss_fn(outputs.logits_inference, labels)
        loss_obs_retrieved = self.loss_fn(outputs.logits_retrieved, labels)
        loss_obs_ancestral = self.loss_fn(outputs.logits_ancestral, labels)
        loss_obs_nll_sum = self.config.c_obs * (loss_obs_inference + loss_obs_retrieved + loss_obs_ancestral).sum()

        place_transition = sum_latent_terms(mse_consistency, place_transition_relation.lhs, place_transition_relation.rhs)  # fmt: skip
        if place_sensory_relation is not None:
            place_sensory = sum_latent_terms(mse_consistency, place_sensory_relation.lhs, place_sensory_relation.rhs)  # fmt: skip
        else:
            place_sensory = place_transition.new_zeros(place_transition.shape)

        grid_reg_code = get_reg_term(outputs.reg_terms, GRID_REG_TERM)
        if grid_reg_code is None:
            grid_reg_code = grid_relation.lhs

        place_reg_code = get_reg_term(outputs.reg_terms, PLACE_REG_TERM)
        if place_reg_code is None:
            place_reg_code = place_transition_relation.lhs

        grid_reg = self._regularization_terms(grid_reg_code, self.config.grid_reg_norm, self.config.c_grid_reg)
        place_reg = self._regularization_terms(place_reg_code, self.config.place_reg_norm, self.config.c_place_reg)  # fmt: skip

        return {
            "loss_obs_nll_sum": loss_obs_nll_sum.detach(),
            "loss_grid_kl_sum": (self.config.c_grid * sum_latent_terms(mse_consistency, grid_relation.lhs, grid_relation.rhs).sum()).detach(),  # fmt: skip
            "loss_place_consistency_sum": (self.config.c_place * (place_transition + place_sensory).sum()).detach(),
            "loss_reg_sum": (grid_reg.sum() + place_reg.sum()).detach(),
        }

    def _regularization_terms(self, code: LatentCode, norm: RegularizationNorm, coefficient: float,) -> Tensor:  # fmt: skip  # ----------------------------------------------------------------
        """Return weighted per-example regularization for one latent group."""
        if coefficient == 0.0 or norm == "none":
            first_block = code if isinstance(code, Tensor) else next(iter(code))
            return first_block.new_zeros((first_block.shape[0],))
        return coefficient * sum_regularization_terms(code, norm)


# =================================================================================================
# Canonical aliases — preferred over the LossHead-style names in new code.
TEMObjectiveConfig = TEMLossConfig
TEMObjective = TEMLossHead
TEMObjectiveStep = TEMLossStep

__all__ = [
    # canonical names
    "TEMObjectiveBinding",
    "TEMObjectiveConfig",
    "TEMObjective",
    "TEMObjectiveStep",
    # backward-compatible aliases
    "TEMSupervisionBinding",
    "TEMLossConfig",
    "TEMLossHead",
    "TEMLosses",
    "TEMLossStep",
]
