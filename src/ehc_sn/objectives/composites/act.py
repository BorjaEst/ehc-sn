"""Regime-level ACT scorer combining task and control objectives.

This module owns the composition of supervised task loss (token prediction
or field regression) with ACT computation-control losses (halt BCE, optional
continue bootstrap).

Usage::

    scorer = ACTSupervisedScorer(
        ACTSupervisedScorerConfig(task_modality="field"),
        task_objective=FieldRegressionObjective(),
        halt_objective=HaltClassificationObjective(),
    )
    step = scorer.evaluate_step(record, inputs=scoring_inputs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.metrics.keys import (
    ACT_LOSS_Q_CONTINUE,
    ACT_LOSS_Q_DONE,
    LOSS_TOKEN,
)
from ehc_sn.metrics.signals import (
    CONTINUE_LOGIT_MEAN,
    HALT_LOGIT_MEAN,
    LOSS_Q_DONE,
)
from ehc_sn.metrics.step_metrics import (
    RatioStat,
    RolloutAgg,
    StepMetrics,
    TokenAgg,
    TransitionAgg,
)
from ehc_sn.metrics.token import (
    AccuracyStats,
    build_token_step_metrics,
)
from ehc_sn.objectives.control.halt import (
    HaltClassificationObjective,
    HaltObjectiveInput,
)
from ehc_sn.objectives.supervised.field import (
    FieldObjectiveInput,
    FieldRegressionObjective,
)
from ehc_sn.objectives.supervised.token import (
    TokenObjectiveInput,
    TokenPredictionObjective,
)
from ehc_sn.rollouts.runtime import StepRecord
from ehc_sn.utils.detach import DetachMixin

# =============================================================================
# Per-step typed contracts
# =============================================================================


@dataclass(frozen=True)
class TokenACTInput:
    """Token-prediction supervision for one ACT step."""

    logits: Tensor  # (B, S, V)
    labels: Tensor  # (B, S)
    weights: Tensor | None  # (B, S) optional token weights


@dataclass(frozen=True)
class FieldACTInput:
    """Field-regression supervision for one ACT step."""

    prediction: Tensor  # (B, N)
    target: Tensor  # (B, N)
    mask: Tensor  # (B, N) bool


@dataclass(frozen=True)
class ACTHaltInput:
    """Halt-classification supervision for one ACT step."""

    logits: Tensor  # (B,)
    targets: Tensor  # (B,) float in [0, 1]


@dataclass(frozen=True)
class ACTContinuationInput:
    """Continue-bootstrap supervision for one ACT step."""

    continue_logit: Tensor  # (B,)
    target_q: Tensor  # (B,) detached


@dataclass(frozen=True)
class TokenACTScoringInput:
    """Complete typed scoring input for token-modality ACT steps."""

    prediction: TokenACTInput
    halt: ACTHaltInput
    continuation: ACTContinuationInput | None = None
    accuracy: AccuracyStats | None = None


@dataclass(frozen=True)
class FieldACTScoringInput:
    """Complete typed scoring input for field-modality ACT steps."""

    prediction: FieldACTInput
    halt: ACTHaltInput
    continuation: ACTContinuationInput | None = None


# Union type used by traversal infrastructure.
ACTScoringInput: TypeAlias = TokenACTScoringInput | FieldACTScoringInput


# =============================================================================
class ACTStepOutput(Protocol):
    """Objective-facing output contract for ACT rollout steps."""

    task: object
    action_logits: Tensor
    done_action: int


# =============================================================================
class ACTSupervisedScorerConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`ACTSupervisedScorer`.

    Attributes:
        task_modality: Whether the scorer evaluates token predictions or
            continuous-field predictions.
        c_task: Task-supervision loss coefficient (token CE or field MSE).
        c_halt: Halt BCE (Q-done) loss coefficient.
        c_continue: Q(continue) bootstrap loss coefficient.
        token_loss: Cross-entropy variant for token tasks.
        field_loss: Loss function for field tasks (only ``"mse"`` in v1).
    """

    task_modality: Literal["token", "field"] = Field(
        ...,
        description="Modality of the task objective: 'token' or 'field'.",
    )
    c_task: float = Field(default=1.0, ge=0.0)
    c_halt: float = Field(default=0.5, ge=0.0)
    c_continue: float = Field(default=0.5, ge=0.0)
    token_loss: str = Field(
        default="softmax_cross_entropy",
        description="Token CE loss variant (only used when task_modality='token').",
    )
    field_loss: str = Field(
        default="mse",
        description="Field loss variant (only used when task_modality='field').",
    )


# =============================================================================
@dataclass(frozen=True)
class ACTStepLosses(DetachMixin):
    """Summed-over-batch ACT loss terms shared by token and field tasks."""

    task_sum: Tensor
    halt_sum: Tensor
    continue_sum: Tensor | None

    c_task: float = 1.0
    c_halt: float = 0.5
    c_continue: float = 0.5

    @property
    def total(self) -> Tensor:
        """Total scalar loss for back-propagation."""
        continue_sum = self.continue_sum
        if continue_sum is None:
            continue_sum = torch.tensor(0.0, device=self.task_sum.device)
        return (
            self.c_task * self.task_sum
            + self.c_halt * self.halt_sum
            + self.c_continue * continue_sum
        )


# =============================================================================
@dataclass(frozen=True)
class ACTSupervisedStep:
    """A single rollout/loss step produced by :class:`ACTSupervisedScorer`."""

    losses: ACTStepLosses
    metrics: StepMetrics
    outputs: object | None = None
    target_q: Tensor | None = None
    signals: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.signals is None:
            object.__setattr__(self, "signals", {})

    @property
    def loss(self) -> Tensor:
        """Scalar loss for this step, used for back-propagation."""
        return self.losses.total


# =============================================================================
def _maybe_sum(tensor: Tensor | None) -> Tensor | None:
    """Sum a tensor if it's not None, otherwise return None."""
    if tensor is not None:
        return tensor.sum()
    return None


# =============================================================================
class ACTSupervisedScorer(nn.Module):
    """Regime-level scorer composing task supervision with ACT control.

    Injects a task-specific objective (``TokenPredictionObjective`` or
    ``FieldRegressionObjective``) and shares a common
    ``HaltClassificationObjective``.  The task_losses are combined with
    halt and optional continue losses using configurable coefficients.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: ACTSupervisedScorerConfig,
        *,
        task_objective: nn.Module | None = None,
        halt_objective: HaltClassificationObjective | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._task_modality = config.task_modality

        if task_objective is not None:
            self._task = task_objective
        elif self._task_modality == "token":
            self._task = TokenPredictionObjective(loss_fn=config.token_loss)
        elif self._task_modality == "field":
            self._task = FieldRegressionObjective()
        else:
            raise ValueError(
                f"Unknown task_modality: {config.task_modality!r}."
            )

        self._halt = halt_objective or HaltClassificationObjective()

    @property
    def config(self) -> ACTSupervisedScorerConfig:
        """Return the scorer configuration."""
        return self._config

    # ------------------------------------------------------------------

    def evaluate_step(  # -----------------------------------------------------
        self,
        record: StepRecord,
        *,
        inputs: ACTScoringInput,
    ) -> ACTSupervisedStep:
        """Score one ACT rollout step for token or field tasks.

        Args:
            record: Executed step record.
            inputs: Typed per-step scoring input.

        Returns:
            Scored step with unified :class:`ACTStepLosses`.
        """
        outputs = record.outputs

        # --- Task loss -------------------------------------------------------
        if isinstance(inputs, TokenACTScoringInput):
            token_obj_input = TokenObjectiveInput(
                logits=inputs.prediction.logits,
                labels=inputs.prediction.labels,
                weights=inputs.prediction.weights,
            )
            task_result = self._task(token_obj_input)
            loss_task_per_sample = task_result.terms["token_per_sample"]
            stats: AccuracyStats | None = inputs.accuracy
        else:
            field_obj_input = FieldObjectiveInput(
                prediction=inputs.prediction.prediction,
                target=inputs.prediction.target,
                mask=inputs.prediction.mask,
            )
            task_result = self._task(field_obj_input)
            loss_task_per_sample = task_result.terms["field_per_sample"]
            stats = None

        # --- Halt loss -------------------------------------------------------
        halt_result = self._halt(
            HaltObjectiveInput(
                logits=inputs.halt.logits,
                targets=inputs.halt.targets,
            )
        )
        loss_halt_per_sample = halt_result.terms["halt_per_sample"]

        # --- Continue loss ---------------------------------------------------
        loss_continue_per_sample: Tensor | None = None
        if inputs.continuation is not None:
            loss_continue_per_sample = F.binary_cross_entropy_with_logits(
                input=inputs.continuation.continue_logit,
                target=inputs.continuation.target_q,
                reduction="none",
            )

        # --- Assemble losses ------------------------------------------------
        losses = ACTStepLosses(
            task_sum=loss_task_per_sample.sum(),
            halt_sum=loss_halt_per_sample.sum(),
            continue_sum=_maybe_sum(loss_continue_per_sample),
            c_task=self.config.c_task,
            c_halt=self.config.c_halt,
            c_continue=self.config.c_continue,
        )

        # --- Metrics ---------------------------------------------------------
        B = max(loss_task_per_sample.shape[0], 1)
        snapshot = record.snapshot
        steps_t = getattr(snapshot, "steps", None)
        if steps_t is None:
            steps_t = losses.task_sum.new_zeros(B, dtype=torch.long)
        halted_t = getattr(snapshot, "halted", None)
        if halted_t is None:
            halted_t = torch.zeros(
                B, dtype=torch.bool, device=losses.task_sum.device
            )

        extras: dict[str, RatioStat] = {}
        if isinstance(inputs, TokenACTScoringInput):
            extra_ratios = self._build_token_metric_ratios(losses, B)
            if extra_ratios:
                extras.update(extra_ratios)
            metrics = build_token_step_metrics(
                steps_t,
                halted_t,
                stats,
                extras,
            )
        else:
            metrics = self._build_field_fallback_metrics(
                halted_t,
                steps_t,
                losses,
                B,
            )
            extras["field_mse"] = RatioStat(
                numerator_sum=losses.task_sum.detach(),
                denominator_sum=torch.tensor(float(B)),
            )
            done_logit = outputs.action_logits[..., outputs.done_action]
            extras["q_done_accuracy"] = RatioStat(
                numerator_sum=(
                    (done_logit > 0.0) == (inputs.halt.targets > 0.5)
                )
                .sum()
                .float()
                .detach(),
                denominator_sum=torch.tensor(float(B)),
            )
            metrics = StepMetrics(
                episode=RolloutAgg(
                    completed_count=halted_t.sum(),
                    eligible_count=torch.tensor(float(B)),
                    accuracy_sum=torch.tensor(0.0),
                    exact_sum=torch.tensor(0.0),
                    steps_sum=(steps_t * halted_t).sum(),
                ),
                episode_tokens=TokenAgg(
                    token_correct_sum=torch.tensor(0.0),
                    token_count_sum=torch.tensor(0.0),
                ),
                step=TransitionAgg(
                    evaluated_count=torch.tensor(float(B)),
                    eligible_count=torch.tensor(float(B)),
                    accuracy_sum=torch.tensor(0.0),
                    exact_sum=torch.tensor(0.0),
                    steps_sum=steps_t.sum(),
                ),
                step_tokens=TokenAgg(
                    token_correct_sum=torch.tensor(0.0),
                    token_count_sum=torch.tensor(0.0),
                ),
                extras=extras,
            )

        # --- Signals ---------------------------------------------------------
        done_logit = outputs.action_logits[..., outputs.done_action]
        non_done = [
            i
            for i in range(outputs.action_logits.shape[-1])
            if i != outputs.done_action
        ]
        continue_logit = outputs.action_logits[..., non_done].max(dim=-1).values
        signals: dict[str, Tensor] = {
            LOSS_Q_DONE: losses.halt_sum.detach(),
            HALT_LOGIT_MEAN: done_logit.detach().mean(),
            CONTINUE_LOGIT_MEAN: continue_logit.detach().mean(),
            ACT_LOSS_Q_CONTINUE: (
                losses.continue_sum.detach()
                if losses.continue_sum is not None
                else losses.task_sum.new_zeros(())
            ),
        }

        target_q = (
            inputs.continuation.target_q
            if inputs.continuation is not None
            else None
        )

        return ACTSupervisedStep(
            losses=losses,
            metrics=metrics,
            outputs=outputs,
            target_q=target_q,
            signals=signals,
        )

    # ------------------------------------------------------------------
    def _build_token_metric_ratios(
        self,
        losses: ACTStepLosses,
        batch_size: int,
    ) -> dict[str, RatioStat]:
        """Build detached ratio metrics for token tasks."""
        bc = losses.task_sum.new_tensor(float(batch_size))
        cs = losses.continue_sum
        return {
            LOSS_TOKEN: RatioStat(losses.task_sum.detach(), bc),
            ACT_LOSS_Q_DONE: RatioStat(losses.halt_sum.detach(), bc),
            ACT_LOSS_Q_CONTINUE: RatioStat(
                (
                    cs.detach()
                    if cs is not None
                    else losses.task_sum.new_zeros(())
                ),
                bc,
            ),
        }

    def _build_field_fallback_metrics(
        self,
        halted: Tensor,
        steps: Tensor,
        losses: ACTStepLosses,
        batch_size: int,
    ) -> StepMetrics:
        """Build minimal StepMetrics when token stats are unavailable."""
        B = max(batch_size, 1)
        return StepMetrics(
            episode=RolloutAgg(
                completed_count=halted.sum(),
                eligible_count=torch.tensor(float(B)),
                accuracy_sum=torch.tensor(0.0),
                exact_sum=torch.tensor(0.0),
                steps_sum=(steps * halted).sum(),
            ),
            episode_tokens=TokenAgg(
                token_correct_sum=torch.tensor(0.0),
                token_count_sum=torch.tensor(0.0),
            ),
            step=TransitionAgg(
                evaluated_count=torch.tensor(float(B)),
                eligible_count=torch.tensor(float(B)),
                accuracy_sum=torch.tensor(0.0),
                exact_sum=torch.tensor(0.0),
                steps_sum=steps.sum(),
            ),
            step_tokens=TokenAgg(
                token_correct_sum=torch.tensor(0.0),
                token_count_sum=torch.tensor(0.0),
            ),
            extras={},
        )


# =============================================================================
__all__ = [
    "ACTSupervisedScorerConfig",
    "ACTSupervisedScorer",
    "ACTSupervisedStep",
    "ACTStepLosses",
    "ACTStepOutput",
    "ACTScoringInput",
    "TokenACTScoringInput",
    "FieldACTScoringInput",
    "TokenACTInput",
    "FieldACTInput",
    "ACTHaltInput",
    "ACTContinuationInput",
]
