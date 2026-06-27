"""Generic ACT scorer — composition of task, halt, and continuation losses.

This module owns only the ACT-level loss composition:

    total_loss = task_loss_sum + c_halt * halt_loss + c_continue * continue_loss

It does NOT know about:

- task modalities (token, field, sequence, structured)
- task attribute names (firing_field, trajectory_field, labels)
- supervision struct shapes
- task-output extraction
- controller internals

Those are owned by ``objectives/task/`` evaluators and the Lightning
orchestration layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.metrics.keys import (
    ACT_LOSS_Q_CONTINUE,
    ACT_LOSS_Q_DONE,
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
from ehc_sn.metrics.token import build_token_step_metrics
from ehc_sn.objectives.contracts import (
    ACTControlPrediction,
    ACTSupervisedScoringInput,
    TaskStepEvaluation,
)
from ehc_sn.objectives.control.halt import (
    HaltClassificationObjective,
    HaltObjectiveInput,
)
from ehc_sn.rollouts.runtime import StepRecord
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
def _validate_ratio_extras(  # -------------------------------------------------
    extras: Mapping[str, RatioStat],
) -> None:
    """Validate task-provided ratio extras for structural correctness.

    Raises ``ValueError`` if:
    - a key is empty,
    - a value has non-scalar ``numerator_sum`` or ``denominator_sum``.
    """
    for name, stat in extras.items():
        if not name:
            raise ValueError("Task extra names must be non-empty")
        if stat.numerator_sum.ndim != 0:
            raise ValueError(
                f"Task extra {name!r} numerator must be scalar, "
                f"got shape {tuple(stat.numerator_sum.shape)}"
            )
        if stat.denominator_sum.ndim != 0:
            raise ValueError(
                f"Task extra {name!r} denominator must be scalar, "
                f"got shape {tuple(stat.denominator_sum.shape)}"
            )


# =============================================================================
class ACTSupervisedScorerConfig(BaseModel, extra="forbid", frozen=True):
    """Configuration for generic ACT loss composition.

    Owns only the coefficients that compose task loss with control
    losses.  Task-specific evaluation configuration belongs in the
    corresponding ``*TaskEvaluatorConfig``.

    Attributes:
        task_loss_coefficient: Weight applied to the task loss sum.
        halt_loss_coefficient: Weight applied to the halt BCE.
        continue_loss_coefficient: Weight applied to the continuation BCE.
    """

    task_loss_coefficient: float = Field(default=1.0, ge=0.0)
    halt_loss_coefficient: float = Field(default=0.5, ge=0.0)
    continue_loss_coefficient: float = Field(default=0.5, ge=0.0)


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
    """Generic ACT scorer — task-agnostic loss composition.

    Composes a pre-computed task evaluation with halt and continuation
    losses.  Does NOT own task-specific objectives, supervision
    interpretation, or output-name probing.

    Example::

        scorer = ACTSupervisedScorer(c_task=1.0, c_halt=0.5, c_continue=0.5)
        step = scorer.evaluate_step(
            task=task_eval,       # TaskStepEvaluation
            control=control_pred, # ACTControlPrediction
        )
    """

    def __init__(  # ----------------------------------------------------------
        self,
        halt_objective: HaltClassificationObjective | None = None,
        c_task: float = 1.0,
        c_halt: float = 0.5,
        c_continue: float = 0.5,
    ) -> None:
        super().__init__()
        self._halt = halt_objective or HaltClassificationObjective()
        self._c_task = c_task
        self._c_halt = c_halt
        self._c_continue = c_continue

    # ------------------------------------------------------------------

    # Public protocol-conforming entry point (RolloutScorer contract).
    def evaluate_step(  # -----------------------------------------------------
        self,
        record: StepRecord,
        *,
        inputs: ACTSupervisedScoringInput,
    ) -> ACTSupervisedStep:
        """Compose task loss with ACT control losses (RolloutScorer protocol).

        Delegates to :meth:`_evaluate_task_control`.  The ``record``
        parameter is accepted for protocol conformance but unused —
        task evaluation is already complete in ``inputs``.

        Args:
            record: Executed step record (unused — protocol compat).
            inputs: Named container with pre-computed task evaluation
                and control prediction.

        Returns:
            Composed step with all loss terms and signals.
        """
        del record
        return self._evaluate_task_control(
            task=inputs.task,
            control=inputs.control,
        )

    # Private task-agnostic implementation.
    def _evaluate_task_control(  # --------------------------------------------
        self,
        *,
        task: TaskStepEvaluation,
        control: ACTControlPrediction,
    ) -> ACTSupervisedStep:
        """Compose task loss with ACT control losses (task-agnostic).

        This is the core implementation.  ``evaluate_step`` delegates
        to this method after unpacking the named input.

        Args:
            task: Pre-computed task evaluation (loss, completion, metrics).
            control: Control prediction (halt logit, optionally continue).

        Returns:
            Composed step with all loss terms and signals.
        """
        # --- Halt loss -------------------------------------------------------
        halt_result = self._halt(
            HaltObjectiveInput(
                logits=control.halt_logit,
                targets=task.completion_target,
            )
        )
        loss_halt_per_sample = halt_result.terms["halt_per_sample"]

        # --- Continue loss ---------------------------------------------------
        loss_continue_per_sample: Tensor | None = None
        if (
            control.continue_logit is not None
            and task.continuation_target is not None
        ):
            loss_continue_per_sample = F.binary_cross_entropy_with_logits(
                input=control.continue_logit,
                target=task.continuation_target,
                reduction="none",
            )

        # --- Assemble losses ------------------------------------------------
        losses = ACTStepLosses(
            task_sum=task.task_loss_sum,
            halt_sum=loss_halt_per_sample.sum(),
            continue_sum=_maybe_sum(loss_continue_per_sample),
            c_task=self._c_task,
            c_halt=self._c_halt,
            c_continue=self._c_continue,
        )

        # --- Signals ---------------------------------------------------------
        done_logit = control.halt_logit
        cont_logit = control.continue_logit
        if cont_logit is None:
            cont_logit = torch.tensor(0.0, device=done_logit.device)
        signals: dict[str, Tensor] = {
            LOSS_Q_DONE: losses.halt_sum.detach(),
            HALT_LOGIT_MEAN: done_logit.detach().mean(),
            CONTINUE_LOGIT_MEAN: cont_logit.detach().mean(),
        }
        if losses.continue_sum is not None:
            signals["act_loss_q_continue"] = losses.continue_sum.detach()

        # Build StepMetrics — structural forwarding of task extras
        # plus scorer-owned objective/controller metrics.
        B = max(task.completion_target.shape[0], 1)
        bc = torch.tensor(float(B))
        device = task.completion_target.device

        # --- Validate and forward task-provided extras ----------------------
        task_extras = (
            task.task_extras if isinstance(task.task_extras, dict) else {}
        )
        _validate_ratio_extras(task_extras)

        extras: dict[str, RatioStat] = dict(task_extras)

        # --- Scorer-owned extras (collision-checked) ------------------------
        scorer_extras: dict[str, RatioStat] = {
            ACT_LOSS_Q_DONE: RatioStat(
                numerator_sum=losses.halt_sum.detach(),
                denominator_sum=bc,
            ),
            ACT_LOSS_Q_CONTINUE: RatioStat(
                numerator_sum=(
                    losses.continue_sum.detach()
                    if losses.continue_sum is not None
                    else losses.halt_sum.new_zeros(())
                ),
                denominator_sum=bc,
            ),
        }
        collisions = extras.keys() & scorer_extras.keys()
        if collisions:
            raise ValueError(
                f"Task evaluator extras collide with scorer-owned keys: "
                f"{sorted(collisions)}"
            )
        extras.update(scorer_extras)

        if task.accuracy_stats is not None:
            metrics = build_token_step_metrics(
                steps=torch.zeros(B, dtype=torch.long, device=device),
                completed=torch.zeros(B, dtype=torch.bool, device=device),
                stats=task.accuracy_stats,
                extras=extras,
            )
        else:
            metrics = StepMetrics(
                episode=RolloutAgg(
                    completed_count=torch.tensor(0.0),
                    eligible_count=bc,
                    accuracy_sum=torch.tensor(0.0),
                    exact_sum=torch.tensor(0.0),
                    steps_sum=torch.tensor(0.0),
                ),
                episode_tokens=TokenAgg(
                    token_correct_sum=torch.tensor(0.0),
                    token_count_sum=torch.tensor(0.0),
                ),
                step=TransitionAgg(
                    evaluated_count=bc,
                    eligible_count=bc,
                    accuracy_sum=torch.tensor(0.0),
                    exact_sum=torch.tensor(0.0),
                    steps_sum=torch.tensor(0.0),
                ),
                step_tokens=TokenAgg(
                    token_correct_sum=torch.tensor(0.0),
                    token_count_sum=torch.tensor(0.0),
                ),
                extras=extras,
            )

        return ACTSupervisedStep(
            losses=losses,
            metrics=metrics,
            outputs=None,
            target_q=task.continuation_target,
            signals=signals,
        )


# =============================================================================
__all__ = [
    "ACTSupervisedScorer",
    "ACTSupervisedStep",
    "ACTStepLosses",
    "ACTSupervisedScorerConfig",
]
