"""MazeHard token-prediction evaluator for ACT training regimes.

Wraps ``TokenPredictionObjective`` with exact-match completion
semantics to produce a ``TaskStepEvaluation`` for the generic ACT
scorer.

Ownership boundary:
- This evaluator knows about ``MazeHardTaskOutput`` and
  ``MazeHardTokenSupervision``.
- It does not know about HRM, ACT controllers, rollouts, or Lightning.
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.metrics.token import (
    compute_accuracy_stats,
)
from ehc_sn.objectives.contracts import (
    TaskStepEvaluation,
    TaskStepEvaluator,
)
from ehc_sn.objectives.supervised.token import (
    TokenObjectiveInput,
    TokenPredictionObjective,
)
from ehc_sn.tasks.mazehard.contracts import (
    MAZE_HARD_IGNORE_LABEL_ID,
    MazeHardTaskOutput,
)
from ehc_sn.tasks.mazehard.supervision import (
    MazeHardTokenSupervision,
)


class MazeHardTaskEvaluatorConfig(BaseModel, extra="forbid"):
    """Configuration for MazeHard token task evaluation.

    Attributes:
        token_loss: Cross-entropy variant for token prediction.
    """

    token_loss: Literal["softmax_cross_entropy", "stablemax_cross_entropy"] = (
        "stablemax_cross_entropy"
    )


class MazeHardTokenEvaluator:
    """Evaluates MazeHard token predictions for ACT training.

    The completion target is an exact-match indicator (1.0 if all
    non-ignored tokens are correct, 0.0 otherwise).
    """

    def __init__(
        self,
        config: MazeHardTaskEvaluatorConfig | None = None,
        token_objective: TokenPredictionObjective | None = None,
    ) -> None:
        self._token = token_objective or TokenPredictionObjective(
            loss_fn=(
                config.token_loss
                if config is not None
                else "softmax_cross_entropy"
            )
        )

    # ── TaskStepEvaluator protocol ──────────────────────────────────────────

    def evaluate(
        self,
        *,
        task_output: MazeHardTaskOutput,
        supervision: MazeHardTokenSupervision,
    ) -> TaskStepEvaluation:
        """Evaluate one step of mazehard token predictions.

        Args:
            task_output: Must have ``task_logits`` shape ``(B, S, V)``.
            supervision: Must have ``labels`` and ``weights``.

        Returns:
            TaskStepEvaluation with token CE loss and exact-match
            completion target.
        """
        result = self._token(
            TokenObjectiveInput(
                logits=task_output.task_logits,
                labels=supervision.labels,
                weights=supervision.weights,
            )
        )
        per_sample = result.terms["token_per_sample"]
        loss_sum = per_sample.sum()

        # Count non-ignored tokens for normalization.
        count = (
            (supervision.labels != MAZE_HARD_IGNORE_LABEL_ID).sum().clamp(min=1)
        )

        # Completion = exact sequence match (all non-ignored correct).
        stats = compute_accuracy_stats(
            logits_token=task_output.task_logits,
            labels=supervision.labels,
        )
        seq_correct = stats.seq_is_correct.float()  # (B,)
        completion_target = seq_correct.detach()

        metrics = {
            "token_ce_sum": loss_sum.detach(),
            "token_ce_count": count.detach(),
        }

        return TaskStepEvaluation(
            task_loss_sum=loss_sum,
            task_loss_count=count,
            completion_target=completion_target,
            continuation_target=None,
            accuracy_stats=stats,
            metrics=metrics,
        )


__all__ = ["MazeHardTokenEvaluator"]
