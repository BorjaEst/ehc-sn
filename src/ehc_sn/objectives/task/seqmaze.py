"""SeqMaze token-prediction evaluator for ACT training.

Evaluates path-sequence token predictions with EOS-canonicalized
exact-match completion semantics.

Ownership boundary:
- This evaluator knows about ``SeqMazeTaskOutput`` and
  ``SeqMazeTokenSupervision``.
- It does not know about HRM, ACT controllers, rollouts, or Lightning.
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.metrics.token import (
    build_token_step_metrics,
    compute_accuracy_stats,
)
from ehc_sn.objectives.contracts import (
    RatioStat,
    TaskStepEvaluation,
    TaskStepEvaluator,
)
from ehc_sn.objectives.supervised.token import (
    TokenObjectiveInput,
    TokenPredictionObjective,
)
from ehc_sn.tasks.seqmaze.contracts import SeqMazeTaskOutput
from ehc_sn.tasks.seqmaze.supervision import SeqMazeTokenSupervision


class SeqMazeTaskEvaluatorConfig(BaseModel, extra="forbid"):
    """Configuration for SeqMaze path-prediction task evaluation.

    Attributes:
        token_loss: Cross-entropy variant for path token prediction.
    """

    token_loss: Literal["softmax_cross_entropy", "stablemax_cross_entropy"] = (
        "softmax_cross_entropy"
    )


class SeqMazeTaskEvaluator:
    """Evaluates SeqMaze path predictions for ACT training.

    Completion target is EOS-canonicalized exact-match correctness.
    """

    def __init__(
        self,
        config: SeqMazeTaskEvaluatorConfig | None = None,
        token_objective: TokenPredictionObjective | None = None,
    ) -> None:
        c = config or SeqMazeTaskEvaluatorConfig()
        self._token = token_objective or TokenPredictionObjective(
            loss_fn=c.token_loss
        )

    # ── TaskStepEvaluator protocol ──────────────────────────────────────────

    def evaluate(
        self,
        *,
        task_output: SeqMazeTaskOutput,
        supervision: SeqMazeTokenSupervision,
    ) -> TaskStepEvaluation:
        """Evaluate one step of seqmaze path predictions.

        Args:
            task_output: Must have ``path_logits`` shape ``(B, T, V)``.
            supervision: Must have ``labels`` and ``weights``.

        Returns:
            TaskStepEvaluation with token CE loss and EOS-canonicalized
            exact-match completion target.
        """
        result = self._token(
            TokenObjectiveInput(
                logits=task_output.path_logits,
                labels=supervision.labels,
                weights=supervision.weights,
            )
        )
        per_sample = result.terms["token_per_sample"]
        loss_sum = per_sample.sum()

        # Completion = exact sequence match (all non-ignored correct).
        stats = compute_accuracy_stats(
            logits_token=task_output.path_logits,
            labels=supervision.labels,
        )
        seq_correct = stats.seq_is_correct.float()
        completion_target = seq_correct.detach()

        # Delegate metrics to the canonical consumer.
        device = stats.is_correct.device
        step_metrics = build_token_step_metrics(
            torch.zeros(seq_correct.shape[0], dtype=torch.long, device=device),
            torch.zeros(seq_correct.shape[0], dtype=torch.bool, device=device),
            stats,
            extras={},
        )
        extras = step_metrics.extras or {}
        metrics = {
            "token_ce_sum": loss_sum.detach(),
        }
        for key, stat in extras.items():
            metrics[key] = stat.numerator_sum.detach()

        return TaskStepEvaluation(
            task_loss_sum=loss_sum,
            task_loss_count=supervision.labels.numel(),
            completion_target=completion_target,
            continuation_target=1.0 - completion_target,
            metrics=metrics,
            task_extras={
                "loss_token": RatioStat(
                    numerator_sum=loss_sum.detach(),
                    denominator_sum=torch.tensor(
                        float(supervision.labels.numel())
                    ),
                ),
            },
        )


__all__ = [
    "SeqMazeTaskEvaluatorConfig",
    "SeqMazeTaskEvaluator",
]
