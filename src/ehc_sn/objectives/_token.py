"""Shared token-supervision objective abstractions.

This module defines the :class:`TokenSupervisionBinding` protocol and the
:class:`TokenObjectiveBase` family layer (also exported as ``TokenObjectiveBase``).
Every concrete objective must supply an explicit binding; there is no generic
default because generic code cannot know what a task payload looks like.  Bind
task-specific extraction in the objective layer (e.g.
:class:`~ehc_sn.objectives.act.ACTObjective`) or in an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from pydantic import BaseModel
from torch import Tensor

import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.objectives._base import BaseObjective
from ehc_sn.rollouts import StepRecord
from ehc_sn.training.types import RatioStat, RolloutAgg, StepMetrics, TokenAgg, TransitionAgg
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin

IGNORE_LABEL_ID: int = -100


# =================================================================================================
class TokenSupervisionBinding[TargetsT](Protocol):
    """Extraction seam for token-supervised rollout objectives."""

    def extract_logits(self, batch: Batch, carry: Any, step_output: Any) -> Tensor:
        """Return supervised logits for one executed step."""

    def extract_targets(self, batch: Batch, carry: Any, step_output: Any) -> TargetsT:
        """Return task-owned supervision targets for one executed step."""

    def evaluate_sequences(self, logits: Tensor, targets: TargetsT) -> "AccuracyStats":
        """Return sequence-level accuracy statistics for one executed step."""


# =================================================================================================
@dataclass(frozen=True)
class AccuracyStats:
    """Token-level correctness statistics shared by token-supervised heads."""

    mask: Tensor
    is_correct: Tensor

    @property
    def loss_counts(self) -> Tensor:
        """Number of eligible tokens per sequence, shape ``(B,)``."""
        return self.mask.sum(-1)

    @property
    def loss_divisor(self) -> Tensor:
        """Safe divisor for per-sequence averages, shape ``(B, 1)``."""
        return self.loss_counts.clamp_min(1).unsqueeze(-1)

    @property
    def seq_is_correct(self) -> Tensor:
        """Whether every eligible token in a sequence was predicted correctly."""
        return self.is_correct.sum(-1) == self.loss_counts


# =================================================================================================
@dataclass(frozen=True)
class TokenLosses(DetachMixin):
    """Shared parent loss structure for token-family heads."""


# =================================================================================================
class TokenObjectiveBase[ConfigT: BaseModel](BaseObjective[ConfigT]):  # fmt: skip
    """Rollout head specialization for token-supervised loss with generic metrics."""

    def __init__(
        self,
        config: ConfigT,
        *,
        token_binding: TokenSupervisionBinding[Any],
    ) -> None:
        """Create a token-supervised objective with an explicit extraction binding.

        Args:
            config: Head-specific configuration.
            token_binding: Binding that knows how to extract logits and targets
                from this controller's step output.  Must be supplied explicitly;
                there is no generic default.
        """
        super().__init__(config=config)
        self._token_binding = token_binding

    @property
    def token_binding(self) -> TokenSupervisionBinding[Any]:
        """Return the token-supervision extraction binding for this objective."""
        return self._token_binding

    @property
    def loss_fn(self) -> Any:
        """Return the configured token-level loss function."""
        return getattr(cross_entropy_module, self._config.function)

    def evaluate_step(  # ------------------------------------------------------------------------
        self,
        record: StepRecord,
        **options: Any,
    ) -> Any:
        """Score one executed token-supervision step."""
        return self._run_token_step(record.batch, record.carry, record.outputs, **options)

    def _run_token_step(  # -----------------------------------------------------------------------
        self,
        batch: Batch,
        carry: Any,
        outputs: Any,
        **loss_options: Any,
    ) -> Any:
        """Execute the shared token-supervision pipeline once outputs exist."""
        targets = self.token_binding.extract_targets(batch, carry, outputs)
        logits = self.token_binding.extract_logits(batch, carry, outputs)
        with torch.no_grad():
            stats = self.compute_accuracy(logits, targets)

        losses = self.compute_losses(outputs, targets, stats, logits=logits, **loss_options)
        extras = self._build_metric_ratios(losses, batch_size=int(carry.halted.shape[0]))
        metrics = build_token_step_metrics(carry.steps, carry.halted, stats, extras)
        signals = self.compute_signals(batch, carry, outputs, losses, logits=logits, targets=targets, stats=stats, **loss_options)
        return self._build_step_output(losses, metrics, signals, outputs, logits=logits, targets=targets, stats=stats, **loss_options)

    def compute_accuracy(  # ----------------------------------------------------------------------
        self,
        logits: Tensor,
        targets: Any,
    ) -> AccuracyStats:
        """Compute masked token correctness statistics (out of graph)."""
        return self.token_binding.evaluate_sequences(logits, targets)

    def compute_lm_loss(  # -----------------------------------------------------------------------
        self,
        logits_lm: Tensor,
        labels: Tensor,
        stats: AccuracyStats,
    ) -> Tensor:
        """Compute the summed supervised token loss for a step."""
        return compute_lm_loss_sum(self.loss_fn, logits_lm, labels, stats)

    def compute_losses(  # ------------------------------------------------------------------------
        self,
        outputs: Any,
        targets: Any,
        stats: AccuracyStats,
        **options: Any,
    ) -> Any:
        """Return algorithm-specific loss terms."""
        raise NotImplementedError

    def _build_metric_ratios(  # ------------------------------------------------------------------
        self,
        losses: Any,
        *,
        batch_size: int,
    ) -> dict[str, RatioStat]:
        """Pack algorithm-specific ratio metrics for logging."""
        raise NotImplementedError

    def _build_step_output(  # --------------------------------------------------------------------
        self,
        losses: Any,
        metrics: Any,
        signals: dict,
        outputs: Any,
        **context: Any,
    ) -> Any:
        """Wrap losses, metrics, and signals into the concrete step-output type."""
        raise NotImplementedError

    def compute_signals(  # -----------------------------------------------------------------------
        self,
        batch: Batch,
        carry: Any,
        outputs: Any,
        losses: Any,
        **context: Any,
    ) -> dict:
        """Return a dict of scalar diagnostic tensors for logging."""
        return {}


# =================================================================================================
def compute_accuracy_stats(  # --------------------------------------------------------------------
    logits_lm: Tensor,
    labels: Tensor,
    *,
    ignore_label_id: int = IGNORE_LABEL_ID,
) -> AccuracyStats:
    """Compute masked token correctness statistics out of graph."""
    mask = labels != ignore_label_id
    is_correct = mask & (torch.argmax(logits_lm, dim=-1) == labels)
    return AccuracyStats(mask=mask, is_correct=is_correct)


# =================================================================================================
def compute_lm_loss_sum(  # -----------------------------------------------------------------------
    loss_fn: Any,
    logits_lm: Tensor,
    labels: Tensor,
    stats: AccuracyStats,
    *,
    ignore_label_id: int = IGNORE_LABEL_ID,
) -> Tensor:
    """Compute the summed supervised LM loss over the batch."""
    loss_per_token = loss_fn(logits_lm, labels, ignore_index=ignore_label_id)
    loss_per_seq = loss_per_token.sum(-1) / stats.loss_counts.clamp_min(1)
    return loss_per_seq.sum()


# =================================================================================================
def build_token_step_metrics(  # ------------------------------------------------------------------
    steps: Tensor,
    completed: Tensor,
    stats: AccuracyStats,
    extras: dict[str, RatioStat],
) -> StepMetrics:
    """Build generic per-step token-supervision metrics."""
    eligible_mask = stats.loss_counts > 0
    completed_mask = completed & eligible_mask

    eligible_weights = eligible_mask.to(torch.float32)
    completed_weights = completed_mask.to(torch.float32)

    token_correct_per_seq = stats.is_correct.to(torch.float32).sum(-1)
    token_count_per_seq = stats.loss_counts.clamp_min(1).to(torch.float32)
    seq_accuracy = token_correct_per_seq / token_count_per_seq
    seq_exact = stats.seq_is_correct.to(torch.float32)

    return StepMetrics(
        episode=RolloutAgg(
            completed_count=completed_weights.sum(),
            eligible_count=eligible_weights.sum(),
            accuracy_sum=(seq_accuracy * completed_weights).sum(),
            exact_sum=(seq_exact * completed_weights).sum(),
            steps_sum=(steps * completed_weights.to(steps.dtype)).sum(),
        ),
        episode_tokens=TokenAgg(
            token_correct_sum=(token_correct_per_seq * completed_weights).sum(),
            token_count_sum=(token_count_per_seq * completed_weights).sum(),
        ),
        step=TransitionAgg(
            evaluated_count=eligible_weights.sum(),
            eligible_count=eligible_weights.sum(),
            accuracy_sum=(seq_accuracy * eligible_weights).sum(),
            exact_sum=(seq_exact * eligible_weights).sum(),
            steps_sum=(steps * eligible_weights.to(steps.dtype)).sum(),
        ),
        step_tokens=TokenAgg(
            token_correct_sum=(token_correct_per_seq * eligible_weights).sum(),
            token_count_sum=(token_count_per_seq * eligible_weights).sum(),
        ),
        extras=extras,
    )


# =================================================================================================
__all__ = [
    "IGNORE_LABEL_ID", "AccuracyStats", "TokenSupervisionBinding", "TokenLosses", "TokenObjectiveBase",
     "build_token_step_metrics", "compute_accuracy_stats", "compute_lm_loss_sum",
]  # fmt: skip
