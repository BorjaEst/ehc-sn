"""Shared token-supervision head abstractions.

This module defines the family layer used by rollout heads whose training
objective includes token-level supervision over ``outputs.lm_logits``.
Concrete ACT and RL heads build on top of this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from pydantic import BaseModel
from torch import Tensor

import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.heads._base import BaseLossHead, ControllerWithInitialState
from ehc_sn.training.types import RatioStat, RolloutAgg, StepMetrics, TokenAgg, TransitionAgg

IGNORE_LABEL_ID: int = -100


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
def compute_accuracy_stats(  # --------------------------------------------------------------------
    logits_lm: Tensor, labels: Tensor, *,
    ignore_label_id: int = IGNORE_LABEL_ID,
) -> AccuracyStats:  # fmt: skip
    """Compute masked token correctness statistics out of graph."""
    mask = labels != ignore_label_id
    is_correct = mask & (torch.argmax(logits_lm, dim=-1) == labels)
    return AccuracyStats(mask=mask, is_correct=is_correct)


# =================================================================================================
def compute_lm_loss_sum(  # -----------------------------------------------------------------------
    loss_fn: Any, logits_lm: Tensor, labels: Tensor, stats: AccuracyStats, *,
    ignore_label_id: int = IGNORE_LABEL_ID,
) -> Tensor:  # fmt: skip
    """Compute the summed supervised LM loss over the batch."""
    loss_per_token = loss_fn(logits_lm, labels, ignore_index=ignore_label_id)
    loss_per_seq = loss_per_token.sum(-1) / stats.loss_counts.clamp_min(1)
    return loss_per_seq.sum()


# =================================================================================================
def build_rollout_token_aggs(  # -------------------------------------------------------------------
    *, steps: Tensor, completed: Tensor, stats: AccuracyStats,
) -> tuple[RolloutAgg, TokenAgg, TransitionAgg, TokenAgg]:  # fmt: skip
    """Build episode-level and step-level aggregate metrics for a token step."""
    eligible_mask = stats.loss_counts > 0
    completed_mask = completed & eligible_mask
    completed_weights = completed_mask.to(torch.float32)
    eligible_weights = eligible_mask.to(torch.float32)

    token_correct_per_seq = stats.is_correct.to(torch.float32).sum(-1)
    token_count_per_seq = stats.loss_counts.clamp_min(1).to(torch.float32)
    seq_accuracy = token_correct_per_seq / token_count_per_seq

    rollout_agg = RolloutAgg(
        completed_count=completed_weights.sum(),
        eligible_count=eligible_mask.to(torch.float32).sum(),
        accuracy_sum=(seq_accuracy * completed_weights).sum(),
        exact_sum=(stats.seq_is_correct & completed_mask).to(torch.float32).sum(),
        steps_sum=(steps * completed_weights.to(steps.dtype)).sum(),
    )
    episode_token_agg = TokenAgg(
        token_correct_sum=(token_correct_per_seq * completed_weights).sum(),
        token_count_sum=(token_count_per_seq * completed_weights).sum(),
    )
    step_agg = TransitionAgg(
        evaluated_count=eligible_weights.sum(),
        eligible_count=eligible_weights.sum(),
        accuracy_sum=(seq_accuracy * eligible_weights).sum(),
        exact_sum=(stats.seq_is_correct & eligible_mask).to(torch.float32).sum(),
        steps_sum=(steps * eligible_weights.to(steps.dtype)).sum(),
    )
    step_token_agg = TokenAgg(
        token_correct_sum=(token_correct_per_seq * eligible_weights).sum(),
        token_count_sum=(token_count_per_seq * eligible_weights).sum(),
    )
    return rollout_agg, episode_token_agg, step_agg, step_token_agg


# =================================================================================================
class TokenLossHeadBase[ControllerT: ControllerWithInitialState, ConfigT: BaseModel](
    BaseLossHead[ControllerT, ConfigT]
):  # fmt: skip
    """Rollout head specialization for token-supervised loss with generic metrics."""

    @property
    def loss_fn(self) -> Any:
        """Return the configured token-level loss function."""
        return getattr(cross_entropy_module, self._config.function)

    def forward(  # -------------------------------------------------------------------------------
        self, batch: Any, carry: Any, **options: Any,
    ) -> tuple[Any, Any, bool]:  # fmt: skip
        """Run the controller and apply the token-supervision pipeline."""
        carry, outputs = self.controller.step(carry, batch, **options)
        step_output = self._run_token_step(batch, carry, outputs)
        return step_output, carry, bool(carry.halted.all())

    def _run_token_step(  # -----------------------------------------------------------------------
        self, batch: Any, carry: Any, outputs: Any, **loss_options: Any,
    ) -> Any:  # fmt: skip
        """Execute the shared token-supervision pipeline once outputs exist."""
        labels = carry.data["labels"]
        with torch.no_grad():
            stats = self.compute_accuracy(outputs, labels)
        losses = self.compute_losses(outputs, labels, stats, **loss_options)
        episode_agg, episode_token_agg, step_agg, step_token_agg = build_rollout_token_aggs(
            steps=carry.steps,
            completed=carry.halted,
            stats=stats,
        )
        metric_ratios = self._build_metric_ratios(losses, batch_size=int(carry.halted.shape[0]))
        metrics = StepMetrics(
            episode=episode_agg,
            episode_tokens=episode_token_agg,
            step=step_agg,
            step_tokens=step_token_agg,
            extras=metric_ratios,
        )
        signals = self.compute_signals(carry, outputs, losses)
        return self._build_step_output(losses, metrics, signals, outputs)

    def compute_accuracy(self, outputs: Any, labels: Tensor) -> AccuracyStats:
        """Compute masked token correctness statistics (out of graph)."""
        return compute_accuracy_stats(outputs.lm_logits, labels)

    def compute_lm_loss(self, logits_lm: Tensor, labels: Tensor, stats: AccuracyStats) -> Tensor:
        """Compute the summed supervised token loss for a step."""
        return compute_lm_loss_sum(self.loss_fn, logits_lm, labels, stats)

    def compute_losses(  # ------------------------------------------------------------------------
        self, outputs: Any, labels: Tensor, stats: AccuracyStats, **options: Any,
    ) -> Any:  # fmt: skip
        """Return algorithm-specific loss terms."""
        raise NotImplementedError

    def _build_metric_ratios(  # ------------------------------------------------------------------
        self, losses: Any, *, batch_size: int,
    ) -> dict[str, RatioStat]:  # fmt: skip
        """Pack algorithm-specific ratio metrics for logging."""
        raise NotImplementedError

    def _build_step_output(  # --------------------------------------------------------------------
        self, losses: Any, metrics: Any, signals: dict, outputs: Any,
    ) -> Any:  # fmt: skip
        """Wrap losses, metrics, and signals into the concrete step-output type."""
        raise NotImplementedError

    def compute_signals(  # -----------------------------------------------------------------------
        self, carry: Any, outputs: Any, losses: Any,
    ) -> dict:  # fmt: skip
        """Return a dict of scalar diagnostic tensors for logging."""
        return {}


# =================================================================================================
__all__ = [
    "AccuracyStats", "IGNORE_LABEL_ID", "TokenLossHeadBase", "build_rollout_token_aggs",
    "compute_accuracy_stats", "compute_lm_loss_sum",
]  # fmt: skip
