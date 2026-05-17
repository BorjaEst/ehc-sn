"""Shared token-supervision objective abstractions.

This module defines the :class:`TokenSupervisionBinding` protocol and the
:class:`TokenObjectiveBase` family layer (also exported as ``TokenObjectiveBase``).
Every concrete objective must supply an explicit binding; there is no generic
default because generic code cannot know what a task payload looks like.  Bind
task-specific extraction in the objective layer (e.g.
:class:`~ehc_sn.objectives.act.ACTObjective`) or in an adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch
from pydantic import BaseModel
from torch import Tensor

import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.objectives._base import BaseObjective
from ehc_sn.rollouts import StepRecord
from ehc_sn.training.types import (
    RatioStat,
    RolloutAgg,
    StepMetrics,
    TokenAgg,
    TransitionAgg,
)
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin

IGNORE_LABEL_ID: int = -100


# =============================================================================
@dataclass(frozen=True)
class TokenLosses(DetachMixin, ABC):
    """Shared parent loss structure for token-family heads."""


# =============================================================================
@dataclass(frozen=True)
class TokenLossStep:
    """Container for the outputs of one token-supervision step."""

    losses: TokenLosses
    metrics: StepMetrics
    outputs: Optional[Any] = None
    signals: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.signals is None:
            object.__setattr__(self, "signals", {})

    @property
    def loss(self) -> Tensor:
        """Scalar loss for this step, used for back-propagation."""
        return self.losses.total


# =============================================================================
@dataclass(frozen=True)
class TokenTerms:
    """ """


# =============================================================================
@dataclass(frozen=True)
class TokenContext:
    """ """

    targets: Any
    protocol_mask: Tensor
    latent_relations: dict[str, Any]
    reg_terms: Optional[dict[str, Any]] = None


# =============================================================================
@dataclass(frozen=True)
class TokenOutputs:
    """ """


# =============================================================================
class TokenSupervisionBinding[TargetsT](Protocol):
    """Extraction seam for token-supervised rollout objectives."""

    def extract_logits(  # ----------------------------------------------------
        self,
        batch: Batch,
        carry: Any,
        step_output: Any,
    ) -> Tensor:
        """Return supervised logits for one executed step."""

    def extract_targets(  # ---------------------------------------------------
        self,
        batch: Batch,
        carry: Any,
        step_output: Any,
    ) -> TargetsT:
        """Return task-owned supervision targets for one executed step."""

    def evaluate_sequences(  # ------------------------------------------------
        self,
        logits: Tensor,
        targets: TargetsT,
    ) -> "AccuracyStats":
        """Return sequence-level accuracy statistics for one executed step."""


# =============================================================================
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


# =============================================================================
class TokenObjectiveBase[ConfigT: BaseModel](BaseObjective[ConfigT]):
    """Rollout head specialization for token-supervised loss with generic metrics.

    Family-level output ...
    """

    @property
    def token_loss_fn(self) -> Any:
        """Return the configured token-level loss function."""
        return getattr(cross_entropy_module, self._config.function)

    @property
    def token_binding(self) -> TokenSupervisionBinding[Any]:
        """Return the token-supervision extraction binding for this objective."""
        return self._token_binding

    def evaluate_step(  # ------------------------------------------------------
        self,
        record: StepRecord,
        **options: Any,
    ) -> Any:
        """Score one executed token-supervision step."""
        step_output = record.outputs  # original controller output
        outputs = getattr(step_output, "backbone_output", step_output)

        # --- single scored-step computation ---
        context = self.build_context(record, outputs, **options)
        terms = self.compute_terms(outputs, context, **options)
        losses = self.compute_losses(terms, context, **options)

        # --- metrics from precomputed losses/terms ---
        metrics = self.evaluate_metrics(record, context, terms, **options)

        # --- signals from precomputed losses/context (no re-extraction) ---
        signals = self.compute_signals(outputs, losses, **options)

        return self.build_output(losses, metrics, signals, outputs)

    def build_context(  # -----------------------------------------------------
        self,
        record: StepRecord,
        outputs: TokenOutputs,
        **options: Any,
    ) -> TokenContext:
        """Extract a token context from the current step."""
        raise NotImplementedError

    def compute_terms(  # -----------------------------------------------------
        self,
        outputs: TokenOutputs,
        context: TokenContext,
        **options: Any,
    ) -> dict[str, Any]:
        """Extract token-family loss terms from the current step."""
        raise NotImplementedError

    def compute_losses(  # ----------------------------------------------------
        self,
        terms: TokenTerms,
        context: TokenContext,
        **options: Any,
    ) -> TokenLosses:
        """Return token-family losses for the current step."""
        raise NotImplementedError

    def evaluate_metrics(  # --------------------------------------------------
        self,
        record: StepRecord,
        context: TokenContext,
        terms: TokenTerms,
        **options: Any,
    ) -> StepMetrics:
        """Return token-family metrics for the current step."""
        raise NotImplementedError

    def compute_signals(  # ---------------------------------------------------
        self,
        outputs: TokenOutputs,
        losses: TokenLosses,
        **options: Any,
    ) -> dict[str, Tensor]:
        """Return detached generic token-family diagnostic signals."""
        raise NotImplementedError

    def build_output(  # ------------------------------------------------------
        self,
        losses: TokenLosses,
        metrics: StepMetrics,
        signals: dict[str, Tensor],
        outputs: TokenOutputs,
    ) -> Any:
        """Wrap losses, metrics, and signals into the concrete step-output type."""
        raise NotImplementedError


# =============================================================================
def compute_accuracy_stats(  # ------------------------------------------------
    logits_lm: Tensor,
    labels: Tensor,
    *,
    ignore_label_id: int = IGNORE_LABEL_ID,
) -> AccuracyStats:
    """Compute masked token correctness statistics out of graph."""
    mask = labels != ignore_label_id
    is_correct = mask & (torch.argmax(logits_lm, dim=-1) == labels)
    return AccuracyStats(mask=mask, is_correct=is_correct)


# =============================================================================
def compute_lm_loss_sum(  # ---------------------------------------------------
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


# =============================================================================
def build_token_step_metrics(  # ----------------------------------------------
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


# =============================================================================
__all__ = [
    "IGNORE_LABEL_ID",
    "AccuracyStats",
    "TokenSupervisionBinding",
    "TokenLosses",
    "TokenObjectiveBase",
    "build_token_step_metrics",
    "compute_accuracy_stats",
    "compute_lm_loss_sum",
]
