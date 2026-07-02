"""Atomic token-prediction CE objective.

:class:`TokenPredictionObjective` is a reusable ``nn.Module`` that computes
masked cross-entropy loss over token logits.  It consumes a
:class:`TokenObjectiveInput` and returns an :class:`ObjectiveResult` with
per-element and per-sample terms.

Internally delegates to :func:`~ehp_sn.objectives._token.compute_token_loss_unreduced`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.objectives.types import ObjectiveResult

IGNORE_LABEL_ID: int = -100


# =============================================================================
@dataclass(frozen=True)
class TokenObjectiveInput:
    """Prepared token-prediction input for :class:`TokenPredictionObjective`.

    Attributes:
        logits: Token logits, shape ``(B, S, V)``.
        labels: Integer labels, shape ``(B, S)``.  Ignored positions must be
            set to :data:`IGNORE_LABEL_ID` (``-100``).
        weights: Optional per-token loss weights, shape ``(B, S)``.  When
            provided, the per-token loss is multiplied element-wise before
            sequence reduction.
    """

    logits: Tensor
    labels: Tensor
    weights: Tensor | None = None


# =============================================================================
class TokenPredictionObjective(nn.Module):
    """Masked cross-entropy token prediction loss.

    Args:
        loss_fn: Name of the cross-entropy primitive from
            :mod:`ehp_sn.loss.cross_entropy`.
        ignore_label_id: Label index to ignore in loss computation.
            Defaults to :data:`IGNORE_LABEL_ID` (``-100``).
    """

    def __init__(
        self,
        loss_fn: str = "softmax_cross_entropy",
        ignore_label_id: int = IGNORE_LABEL_ID,
    ) -> None:
        super().__init__()
        self._loss_fn_name = loss_fn
        self._ignore_label_id = ignore_label_id

    @property
    def _loss_fn(self) -> Any:
        """Return the configured cross-entropy primitive."""
        import ehc_sn.loss.cross_entropy as ce

        return getattr(ce, self._loss_fn_name)

    def forward(
        self,
        inputs: TokenObjectiveInput,
    ) -> ObjectiveResult:
        """Compute token-prediction loss for one batch.

        Args:
            inputs: Prepared token objective input.

        Returns:
            :class:`ObjectiveResult` with:
            - ``loss``: scalar, ``loss_per_sample.sum()``.
            - ``losses["token"]``: same scalar.
            - ``terms["token_per_element"]``: shape ``(B, S)``, unreduced.
            - ``terms["token_per_sample"]``: shape ``(B,)``,
              ``per_element.sum(-1) / valid_counts``.
        """

        valid = inputs.labels.ne(self._ignore_label_id)
        valid_counts = valid.sum(dim=-1).clamp(min=1).to(inputs.logits.dtype)

        loss_per_element = self._loss_fn(
            logits=inputs.logits,
            labels=inputs.labels,
            ignore_index=self._ignore_label_id,
        )
        if inputs.weights is not None:
            loss_per_element = loss_per_element * inputs.weights
        per_element = loss_per_element * valid.to(loss_per_element.dtype)
        per_sample = per_element.sum(dim=-1) / valid_counts
        loss = per_sample.sum()

        return ObjectiveResult(
            loss=loss,
            losses={"token": loss},
            terms={
                "token_per_element": per_element,
                "token_per_sample": per_sample,
            },
        )


# =============================================================================
__all__ = [
    "TokenObjectiveInput",
    "TokenPredictionObjective",
]
