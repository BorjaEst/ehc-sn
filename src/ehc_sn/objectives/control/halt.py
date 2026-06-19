"""Atomic halt-classification BCE objective.

:class:`HaltClassificationObjective` is a reusable ``nn.Module`` that computes
binary cross-entropy over halt logits.  It consumes a
:class:`HaltObjectiveInput` and returns an :class:`ObjectiveResult` with
per-sample terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ehc_sn.objectives.types import ObjectiveResult


# =============================================================================
@dataclass(frozen=True)
class HaltObjectiveInput:
    """Prepared halt-classification input for :class:`HaltClassificationObjective`.

    Attributes:
        logits: Raw halt logits, shape ``(B,)``.
        targets: Float halt targets in ``[0, 1]``, shape ``(B,)``.
        mask: Optional bool mask, shape ``(B,)``. Masked positions contribute
            zero loss. When ``None`` all slots are active.
    """

    logits: Tensor
    targets: Tensor
    mask: Tensor | None = None


# =============================================================================
class HaltClassificationObjective(nn.Module):
    """Binary cross-entropy classification for halt/continue decisions.

    Applies ``F.binary_cross_entropy_with_logits`` with ``reduction="none"``
    and returns per-sample BCE terms.
    """

    def forward(
        self,
        inputs: HaltObjectiveInput,
    ) -> ObjectiveResult:
        """Compute halt BCE for one batch.

        Args:
            inputs: Prepared halt objective input.

        Returns:
            :class:`ObjectiveResult` with:
            - ``loss``: scalar, ``per_sample.sum()``.
            - ``losses["halt_bce"]``: same scalar.
            - ``terms["halt_per_sample"]``: shape ``(B,)``, per-sample BCE
              (zeroed for masked positions when mask is provided).
        """
        loss_per_sample = F.binary_cross_entropy_with_logits(
            input=inputs.logits,
            target=inputs.targets.to(inputs.logits.dtype),
            reduction="none",
        )

        if inputs.mask is not None:
            loss_per_sample = loss_per_sample * inputs.mask.to(
                loss_per_sample.dtype
            )

        loss = loss_per_sample.sum()

        return ObjectiveResult(
            loss=loss,
            losses={"halt_bce": loss},
            terms={"halt_per_sample": loss_per_sample},
        )


# =============================================================================
__all__ = [
    "HaltObjectiveInput",
    "HaltClassificationObjective",
]
