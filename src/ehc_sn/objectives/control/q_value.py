"""Atomic Q-value MSE objective.

:class:`QValueObjective` is a reusable ``nn.Module`` that computes MSE on
Q-values at the selected action indices.  It consumes a
:class:`SelectedQInput` and returns an :class:`ObjectiveResult` with
per-sample terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor, nn

from ehc_sn.objectives.types import ObjectiveResult


# =============================================================================
@dataclass(frozen=True)
class SelectedQInput:
    """Prepared Q-value input for :class:`QValueObjective`.

    Attributes:
        q_values: Q-values over all actions, shape ``(B, A)``.
        actions: Selected action indices, shape ``(B,)`` int64.
        targets: Regression targets (returns), shape ``(B,)``.
            Detached upstream by the caller.
    """

    q_values: Tensor
    actions: Tensor
    targets: Tensor


# =============================================================================
class QValueObjective(nn.Module):
    """MSE regression on selected-action Q-values.

    Gathers Q-values at the given action indices and computes
    ``F.mse_loss(gathered, targets, reduction="none")``.
    """

    def forward(
        self,
        inputs: SelectedQInput,
    ) -> ObjectiveResult:
        """Compute selected Q-value MSE for one batch.

        Args:
            inputs: Prepared Q-value input.

        Returns:
            :class:`ObjectiveResult` with:
            - ``loss``: scalar, ``per_sample.sum()``.
            - ``losses["q_value"]``: same scalar.
            - ``terms["q_value_per_sample"]``: shape ``(B,)``, per-sample MSE.
        """
        gathered = inputs.q_values.gather(
            1, inputs.actions.unsqueeze(-1)
        ).squeeze(-1)
        per_sample = F.mse_loss(gathered, inputs.targets, reduction="none")
        loss = per_sample.sum()

        return ObjectiveResult(
            loss=loss,
            losses={"q_value": loss},
            terms={"q_value_per_sample": per_sample},
        )


# =============================================================================
__all__ = [
    "SelectedQInput",
    "QValueObjective",
]
