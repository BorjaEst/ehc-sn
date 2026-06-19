"""Atomic state-value MSE objective.

:class:`StateValueObjective` is a reusable ``nn.Module`` that computes MSE on
predicted state values.  It consumes a :class:`ValueRegressionInput` and
returns an :class:`ObjectiveResult` with per-sample terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor, nn

from ehc_sn.objectives.types import ObjectiveResult


# =============================================================================
@dataclass(frozen=True)
class ValueRegressionInput:
    """Prepared value-regression input for :class:`StateValueObjective`.

    Attributes:
        predictions: Predicted state values, shape ``(B,)``.
        targets: Regression targets, shape ``(B,)``.
    """

    predictions: Tensor
    targets: Tensor


# =============================================================================
class StateValueObjective(nn.Module):
    """MSE regression on state values.

    Computes ``F.mse_loss(predictions, targets, reduction="none")``.
    """

    def forward(
        self,
        inputs: ValueRegressionInput,
    ) -> ObjectiveResult:
        """Compute state-value MSE for one batch.

        Args:
            inputs: Prepared value regression input.

        Returns:
            :class:`ObjectiveResult` with:
            - ``loss``: scalar, ``per_sample.sum()``.
            - ``losses["state_value"]``: same scalar.
            - ``terms["state_value_per_sample"]``: shape ``(B,)``, per-sample MSE.
        """
        per_sample = F.mse_loss(
            inputs.predictions, inputs.targets, reduction="none"
        )
        loss = per_sample.sum()

        return ObjectiveResult(
            loss=loss,
            losses={"state_value": loss},
            terms={"state_value_per_sample": per_sample},
        )


# =============================================================================
__all__ = [
    "ValueRegressionInput",
    "StateValueObjective",
]
