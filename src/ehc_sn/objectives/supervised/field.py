"""Atomic continuous-field MSE objective.

:class:`FieldRegressionObjective` is a reusable ``nn.Module`` that computes
masked MSE over continuous-valued predictions.  It consumes a
:class:`FieldObjectiveInput` and returns an :class:`ObjectiveResult` with
per-element and per-sample terms.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from ehc_sn.objectives.types import ObjectiveResult


# =============================================================================
@dataclass(frozen=True)
class FieldObjectiveInput:
    """Prepared field-prediction input for :class:`FieldRegressionObjective`.

    Attributes:
        prediction: Predicted field, shape ``(B, N)``, values in ``[0, 1]``.
        target: Target field, shape ``(B, N)``, values in ``[0, 1]``.
        mask: Valid-node mask, shape ``(B, N)`` bool.
    """

    prediction: Tensor
    target: Tensor
    mask: Tensor


# =============================================================================
class FieldRegressionObjective(nn.Module):
    """Masked MSE regression for continuous-valued field predictions.

    The loss is computed as ``(prediction - target) ** 2``, masked by *mask*,
    then averaged over valid nodes per-sample.
    """

    def forward(
        self,
        inputs: FieldObjectiveInput,
    ) -> ObjectiveResult:
        """Compute masked field MSE for one batch.

        Args:
            inputs: Prepared field objective input.

        Returns:
            :class:`ObjectiveResult` with:
            - ``loss``: scalar, ``per_sample.sum()``.
            - ``losses["field_mse"]``: same scalar.
            - ``terms["field_per_element"]``: shape ``(B, N)``, masked squared
              error (unmasked positions are zero).
            - ``terms["field_per_sample"]``: shape ``(B,)``, per-sample mean
              over valid nodes.
        """
        diff = inputs.prediction - inputs.target
        per_element = diff**2 * inputs.mask
        valid_counts = inputs.mask.sum(dim=1).clamp(min=1)
        per_sample = per_element.sum(dim=1) / valid_counts
        loss = per_sample.sum()

        return ObjectiveResult(
            loss=loss,
            losses={"field_mse": loss},
            terms={
                "field_per_element": per_element,
                "field_per_sample": per_sample,
            },
        )


# =============================================================================
__all__ = [
    "FieldObjectiveInput",
    "FieldRegressionObjective",
]
