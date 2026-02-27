""" """  # TODO: We need a better name for this module and probably split it into multiple files.

# TODO: Migrate to torchmetrics

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class AccuracyO:
    """Sensory prediction accuracy metrics with weighted averaging support.

    Attributes:
        acc_p_inf: Accuracy for inference pathway (float in [0.0, 1.0]).
        acc_gen_gi: Accuracy for retrieved pathway (float in [0.0, 1.0]).
        acc_gen_gg: Accuracy for ancestral pathway (float in [0.0, 1.0]).

    Note:
        Internal weight tracking (_total) is used for weighted averaging.
        Users should not access or modify this field directly.
    """

    acc_p_inf: Tensor  # inference pathway
    acc_gen_gi: Tensor  # retrieved pathway
    acc_gen_gg: Tensor  # ancestral pathway
    _total: Tensor | None = None  # weight for averaging (internal)

    @classmethod
    def zero(cls, *, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "AccuracyO":
        """Create a zero-initialized accuracy.

        Args:
            device: Device for tensor allocation.
            dtype: Data type for tensors.

        Returns:
            Zero-initialized :class:`AccuracyO`.
        """
        z = torch.zeros((), device=device, dtype=dtype)
        return cls(acc_p_inf=z.clone(), acc_gen_gi=z.clone(), acc_gen_gg=z.clone(), _total=z.clone())

    def __post_init__(self):
        """Set default _total to 1 if not provided."""
        if self._total is None:
            # Use device and dtype from first accuracy tensor
            self._total = torch.ones((), device=self.acc_p_inf.device, dtype=self.acc_p_inf.dtype)

    def __add__(self, other: "AccuracyO") -> "AccuracyO":
        """Add two accuracies with weighted averaging.

        Combines weighted accuracies: (acc1 * weight1 + acc2 * weight2) / (weight1 + weight2)

        Args:
            other: Another :class:`AccuracyO` to add.

        Returns:
            New :class:`AccuracyO` with weighted-averaged components.
        """
        total_new = self._total + other._total
        # Handle zero denominator case (both weights are 0)
        # Clamp to minimum 1.0 to avoid NaN
        denom = torch.clamp(total_new, min=1.0)
        # Weighted average: (a1*w1 + a2*w2) / (w1 + w2)
        return AccuracyO(
            acc_p_inf=(self.acc_p_inf * self._total + other.acc_p_inf * other._total) / denom,
            acc_gen_gi=(self.acc_gen_gi * self._total + other.acc_gen_gi * other._total) / denom,
            acc_gen_gg=(self.acc_gen_gg * self._total + other.acc_gen_gg * other._total) / denom,
            _total=total_new,
        )

    def __truediv__(self, divisor: int | float) -> "AccuracyO":
        """Divide internal weight by a scalar (accuracies unchanged).

        Args:
            divisor: Scalar divisor.

        Returns:
            New :class:`AccuracyO` with scaled internal weight.
        """
        return AccuracyO(self.acc_p_inf, self.acc_gen_gi, self.acc_gen_gg, _total=self._total / divisor)


class SensoryAccuracy(nn.Module):
    """Compute sensory prediction accuracies.

    This computes categorical prediction accuracy for each of the three TEM
    sensory prediction pathways, matching the structure of `SensoryReconstructionLoss`.
    """

    def __init__(self, reduction: Literal["none", "mean"] = "none"):
        """Initialize the accuracy metric.

        Args:
            reduction: Output reduction.
                - "none": return per-environment vectors of shape `[B]`.
                - "mean": return a scalar mean across environments.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, output: TEMOutput, label: WorldStep) -> AccuracyO:
        """Compute `AccuracyO` from logits and ground-truth observations.

        Args:
            y_p_inf: Predicted observations and logits from inference pathway.
            y_gen_gi: Predicted observations and logits from retrieved pathway.
            y_gen_gg: Predicted observations and logits from ancestral pathway.
            observation: Ground-truth observations (class indices or one-hot).

        Returns:
            AccuracyO: Per-pathway prediction accuracies (float in [0.0, 1.0]).

        Raises:
            ValueError: If `o_logits` does not contain exactly 3 tensors.
        """
        reconstruction, observation = output.reconstruction, label.observation
        y_p_inf, y_gen_gi, y_gen_gg = reconstruction.y_p_inf, reconstruction.y_gen_gi, reconstruction.y_gen_gg

        # Convert o to class indices if one-hot
        if observation.dim() == 2 and observation.shape[1] > 1:
            labels = torch.argmax(observation, dim=1)
        else:
            labels = observation.squeeze(-1) if observation.dim() == 2 else observation

        # Compute per-environment correctness (float 0.0 or 1.0)
        acc_acc_p_inf = (torch.argmax(y_p_inf.logits, dim=1) == labels).float()
        acc_acc_gen_gi = (torch.argmax(y_gen_gi.logits, dim=1) == labels).float()
        acc_acc_gen_gg = (torch.argmax(y_gen_gg.logits, dim=1) == labels).float()

        # Apply reduction if requested
        if self.reduction == "mean":
            acc_acc_p_inf = acc_acc_p_inf.mean()
            acc_acc_gen_gi = acc_acc_gen_gi.mean()
            acc_acc_gen_gg = acc_acc_gen_gg.mean()

        return AccuracyO(acc_acc_p_inf, acc_acc_gen_gi, acc_acc_gen_gg)
