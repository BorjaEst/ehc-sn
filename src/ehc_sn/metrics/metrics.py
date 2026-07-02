"""TEM-specific sensory accuracy metrics.

These classes are specific to the TEM (Tolman-Eichenbaum Machine) architecture
and are **not** part of the generic metrics public API exported by
:mod:`ehp_sn.metrics`.  TEM code should import directly from this module:

    from ehc_sn.metrics.metrics import AccuracyO

Migration note: when TEM is refactored to use the standard paradigm-specific
metrics pipeline (``RatioMetric`` + routing table), replace ``AccuracyO`` with
route entries in a ``metrics/routes/tem.py`` module and delete this file.
"""

# TODO: When TEM gains a full loss head, replace AccuracyO with
#       RatioMetric route entries in metrics/routes/tem.py and delete this file.

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch import device as Device


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
    def zero(
        cls, *, device: Device | str, dtype: torch.dtype = torch.float32
    ) -> "AccuracyO":
        """Create a zero-initialized accuracy.

        Args:
            device: Device for tensor allocation.
            dtype: Data type for tensors.

        Returns:
            Zero-initialized :class:`AccuracyO`.
        """
        z = torch.zeros((), device=device, dtype=dtype)
        return cls(
            acc_p_inf=z.clone(),
            acc_gen_gi=z.clone(),
            acc_gen_gg=z.clone(),
            _total=z.clone(),
        )

    def __post_init__(self):
        """Set default _total to 1 if not provided."""
        if self._total is None:
            # Use device and dtype from first accuracy tensor
            self._total = torch.ones(
                (), device=self.acc_p_inf.device, dtype=self.acc_p_inf.dtype
            )

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
            acc_p_inf=(
                self.acc_p_inf * self._total + other.acc_p_inf * other._total
            )
            / denom,
            acc_gen_gi=(
                self.acc_gen_gi * self._total + other.acc_gen_gi * other._total
            )
            / denom,
            acc_gen_gg=(
                self.acc_gen_gg * self._total + other.acc_gen_gg * other._total
            )
            / denom,
            _total=total_new,
        )

    def __truediv__(self, divisor: int | float) -> "AccuracyO":
        """Divide internal weight by a scalar (accuracies unchanged).

        Args:
            divisor: Scalar divisor.

        Returns:
            New :class:`AccuracyO` with scaled internal weight.
        """
        return AccuracyO(
            self.acc_p_inf,
            self.acc_gen_gi,
            self.acc_gen_gg,
            _total=self._total / divisor,
        )
