"""TorchMetrics helpers for HRM metrics logging."""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric

__all__ = ["RatioMetric"]


# =================================================================================================
class RatioMetric(Metric):
    """Compute a ratio from aggregated numerator/denominator sums."""

    full_state_update = False

    def __init__(  # ------------------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        super().__init__(sync_on_compute=True)
        self.add_state("numerator_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(  # --------------------------------------------------------------------------------
        self, numerator_sum: Tensor, denominator_sum: Tensor,
    ) -> None:  # fmt: skip
        """Accumulate numerator and denominator totals."""
        self.numerator_sum += numerator_sum.float()
        self.denominator_sum += denominator_sum.float()

    def compute(  # -------------------------------------------------------------------------------
        self,
    ) -> Tensor:  # fmt: skip
        """Return the halted-only accuracy ratio."""
        return self.numerator_sum / self.denominator_sum.clamp_min(1.0)  # type: ignore


# =================================================================================================
class LastRatioMetric(Metric):
    """Compute a ratio from the last numerator/denominator sums (no accumulation)."""

    full_state_update = False

    def __init__(  # ------------------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        super().__init__(sync_on_compute=True)
        self.add_state("numerator_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(  # --------------------------------------------------------------------------------
        self, numerator_sum: Tensor, denominator_sum: Tensor,
    ) -> None:  # fmt: skip
        """Set numerator and denominator totals (no accumulation, overwrite previous)."""
        # overwrite (direct metric), don’t accumulate
        self.numerator_sum = numerator_sum.float()
        self.denominator_sum = denominator_sum.float()

    def compute(  # -------------------------------------------------------------------------------
        self,
    ) -> Tensor:  # fmt: skip
        """Return the halted-only accuracy ratio."""
        return self.numerator_sum / self.denominator_sum.clamp_min(1.0)
