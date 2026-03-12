"""LEC frequency filtering.

This module implements an exponential moving-average style filter per frequency
module, parameterized by a learned per-frequency coefficient.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.types import Device, Dtype


# =================================================================================================
class FreqFilterSettings(BaseModel, extra="forbid"):
    """Settings for LEC frequency filtering modules."""


# =================================================================================================
class FrequencyFilter(nn.Module):
    """Temporal frequency filter for sensory input.

    Each frequency module maintains a learned coefficient (stored as a logit)
    used to mix the previous filtered value with the current sensory input.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, f_initial: list[float], config: FreqFilterSettings,
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """Initialize the filter.

        Args:
            f_initial: Initial filter coefficients per frequency module.
            config: Configuration for the filter.
        """
        super().__init__()
        self._config = config
        self._n_freq = len(f_initial)

        # Initialize temporal filtering factors
        # Store as logit(f) so that sigmoid(alpha) recovers the desired frequency
        alpha_logit = [np.log(f / (1 - f)) for f in f_initial]
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(a, dtype=torch.float)) for a in alpha_logit])

    @property
    def config(self) -> FreqFilterSettings:
        """Return the filter config."""
        return self._config

    @property
    def n_freq(self) -> int:
        """Return the number of LEC frequency modules."""
        return self._n_freq

    def forward(  # -------------------------------------------------------------------------------
        self, c: Tensor, x_prev: list[Tensor],
    ) -> list[Tensor]:  # fmt: skip
        """Apply temporal filtering.

        Args:
            c: Current sensory input.
            x_prev: Previous filtered features per frequency.

        Returns:
            Updated filtered features per frequency.
        """
        alpha = [torch.sigmoid(self.alpha[f]) for f in range(self.n_freq)]
        return [(1 - alpha[f]) * x_prev[f] + alpha[f] * c for f in range(self.n_freq)]


# =================================================================================================
__all__ = ["FreqFilterSettings", "FrequencyFilter"]
