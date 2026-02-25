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


# =================================================================================================
class FreqFilterSettings(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """Settings for LEC frequency filtering modules."""


# =================================================================================================
class FrequencyFilter(nn.Module):
    """Temporal frequency filter for sensory input.

    Each frequency module maintains a learned coefficient (stored as a logit)
    used to mix the previous filtered value with the current sensory input.
    """

    def __init__(self, f_init: List[float], settings: FreqFilterSettings):
        """Initialize the filter.

        Args:
            f_init: Initial filter coefficients per frequency module.
            settings: Configuration for the filter.
        """
        super().__init__()
        self._n_freq = len(f_init)
        self._settings = settings

        # Initialize temporal filtering factors
        # Store as logit(f) so that sigmoid(alpha) recovers the desired frequency
        alpha_logit = [np.log(f / (1 - f)) for f in f_init]
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(a, dtype=torch.float)) for a in alpha_logit])

    @property
    def settings(self) -> FreqFilterSettings:
        """Return the filter settings."""
        return self._settings

    @property
    def n_freq(self) -> int:
        """Return the number of frequency modules."""
        return len(self.alpha)

    def forward(self, c: Tensor, x_prev: List[Tensor]) -> List[Tensor]:
        """Apply temporal filtering.

        Args:
            c: Current sensory input.
            x_prev: Previous filtered features per frequency.

        Returns:
            Updated filtered features per frequency.
        """
        alpha = [torch.sigmoid(self.alpha[f]) for f in range(self.n_freq)]
        return [(1 - alpha[f]) * x_prev[f] + alpha[f] * c for f in range(self.n_freq)]
