"""LEC feature normalization.

This module performs a simple per-frequency normalization:
ReLU center-shift followed by vector normalization.
"""

from __future__ import annotations

from typing import List

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils


# =================================================================================================
class FeatureNormSettings(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """Settings for LEC normalization modules."""


# =================================================================================================
class FeatureNorm(nn.Module):
    """Normalize per-frequency feature vectors."""

    def __init__(self, settings: FeatureNormSettings):
        """Initialize the normalization module.

        Args:
            settings: Configuration for normalization.
        """
        super().__init__()
        self._settings = settings

    @property
    def settings(self) -> FeatureNormSettings:
        """Return the normalization settings."""
        return self._settings

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """Normalize features.

        Args:
            x: Per-frequency feature tensors.

        Returns:
            Normalized per-frequency feature tensors.
        """
        n_freq = len(x)
        positive_centered = [utils.relu(x[f] - torch.mean(x[f])) for f in range(n_freq)]
        normalised = [utils.normalise(positive_centered[f]) for f in range(n_freq)]
        return normalised
