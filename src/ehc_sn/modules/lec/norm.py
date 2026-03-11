"""LEC feature normalization.

This module performs a simple per-frequency normalization:
ReLU center-shift followed by vector normalization.
"""

from __future__ import annotations

from typing import Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.types import Device, Dtype


# =================================================================================================
class FeatureNormSettings(BaseModel, extra="forbid"):
    """Settings for LEC normalization modules."""


# =================================================================================================
class FeatureNorm(nn.Module):
    """Normalize per-frequency feature vectors."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: Optional[FeatureNormSettings] = None,
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """Initialize the normalization module.

        Args:
            config: Configuration for normalization.
        """
        super().__init__()
        self._config = config or FeatureNormSettings()

    @property
    def config(self) -> FeatureNormSettings:
        """Return the normalization config."""
        return self._config

    def forward(  # -------------------------------------------------------------------------------
        self, x: list[Tensor],
    ) -> list[Tensor]:  # fmt: skip
        """Normalize features.

        Args:
            x: Per-frequency feature tensors.

        Returns:
            Normalized per-frequency feature tensors.
        """
        n_freq = len(x)
        positive_centered = [utils.relu(x[f] - torch.mean(x[f])) for f in range(n_freq)]
        normalized = [utils.normalize(positive_centered[f]) for f in range(n_freq)]
        return normalized


# =================================================================================================
__all__ = ["FeatureNorm", "FeatureNormSettings"]
