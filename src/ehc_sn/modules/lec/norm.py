"""LEC feature normalization.

This module performs a simple per-frequency normalization:
ReLU center-shift followed by vector normalization.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn import utils


# =================================================================================================
class FeatureNormSettings(BaseModel, extra="forbid"):
    """Settings for LEC normalization modules."""


# =================================================================================================
class FeatureNorm(nn.Module):
    """Normalize per-frequency feature vectors."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: FeatureNormSettings,
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """Initialize the normalization module.

        Args:
            config: Configuration for normalization.
        """
        super().__init__()
        self._config = config

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
        # Subtract per-row mean (keepdim so shape stays (B, D)), then ReLU, then L2-normalize.
        # Reference: tem_tf2/tem_model.py:684 — tf.reduce_mean(x[f], axis=1, keepdims=True).
        return [
            utils.normalize(utils.relu(x[f] - x[f].mean(dim=-1, keepdim=True)))
            for f in range(len(x))
        ]


# =================================================================================================
__all__ = ["FeatureNorm", "FeatureNormSettings"]
