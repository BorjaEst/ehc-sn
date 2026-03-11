"""LEC reconstruction.

This module provides a minimal reconstruction head used in the generative
branch, mapping LEC features back to the sensory input space.
"""

from __future__ import annotations

from typing import Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.types import Device, Dtype


# =================================================================================================
class ReconstructionSettings(BaseModel, extra="forbid"):
    """Settings for LEC reconstruction modules."""


# =================================================================================================
class Reconstruction(nn.Module):
    """Linear reconstruction of sensory input from LEC features."""

    def __init__(
        self, n_c: int, config: Optional[ReconstructionSettings],
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """Initialize reconstruction parameters.

        Args:
            n_c: Number of sensory channels / feature dimensions.
            config: Reconstruction configuration.
        """
        super().__init__()
        self._config = config or ReconstructionSettings()

        # Reconstruction parameters
        self.w_x = torch.nn.Parameter(torch.tensor(1.0))  # For reconstructing c from x
        self.b_x = torch.nn.Parameter(torch.zeros(n_c))  # Bias for reconstructing c from x

    @property
    def config(self) -> ReconstructionSettings:
        """Return the reconstruction config."""
        return self._config

    def forward(self, x: list[Tensor]) -> Tensor:
        """Reconstruct sensory input.

        Args:
            x: Per-frequency LEC features.

        Returns:
            Reconstructed sensory input tensor.
        """
        return self.w_x * x[0] + self.b_x


# =================================================================================================
__all__ = ["Reconstruction", "ReconstructionSettings"]
