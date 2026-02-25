"""LEC reconstruction.

This module provides a minimal reconstruction head used in the generative
branch, mapping LEC features back to the sensory input space.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn


# =================================================================================================
class ReconstructionSettings(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """Settings for LEC reconstruction modules."""


# =================================================================================================
class Reconstruction(nn.Module):
    """Linear reconstruction of sensory input from LEC features."""

    def __init__(self, n_c: int, settings: ReconstructionSettings):
        """Initialize reconstruction parameters.

        Args:
            n_c: Number of sensory channels / feature dimensions.
            settings: Reconstruction configuration.
        """
        super().__init__()
        self._settings = settings

        # Reconstruction parameters
        self.w_x = torch.nn.Parameter(torch.tensor(1.0))  # For reconstructing c from x
        self.b_x = torch.nn.Parameter(torch.zeros(n_c))  # Bias for reconstructing c from x

    @property
    def settings(self) -> ReconstructionSettings:
        """Return the reconstruction settings."""
        return self._settings

    def forward(self, x: List[Tensor]) -> Tensor:
        """Reconstruct sensory input.

        Args:
            x: Per-frequency LEC features.

        Returns:
            Reconstructed sensory input tensor.
        """
        return self.w_x * x[0] + self.b_x
