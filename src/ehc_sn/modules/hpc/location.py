"""HPC grounded-location distribution.

This module infers a distribution over grounded location codes (place-cell-like)
from projected sensory features and projected abstract location.

Notes:
    The uncertainty head is conceptually reusable across modules (e.g. MEC).
    If refactoring towards a shared "LocationBelief uncertainty" component, this
    module is a likely consumer.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.mlp import MLP
from ehc_sn.types import Activation, Device, Dtype, LocationBelief


# =================================================================================================
class GroundLocSettings(BaseModel, extra="forbid"):
    """Settings for location distribution modules."""

    activation: Activation = Field(
        default="leaky_relu",
        frozen=True,
        description="Activation function for attractor dynamics.",
    )
    clamp_min: float = Field(
        default=-1.0,
        description="Minimum clamp value for Hebbian memory weights.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum clamp value for Hebbian memory weights.",
    )


# =================================================================================================
class GroundLocation(nn.Module):
    """Infer grounded-location beliefs from sensory and abstract codes."""

    def __init__(  # ------------------------------------------------------------------------------
        self, shape: list[int], config: GroundLocSettings,
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """Initialize grounded-location inference.

        Args:
            shape: Grounded-location feature sizes per frequency module.
            config: Grounded-location inference config.
        """
        super().__init__()
        self._config = config

        self._shape, self._n_freq = list(shape), len(shape)
        self._activation_fn = utils.activation_from_str(self._config.activation)

        # Uncertainty from predicted grounded location
        self.uncertainty_mlp = MLP(shape, shape, [torch.tanh, torch.exp], [2 * n for n in shape])

    @property
    def config(self) -> GroundLocSettings:
        """Return grounded-location inference config."""
        return self._config

    @property
    def shape(self) -> list[int]:
        """Return the grounded-location shape."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Return the number of grounded-location frequency modules."""
        return self._n_freq

    def forward(  # -------------------------------------------------------------------------------
        self, x_: list[Tensor], g_: list[Tensor],
    ) -> LocationBelief:  # fmt: skip
        """Infer grounded-location mean and uncertainty.

        Args:
            x_: Projected sensory features per frequency module.
            g_: Projected abstract location per frequency module.

        Returns:
            A `LocationBelief` with:

            - `mean`: inferred grounded-location mean per frequency
            - `uncertainty`: inferred grounded-location uncertainty per frequency
        """
        mu_p = [self.activation(g_[f] * x_[f]) for f in range(self.n_freq)]
        sigma_p = self.uncertainty_mlp(mu_p)
        return LocationBelief(mean=mu_p, uncertainty=sigma_p)

    def activation(  # ----------------------------------------------------------------------------
        self, p: Tensor,
    ) -> Tensor:  # fmt: skip
        """Apply the configured activation with clamping."""
        p = torch.clamp(p, min=self.config.clamp_min, max=self.config.clamp_max)
        return self._activation_fn(p)


# =================================================================================================
__all__ = ["GroundLocSettings", "GroundLocation"]
