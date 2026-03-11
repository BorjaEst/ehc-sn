from __future__ import annotations

"""HPC attractor retrieval.

This module implements iterative attractor dynamics used for pattern completion
over a Hebbian memory matrix.

The retrieval operates on a flattened grounded-location code `h` of size
`S = sum(shape)` and iteratively updates subsets of dimensions using a
stage mask schedule. Masks are typically produced by
`ehc_sn.utils.update_to_masks` and represent which dimensions are updated at
each stage.
"""

from typing import List, Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.types import Activation, Device, Dtype


# =================================================================================================
class AttractorSettings(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """Settings for attractor dynamics modules."""

    kappa: float = Field(
        default=0.8,
        description="Hebbian retrieval decay term",
    )
    activation: Activation = Field(
        default="leaky_relu",
        frozen=True,
        description="Activation function for attractor dynamics.",
    )
    clamp_min: float = Field(
        default=-1.0,
        description="Minimum clamp value for attractor dynamics.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum clamp value for attractor dynamics.",
    )


# =================================================================================================
class AttractorNetwork(nn.Module):
    """Attractor retrieval dynamics (pattern completion) over a memory matrix.

    The network flattens the multi-frequency query code, applies iterative
    updates of the form:

        field = kappa * h + h @ M

    and uses stage masks to control which dimensions update at each stage.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, shape: List[int], config: Optional[AttractorSettings] = None,
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """Initialize the attractor.

        Args:
            shape: Feature sizes per frequency module.
            config: Attractor config. If `None`, defaults are used.
        """
        super().__init__()
        self._config = config or AttractorSettings()

        self._shape, self._n_freq = list(shape), len(shape)
        self._activation_fn = utils.activation_from_str(config.activation)

    @property
    def config(self) -> AttractorSettings:
        """Return attractor config."""
        return self._config

    def forward(  # -------------------------------------------------------------------------------
        self, p_query: List[Tensor], M: Tensor, *, masks: Optional[List[Tensor]] = None,
    ) -> List[Tensor]:  # fmt: skip
        """Run attractor retrieval.

        Args:
            p_query: Query grounded-location code (multi-scale), one tensor per
                frequency module with shape `(B, shape[f])`.
            M: Hebbian memory matrix of shape `(B, S, S)` where `S = sum(shape)`.
            masks: Optional stage masks used to gate which dimensions update at
                each iteration stage. Each mask is expected to be broadcastable
                to `h` (shape `(B, S)`).

        Returns:
            Retrieved grounded-location code (multi-scale), where each returned
            tensor has shape `(B, shape[f])`.
        """
        # Flatten query grounded locations across frequency modules.
        p, kappa = torch.cat(p_query, dim=1), self.config.kappa
        h = self.activation(p)

        # Ensure dtype consistency for numerical stability.
        masks = [m.to(dtype=h.dtype) for m in masks]
        M = M.to(dtype=h.dtype)

        for mask in masks:
            field = kappa * h + (h.unsqueeze(1) @ M).squeeze(1)
            h = (1 - mask) * h + mask * self.activation(field)

        # Re-split the grounded location into frequency modules.
        return torch.split(h, split_size_or_sections=self.shape, dim=1)

    def activation( # -----------------------------------------------------------------------------
        self, p: Tensor,
    ) -> Tensor:  # fmt: skip
        """Apply the configured activation with clamping."""
        p = torch.clamp(p, min=self.config.clamp_min, max=self.config.clamp_max)
        return self._activation_fn(p)


# =================================================================================================
__all__ = ["AttractorSettings", "AttractorNetwork"]
