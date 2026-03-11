"""MEC memory inference (p→g).

This module predicts grid-cell activations from hippocampal place-cell patterns
and fuses that prediction with a reference transition using inverse-variance
weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from pydantic import BaseModel, Field
from scipy.stats import truncnorm
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.mlp import MLP
from ehc_sn.types import LocationBelief


# =================================================================================================
class P2GMemSettings(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """Settings for MEC memory inference modules."""

    sigma_init: float = Field(
        default=0.1,
        frozen=True,
        description="Standard deviation to initialise hidden to output layer of MLP for inferring new abstract location",
    )


# =================================================================================================
@dataclass
class Runtime:
    """ """

    uncertainty_offset: float = 0.0


# =================================================================================================
class P2GMemory(nn.Module):
    """Infer grid-cell code from retrieved place-cell activity.

    The model predicts a grid-code mean from `p_x` and estimates uncertainty
    using memory-quality features. The result is fused with a reference
    `LocationBelief` (typically from path integration).
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, n_p: list[int], mec_shape: list[int], config: P2GMemSettings,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self._config = config
        self._runtime = Runtime()

        # Mean prediction from place cells
        self.MLP_mu_g_mem = MLP(n_p, mec_shape, hidden_dim=[2 * g for g in mec_shape])

        # Initialize last layer with truncated normal (legacy parity)
        init_w = lambda f: truncnorm.rvs(-2, 2, size=list(self.MLP_mu_g_mem.w[f][-1].weight.shape), loc=0, scale=config.sigma_init)  # fmt: skip
        self.MLP_mu_g_mem.set_weights(-1, [torch.tensor(init_w(f), dtype=torch.float32) for f in range(self._n_freq)])  # fmt: skip

        # Uncertainty from memory quality indicators
        self.MLP_sigma_g_mem = MLP([2 for _ in n_p], mec_shape, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in mec_shape])  # fmt: skip

    @property
    def config(self) -> P2GMemSettings:
        """Return the P2G memory config."""
        return self._config

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    def forward(  # -------------------------------------------------------------------------------
        self, p_x: list[Tensor], transition: LocationBelief,
    ) -> LocationBelief:  # fmt: skip
        """Infer a corrected grid-code transition from place cells.

        Args:
            p_x: Retrieved place-cell activations per frequency.
            transition: Reference transition to correct (e.g., path integration).

        Returns:
            A fused `LocationBelief` after memory-based correction.
        """
        g_ref, sigma_ref = transition.mean, transition.uncertainty  # Unpack for clarity

        mu = self._inference_mean(p_x)
        sigma = self._inference_uncertainty(g_ref, err=utils.squared_error(mu, g_ref))

        correction = LocationBelief(mean=mu, uncertainty=sigma)
        return utils.inv_var_trans(transition, correction)

    def _inference_mean(  # -----------------------------------------------------------------------
        self, p_x: list[Tensor],
    ) -> list[Tensor]:  # fmt: skip
        """Predict grid-code means from place cells.

        Args:
            p_x: Place-cell activations per frequency.

        Returns:
            Predicted grid-code means per frequency.
        """
        return self.MLP_mu_g_mem(p_x)

    def _inference_uncertainty(  # ----------------------------------------------------------------
        self, g: list[Tensor], err: list[Tensor],
    ) -> list[Tensor]:  # fmt: skip
        """Estimate uncertainty from grid-code magnitude and reconstruction error.

        Args:
            g: Reference grid-code activations per frequency.
            err: Per-frequency reconstruction error features.

        Returns:
            Estimated uncertainty per frequency.
        """
        sigma_g_input = [
            torch.cat((torch.sum(mu_f**2, dim=1, keepdim=True), torch.unsqueeze(err[f], dim=1)), dim=1)
            for f, mu_f in enumerate(g)
        ]
        sigma = self.MLP_sigma_g_mem(sigma_g_input)
        return [sigma[f] + self.runtime.uncertainty_offset for f in range(self._n_freq)]


# =================================================================================================
__all__ = ["P2GMemory", "P2GMemSettings"]
