"""MEC path integration.

This module implements action-conditioned transitions for grid-cell activations
and estimates transition uncertainty.
"""

from __future__ import annotations

from typing import Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.mlp import MLP
from ehc_sn.types import LocationBelief


# =================================================================================================
class PathSettings(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """Settings for path integration modules."""

    hidden_dim: int = Field(
        default=20,
        frozen=True,
        description="Hidden dimension for transition MLP.",
    )


# =================================================================================================
class PathIntegrator(nn.Module):
    """Action-conditioned grid-code transition model.

    The model predicts per-frequency transition matrices conditioned on the
    agent action. Optionally, a subset of environments can use a non-directional
    transition (`D_no_a`) via `no_direc_mask`.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, n_actions: int, mec_shape: list[int], f_initial: list[float], config: PathSettings,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self._config = config
        self._n_actions = n_actions
        self._shape = list(mec_shape)
        self._n_freq = len(mec_shape)

        self._connections = conn = utils.connections(f_initial)
        self._conn_indices = [[f_from for f_from in range(self._n_freq) if conn[f_to][f_from]] for f_to in range(self._n_freq)]  # fmt: skip
        self._in_dims = [sum(mec_shape[f_from] for f_from in self._conn_indices[f_to]) for f_to in range(self._n_freq)]  # fmt: skip
        self._mat_shape = [(self._in_dims[f_to], mec_shape[f_to]) for f_to in range(self._n_freq)]

        # LocationBelief weights (action-conditioned)
        hidden_dim = [config.hidden_dim] * self._n_freq
        self.MLP_D_a = MLP([n_actions] * self._n_freq, mec_shape, [torch.tanh, None], hidden_dim, bias=[True, False])  # fmt: skip
        self.MLP_D_a.set_weights(1, 0.0)
        self.D_no_a = nn.ParameterList([nn.Parameter(torch.zeros(m)) for m in self._mat_shape])  # fmt: skip

        # LocationBelief uncertainty
        self.uncertainty_mlp = MLP(mec_shape, mec_shape, [torch.tanh, torch.exp], [2 * g for g in mec_shape])

    @property
    def config(self) -> PathSettings:
        """Return the path integration config."""
        return self._config

    @property
    def shape(self) -> list[int]:
        """Return the MEC feature shape consumed by the transition."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Return the number of MEC frequency modules."""
        return self._n_freq

    def forward(  # -------------------------------------------------------------------------------
        self, a: Tensor, g_prev: list[Tensor], no_direc_mask: Tensor | None = None,
    ) -> LocationBelief:  # fmt: skip
        """Compute the transition distribution for a single step.

        Args:
            a: One-hot action tensor of shape `(batch, n_a)`.
            g_prev: Previous grid-code activations per frequency.
            no_direc_mask: Optional boolean mask of shape `(batch,)` indicating
                environments that should use the non-directional transition.

        Returns:
            A `LocationBelief` with mean and uncertainty per frequency.
        """
        mu = self.mean(a, g_prev, no_direc_mask)
        sigma = self.uncertainty_mlp(g_prev)
        return LocationBelief(mean=mu, uncertainty=sigma)

    def mean(  # ----------------------------------------------------------------------------------
        self, a: Tensor, g: list[Tensor], no_direc_mask: Tensor | None,
    ) -> list[Tensor]:  # fmt: skip
        """Compute the mean transition update.

        Args:
            a: One-hot action tensor of shape `(batch, n_a)`.
            g: Current grid-code activations per frequency.
            no_direc_mask: Optional boolean mask selecting environments that
                should use the non-directional transition.

        Returns:
            Mean grid-code activations after applying the transition.
        """
        mats = self._transition_matrices(a, no_direc_mask)

        # Build input by concatenating connected frequencies
        g_in = [
            torch.cat([g[f_from] for f_from in self._conn_indices[f_to]], dim=1).unsqueeze(1)
            for f_to in range(self.n_freq)
        ]

        # Apply transition via batch matrix multiply
        delta = [torch.bmm(g_in_f, mat_f).squeeze(1) for g_in_f, mat_f in zip(g_in, mats)]
        return [g_f + delta_f for g_f, delta_f in zip(g, delta)]

    def _transition_matrices(  # ------------------------------------------------------------------
        self, a: Tensor, no_direc_mask: Tensor | None,
    ) -> list[Tensor]:  # fmt: skip
        """Build per-frequency transition matrices.

        Args:
            a: One-hot action tensor of shape `(batch, n_a)`.
            no_direc_mask: Optional boolean mask selecting environments that
                should use `D_no_a`.

        Returns:
            A list of transition matrices, one per frequency module.
        """
        d_flat = self.MLP_D_a([a] * self.n_freq)
        mats = [d.reshape(-1, *self._mat_shape[f]) for f, d in enumerate(d_flat)]

        if no_direc_mask is not None and torch.any(no_direc_mask):
            # Replace where the no-direction mask is active
            mask = no_direc_mask.view(-1, 1, 1)
            for f_to in range(self.n_freq):
                d_no_a = self.D_no_a[f_to].unsqueeze(0).expand_as(mats[f_to])
                mats[f_to] = torch.where(mask, d_no_a, mats[f_to])

        return mats


# =================================================================================================
__all__ = ["PathSettings", "PathIntegrator"]
