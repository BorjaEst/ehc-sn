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
class PathSettings(BaseModel, extra="forbid"):
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
        self._flat_mat_dims = [in_dim * out_dim for in_dim, out_dim in self._mat_shape]

        # LocationBelief weights (action-conditioned)
        hidden_dim = [config.hidden_dim] * self._n_freq
        self.MLP_D_a = MLP([n_actions] * self._n_freq, self._flat_mat_dims, [torch.tanh, None], hidden_dim, bias=[True, False])  # fmt: skip
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
        self, action_ids: Tensor, g_prev: list[Tensor], no_direc_mask: Tensor | None = None,
    ) -> LocationBelief:  # fmt: skip
        """Compute the transition distribution for a single step.

        Args:
            action_ids: Discrete action ids of shape `(batch,)` or `(batch, 1)`.
            g_prev: Previous grid-code activations per frequency.
            no_direc_mask: Optional boolean mask of shape `(batch,)` indicating
                environments that should use the non-directional transition.

        Returns:
            A `LocationBelief` with mean and uncertainty per frequency.
        """
        mu = self.mean(action_ids, g_prev, no_direc_mask)
        sigma = self.uncertainty_mlp(g_prev)
        return LocationBelief(mean=mu, uncertainty=sigma)

    def mean(  # ----------------------------------------------------------------------------------
        self, action_ids: Tensor, g: list[Tensor], no_direc_mask: Tensor | None,
    ) -> list[Tensor]:  # fmt: skip
        """Compute the mean transition update.

        Args:
            action_ids: Discrete action ids of shape `(batch,)` or `(batch, 1)`.
            g: Current grid-code activations per frequency.
            no_direc_mask: Optional boolean mask selecting environments that
                should use the non-directional transition.

        Returns:
            Mean grid-code activations after applying the transition.
        """
        mats = self._transition_matrices(action_ids, no_direc_mask)

        # Build input by concatenating connected frequencies
        g_in = [
            torch.cat([g[f_from] for f_from in self._conn_indices[f_to]], dim=1).unsqueeze(1)
            for f_to in range(self.n_freq)
        ]

        # Apply transition via batch matrix multiply
        delta = [torch.bmm(g_in_f, mat_f).squeeze(1) for g_in_f, mat_f in zip(g_in, mats)]
        return [g_f + delta_f for g_f, delta_f in zip(g, delta)]

    def _transition_matrices(  # ------------------------------------------------------------------
        self, action_ids: Tensor, no_direc_mask: Tensor | None,
    ) -> list[Tensor]:  # fmt: skip
        """Build per-frequency transition matrices.

        Args:
            action_ids: Discrete action ids of shape `(batch,)` or `(batch, 1)`.
            no_direc_mask: Optional boolean mask selecting environments that
                should use `D_no_a`.

        Returns:
            A list of transition matrices, one per frequency module.
        """
        a = self._encode_action_ids(action_ids)
        d_flat = self.MLP_D_a([a] * self.n_freq)
        mats = [d.reshape(-1, *self._mat_shape[f]) for f, d in enumerate(d_flat)]

        if no_direc_mask is not None and torch.any(no_direc_mask):
            # Replace where the no-direction mask is active
            mask = no_direc_mask.view(-1, 1, 1)
            for f_to in range(self.n_freq):
                d_no_a = self.D_no_a[f_to].unsqueeze(0).expand_as(mats[f_to])
                mats[f_to] = torch.where(mask, d_no_a, mats[f_to])

        return mats

    def _encode_action_ids(self, action_ids: Tensor) -> Tensor:
        """Convert discrete action ids to one-hot features after validation."""
        encoded_ids = action_ids.squeeze(-1).to(torch.int64)
        invalid = (encoded_ids < 0) | (encoded_ids >= self._n_actions)
        if torch.any(invalid):
            bad_ids = encoded_ids[invalid].unique(sorted=True)
            raise ValueError(f"Action ids must be in [0, {self._n_actions}), got {bad_ids.tolist()}.")
        return torch.nn.functional.one_hot(encoded_ids, num_classes=self._n_actions).to(torch.float32)


# =================================================================================================
__all__ = ["PathSettings", "PathIntegrator"]
