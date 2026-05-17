"""Grounded-location inference for hippocampal memory.

This module combines projected sensory features and projected abstract-location
features into a grounded-location belief over place-like codes. The resulting
``LocationBelief`` is the HPC-side distribution carried through the TEM memory
cycle.

Notes:
    The uncertainty head is conceptually reusable across modules. If the repo
    later factors out shared ``LocationBelief`` uncertainty prediction, this
    module is a primary candidate consumer.
"""

from __future__ import annotations

from typing import Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn import utils
from ehc_sn.activations.softplus import bounded_positive_scale
from ehc_sn.modules.mlp import MLP
from ehc_sn.types import Activation, LocationBelief


# =============================================================================
class PlaceInferenceSettings(BaseModel, extra="forbid"):
    """Static configuration for grounded-location inference.

    The settings define the nonlinearity applied to the multiplicative sensory
    and structural interaction, together with the activation clamp used for
    numerical stability before uncertainty prediction.
    """

    activation: Activation = Field(
        default="leaky_relu",
        frozen=True,
        description="Activation function for grounded-location inference.",
    )
    clamp_min: float = Field(
        default=-1.0,
        description="Minimum clamp value for grounded-location activations.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum clamp value for grounded-location activations.",
    )


# =============================================================================
class PlaceInference(nn.Module):
    """Infer grounded-location beliefs from sensory and abstract-location cues.

    The module implements the HPC-side fusion step that maps projected sensory
    features ``x_`` and projected structural features ``g_`` into a
    ``LocationBelief`` over grounded, place-like codes.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        shape: list[int],
        config: PlaceInferenceSettings,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        """Initialize grounded-location inference.

        Args:
            shape: Grounded-location feature widths per frequency module.
            config: Static inference configuration.
            device: Optional allocation device for parameter tensors.
            dtype: Optional floating-point dtype for parameter tensors.
        """
        super().__init__()
        self._config = config

        self._shape, self._n_freq = list(shape), len(shape)
        self._activation_fn = utils.activation_from_str(self._config.activation)

        # Predict uncertainty directly from the inferred grounded-location mean.
        self.uncertainty_mlp = MLP(
            shape,
            shape,
            [torch.tanh, bounded_positive_scale],
            [2 * n for n in shape],
        )

    @property
    def config(self) -> PlaceInferenceSettings:
        """Return place-inference config."""
        return self._config

    @property
    def shape(self) -> list[int]:
        """Return the grounded-location shape."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Return the number of grounded-location frequency modules."""
        return self._n_freq

    def forward(  # -----------------------------------------------------------
        self,
        x_: list[Tensor],
        g_: list[Tensor],
    ) -> LocationBelief:
        """Infer grounded-location mean and uncertainty.

        Args:
            x_: Projected sensory features per frequency module. Each tensor has
                shape ``(B, N_f)``.
            g_: Projected abstract-location features per frequency module. Each
                tensor has shape ``(B, N_f)``.

        Returns:
            A ``LocationBelief`` whose ``mean`` and ``uncertainty`` are lists of
            per-frequency tensors with shape ``(B, N_f)``.
        """
        mu_p = [self.activation(g_[f] * x_[f]) for f in range(self.n_freq)]
        sigma_p = self.uncertainty_mlp(mu_p)
        return LocationBelief(mean=mu_p, uncertainty=sigma_p)

    def activation(  # --------------------------------------------------------
        self,
        p: Tensor,
    ) -> Tensor:
        """Clamp and activate one grounded-location tensor.

        Args:
            p: One per-frequency grounded-location pre-activation tensor with
                shape ``(B, N_f)``.

        Returns:
            The clamped and activated tensor with the same shape as ``p``.
        """
        p = torch.clamp(p, min=self.config.clamp_min, max=self.config.clamp_max)
        return self._activation_fn(p)


# =============================================================================
__all__ = ["PlaceInferenceSettings", "PlaceInference"]
