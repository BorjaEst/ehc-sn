"""MEC OVC correction.

This module uses shiny landmark cues to compute corrections for a subset of
frequency modules (OVC modules) and fuses them with a reference transition.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules import MLP
from ehc_sn.types import LocationBelief


# =================================================================================================
class OVCSettings(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """Settings for OVC modules."""

    hidden_dim: int = Field(
        default=20,
        frozen=True,
        description="Hidden dimension for shiny landmark cue processing (OVC correction).",
    )


# =================================================================================================
class OVCCorrection(nn.Module):
    """Fuse shiny landmark cues into selected frequency modules.

    The correction is applied only to the configured OVC frequency slice and
    combined using inverse-variance weighting.
    """

    def __init__(
        self, n_ovc: Optional[List[int]], mec_shape: List[int], *, settings: Optional[OVCSettings] = None
    ):
        super().__init__()
        self._settings = settings or OVCSettings()
        n_freq_ovc = None if n_ovc is None else len(n_ovc)
        self._ovc_start, self._ovc_count = utils.resolve_ovc_slice(len(mec_shape), n_freq_ovc)
        self._shape = mec_shape[self._ovc_start : self._ovc_start + self._ovc_count]
        self._n_freq = len(self._shape)

        # Shiny cue → mean and uncertainty.
        hidden_dim = [settings.hidden_dim] * self.n_freq
        self.g_shiny_mlp = MLP([1] * self.n_freq, self.shape, hidden_dim=hidden_dim)
        self.uncertainty_mlp = MLP(
            [1] * self.n_freq, self.shape, [torch.tanh, torch.exp], hidden_dim=hidden_dim
        )

    @property
    def settings(self) -> OVCSettings:
        """Return the OVC settings."""
        return self._settings

    @property
    def shape(self) -> List[int]:
        """Return OVC module sizes per frequency."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Return the number of OVC frequency modules."""
        return self._n_freq

    @property
    def ovc_start(self) -> int:
        """Return the starting index (into MEC frequencies) for OVC correction."""
        return self._ovc_start

    def forward(self, locations: list[dict], transition: LocationBelief) -> LocationBelief:
        """Apply OVC correction to environments with shiny cues.

        Args:
            locations: Per-environment metadata. If `loc.get("shiny")` is not
                `None`, the value is treated as a scalar cue for that batch
                element.
            transition: Reference transition to correct.

        Returns:
            A corrected `LocationBelief`. If no shiny cues are present, returns the
            input transition unchanged.
        """
        shiny_mask = self._identify_shiny_envs(locations, transition.mean[0].device)
        if shiny_mask is None:  # No shiny envs present
            return transition

        shiny_input = self._extract_shiny_cues(locations, shiny_mask, transition.mean[0].device)
        freqs = range(self.ovc_start, self.ovc_start + self.n_freq)

        correction = self._predict_correction(shiny_input)
        return utils.inv_var_trans(transition, correction, shiny_mask, freqs)

    def _identify_shiny_envs(self, locations: list[dict], device: torch.device) -> Tensor | None:
        """Return a mask selecting environments with shiny cues.

        Args:
            locations: Per-environment metadata.
            device: Device for the returned tensor.

        Returns:
            A boolean mask of shape `(batch,)`, or `None` if no shiny cues are
            present.
        """
        shiny_envs = [loc.get("shiny") is not None for loc in locations]
        if not any(shiny_envs):
            return None
        return torch.tensor(shiny_envs, dtype=torch.bool, device=device)

    def _extract_shiny_cues(
        self, locations: list[dict], shiny_mask: Tensor, device: torch.device
    ) -> List[Tensor]:
        """Extract shiny cue values as inputs for the OVC MLPs.

        Args:
            locations: Per-environment metadata.
            shiny_mask: Boolean mask indicating which batch items have cues.
            device: Device for returned tensors.

        Returns:
            A list of cue tensors (one per OVC module). Each tensor has shape
            `(n_shiny, 1)`.
        """
        shiny_vals = [loc["shiny"] for loc in locations if loc.get("shiny") is not None]
        shiny_tensor = torch.as_tensor(shiny_vals, dtype=torch.float32, device=device).unsqueeze(-1)
        return [shiny_tensor] * self.n_freq

    def _predict_correction(self, shiny_input: List[Tensor]) -> LocationBelief:
        """Predict mean and uncertainty for the OVC correction.

        Args:
            shiny_input: List of cue tensors (one per OVC module).

        Returns:
            A `LocationBelief` containing OVC correction mean and uncertainty.
        """
        # Predict mean with legacy nonlinearity (abs → leaky_relu)
        mu_g = [torch.abs(mu) for mu in self.g_shiny_mlp(shiny_input)]
        mu_g_shiny = [utils.leaky_relu(torch.clamp(g_f, min=-1.0, max=1.0)) for g_f in mu_g]

        # Predict uncertainty
        sigma_g_shiny = self.uncertainty_mlp(shiny_input)

        return LocationBelief(mean=mu_g_shiny, uncertainty=sigma_g_shiny)
