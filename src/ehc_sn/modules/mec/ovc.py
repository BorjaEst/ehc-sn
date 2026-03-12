"""MEC OVC correction.

This module uses shiny landmark cues to compute corrections for a subset of
frequency modules (OVC modules) and fuses them with a reference transition.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from pydantic import BaseModel, Field, model_validator
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.mlp import MLP
from ehc_sn.types import Device, Dtype, LocationBelief
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class OVCSettings(BaseModel, extra="forbid"):
    """Settings for OVC modules."""

    mode: Literal["off", "merged", "separate"] = Field(
        default="merged",
        description=(
            "OVC mode. If 'off', no OVC correction is applied. "
            "If 'merged', all OVC frequencies are merged into a single module "
            "(with size equal to the sum of the specified sizes). "
            "If 'separate', OVC frequencies are kept separate with sizes specified by `shape`."
        ),
    )
    shape: Optional[list[int]] = Field(
        default=None,
        description="Sizes of OVC frequency modules (required if `mode='separate'`).",
    )
    hidden_dim: int = Field(
        default=20,
        ge=1,
        frozen=True,
        description="Hidden dimension for shiny landmark cue processing (OVC correction).",
    )

    @model_validator(mode="after")
    def validate_shape_policy(self) -> "OVCSettings":
        if self.mode == "separate" and not self.shape:
            raise ValueError("mec.ovc.shape is required when mec.ovc.mode='separate'.")
        if self.mode != "separate" and self.shape is not None:
            raise ValueError("mec.ovc.shape is only allowed when mec.ovc.mode='separate'.")
        return self


# =================================================================================================
class OVCCorrection(nn.Module):
    """Fuse shiny landmark cues into selected frequency modules.

    The correction is applied only to the configured OVC frequency slice and
    combined using inverse-variance weighting.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, mec_shape: list[int], config: OVCSettings,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self._config = config or OVCSettings()
        shape = self._config.shape
        n_freq_ovc = None if shape is None else len(shape)
        self._ovc_start, self._ovc_count = utils.resolve_ovc_slice(len(mec_shape), n_freq_ovc)
        self._shape = mec_shape[self._ovc_start : self._ovc_start + self._ovc_count]
        self._n_freq = len(self._shape)

        # Shiny cue → mean and uncertainty.
        hidden_dim = [config.hidden_dim] * self.n_freq
        self.g_shiny_mlp = MLP([1] * self.n_freq, self.shape, hidden_dim=hidden_dim)
        self.uncertainty_mlp = MLP( [1] * self.n_freq, self.shape, [torch.tanh, torch.exp], hidden_dim=hidden_dim)  # fmt: skip

    @property
    def config(self) -> OVCSettings:
        """Return the OVC config."""
        return self._config

    def forward(  # -------------------------------------------------------------------------------
        self, landmark_id: Tensor | None, transition: LocationBelief,
    ) -> LocationBelief:  # fmt: skip
        """Apply OVC correction to environments with shiny cues.

        Args:
            landmark_id: Optional current-cell landmark ids of shape `(batch, 1)`.
            transition: Reference transition to correct.

        Returns:
            A corrected `LocationBelief`. If no shiny cues are present, returns the
            input transition unchanged.
        """
        shiny_mask = self._identify_shiny_envs(landmark_id, transition.mean[0].device)
        if shiny_mask is None:  # No shiny envs present
            return transition

        shiny_input = self._extract_shiny_cues(landmark_id, shiny_mask, transition.mean[0].device)
        freqs = range(self.ovc_start, self.ovc_start + self.n_freq)

        correction = self._predict_correction(shiny_input)
        return utils.inv_var_trans(transition, correction, shiny_mask, freqs)

    def _identify_shiny_envs(  # ------------------------------------------------------------------
        self, landmark_id: Tensor | None, device: Device,
    ) -> Tensor | None:  # fmt: skip
        """Return a mask selecting environments with shiny cues.

        Args:
            landmark_id: Optional current-cell landmark ids.
            device: Device for the returned tensor.

        Returns:
            A boolean mask of shape `(batch,)`, or `None` if no shiny cues are
            present.
        """
        if landmark_id is None:
            return None
        shiny_mask = landmark_id.squeeze(-1).to(torch.int64) != 0
        if not torch.any(shiny_mask):
            return None
        return shiny_mask.to(device=device)

    def _extract_shiny_cues(  # -------------------------------------------------------------------
        self, landmark_id: Tensor | None, shiny_mask: Tensor, device: Device
    ) -> list[Tensor]:  # fmt: skip
        """Extract shiny cue values as inputs for the OVC MLPs.

        Args:
            landmark_id: Optional current-cell landmark ids.
            shiny_mask: Boolean mask indicating which batch items have cues.
            device: Device for returned tensors.

        Returns:
            A list of cue tensors (one per OVC module). Each tensor has shape
            `(n_shiny, 1)`.
        """
        if landmark_id is None:
            raise ValueError("landmark_id is required when shiny_mask selects OVC-corrected rows.")
        shiny_tensor = (
            landmark_id.squeeze(-1)[shiny_mask].to(device=device, dtype=torch.float32).unsqueeze(-1)
        )
        return [shiny_tensor] * self.n_freq

    def _predict_correction(  # -------------------------------------------------------------------
        self, shiny_input: list[Tensor],
    ) -> LocationBelief:  # fmt: skip
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


# =================================================================================================
__all__ = ["OVCCorrection", "OVCSettings"]
