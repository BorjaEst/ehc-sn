"""LEC (lateral entorhinal cortex) feature processing.

This package implements a simple sensory feature pathway used by TEM:

- Temporal frequency filtering of sensory input
- Normalization
- Lightweight reconstruction for the generative branch

The public entry point is `LECModel`, which exposes TEM-compatible `init_state`,
`generative`, and `inference` methods.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn import utils
from ehc_sn.modules.lec.filter import FreqFilterSettings, FrequencyFilter
from ehc_sn.modules.lec.norm import FeatureNorm, FeatureNormSettings
from ehc_sn.modules.lec.reconstruction import Reconstruction, ReconstructionSettings
from ehc_sn.types import MultiScaleCode
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class LECSettings(BaseModel, extra="forbid"):
    """Settings for LEC modules."""

    feature_dim: int = Field(
        ...,
        ge=1,
        description="Feature dimensionality per frequency module.",
    )

    clamp_min: float = Field(
        default=-1.0,
        description="Minimum activation clamp for OVC cells.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum activation clamp for OVC cells.",
    )

    filter: FreqFilterSettings = Field(
        default_factory=FreqFilterSettings,
        description="Feature Frequency filtering module config.",
    )
    norm: FeatureNormSettings = Field(
        default_factory=FeatureNormSettings,
        description="Feature normalization module config.",
    )
    reconstruction: ReconstructionSettings = Field(
        default_factory=ReconstructionSettings,
        description="Feature reconstruction module config.",
    )


# =================================================================================================
@dataclass
class LECState(DetachMixin):
    """Container for LEC state.

    Attributes:
        features: Per-frequency LEC activations.
        filtered_features: Per-frequency unweighted filtered features.
    """

    features: MultiScaleCode  # LEC activations per frequency
    filtered_features: MultiScaleCode  # Unweighted filtered features

    @property
    def cells(self) -> MultiScaleCode:
        """Backward-compatible alias for LEC activations."""
        return self.features

    @property
    def filtered(self) -> MultiScaleCode:
        """Backward-compatible alias for unweighted filtered features."""
        return self.filtered_features

    def new(  # -----------------------------------------------------------------------------------
        self, *, features: MultiScaleCode, filtered_features: MultiScaleCode,
    ) -> LECState:  # fmt: skip
        """Return a copy with updated feature tensors."""
        return replace(self, features=features, filtered_features=filtered_features)

    def replace_rows(self, flag: Tensor, fresh: "LECState") -> "LECState":
        """Merge flagged rows from ``fresh`` for module-owned reset logic."""
        return self.new(
            features=utils.merge_multiscale_rows(flag, self.features, fresh.features),
            filtered_features=utils.merge_multiscale_rows(flag, self.filtered_features, fresh.filtered_features),
        )  # fmt: skip


# =================================================================================================
class LECModel(nn.Module):
    """LEC orchestrator: filter + normalize + reconstruct.

    The model maintains an explicit `LECState` and exposes TEM-compatible
    generative and inference interfaces.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, f_initial: list[float], config: LECSettings,
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self._config = config
        n_features = config.feature_dim
        self._n_freq = len(f_initial)
        self._shape = [n_features] * self._n_freq

        # Composable submodules (single responsibility each)
        self.filter = FrequencyFilter(f_initial, config.filter)
        self.norm = FeatureNorm(config.norm)
        self.reconstruct = Reconstruction(n_features, config.reconstruction)
        self.w_f = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(self._n_freq)])

        self.reset_parameters()

    @property
    def config(self) -> LECSettings:
        """Return the LEC config."""
        return self._config

    @property
    def shape(self) -> list[int]:
        """Return the resolved LEC multiscale shape."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Return the number of LEC frequency modules."""
        return self._n_freq

    def reset_parameters(  # ----------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Initialize parameters and buffers."""
        pass

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int, *,
        device: Optional[Device] = None,
    ) -> LECState:  # fmt: skip
        """Create an initial LEC state.

        Args:
            batch_size: Batch size for the returned state tensors.
            device: Optional device to place the returned tensors on.

        Returns:
            An initialized `LECState`.
        """
        x0 = [torch.zeros((batch_size, n), device=device) for n in self.shape]
        return LECState(features=x0, filtered_features=x0)

    def reset_state(  # ---------------------------------------------------------------------------
        self, state: LECState, reset_flag: Tensor,
    ) -> LECState:  # fmt: skip
        """Reset flagged LEC rows to a fresh episode state.

        Args:
            state: Current LEC state.
            reset_flag: Boolean / 0-1 tensor of shape ``(B,)`` indicating
                which rows should be reset.

        Returns:
            New state with flagged rows replaced by fresh initialization.
        """
        device = state.features[0].device
        reset_flag = reset_flag.to(device=device, dtype=torch.bool).view(-1)
        if not torch.any(reset_flag):
            return state

        fresh = self.init_state(int(reset_flag.shape[0]), device=device)
        return state.replace_rows(reset_flag, fresh)

    def set_runtime(  # ---------------------------------------------------------------------------
        self, **_,
    ) -> None:  # fmt: skip
        """Set runtime hyperparameters.

        This module currently does not use runtime parameters.
        """
        pass

    def forward(  # -------------------------------------------------------------------------------
        self, *, state: LECState,
    ) -> tuple[list[Tensor], LECState]:  # fmt: skip
        """ """
        raise NotImplementedError("LEC forward not implemented. Use generative() or inference().")

    def generative(  # ----------------------------------------------------------------------------
        self, x: list[Tensor],
    ) -> Tensor:  # fmt: skip
        """Reconstruct sensory input from LEC features.

        Args:
            x: Per-frequency LEC features.

        Returns:
            A reconstruction of the sensory input.
        """
        return self.reconstruct(x)

    def inference(  # -----------------------------------------------------------------------------
        self, c: Tensor, state: LECState,
    ) -> tuple[list[Tensor], LECState]:  # fmt: skip
        """Run the LEC inference update.

        Args:
            c: Sensory input tensor.
            state: Current LEC state.

        Returns:
            A tuple `(x_inf, new_state)` where `x_inf` are the inferred
            per-frequency features and `new_state` is the updated LEC state.
        """
        filtered = self.filter(c, state.filtered_features)
        normalized = self.norm(filtered)
        x_inf = next_cells = [torch.sigmoid(self.w_f[f]) * normalized[f] for f in range(self.n_freq)]
        return x_inf, state.new(features=next_cells, filtered_features=filtered)


# =================================================================================================
__all__ = ["LECModel", "LECState"]
