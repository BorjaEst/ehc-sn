"""LEC (lateral entorhinal cortex) feature processing.

This package implements a simple sensory feature pathway used by TEM:

- Temporal frequency filtering of sensory input
- Normalization
- Lightweight reconstruction for the generative branch

The public entry point is `LECModel`, which exposes TEM-compatible `init_state`,
`generative`, and `inference` methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.modules.lec.filter import FreqFilterSettings, FrequencyFilter
from ehc_sn.modules.lec.norm import FeatureNorm, FeatureNormSettings
from ehc_sn.modules.lec.reconstruction import Reconstruction, ReconstructionSettings
from ehc_sn.types import Device, Dtype, MultiScaleCode
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class LECSettings(BaseModel, extra="forbid"):
    """Settings for LEC modules."""

    n_features: int = Field(
        ...,
        description="",
    )
    f_init: list[float] = Field(
        ...,
        description="",
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
        cells: Per-frequency LEC activations.
        filtered: Per-frequency unweighted filtered features.
    """

    cells: MultiScaleCode  # LEC cell activations per frequency
    filtered: MultiScaleCode  # Unweighted filtered features


# =================================================================================================
class LECModel(nn.Module):
    """LEC orchestrator: filter + normalize + reconstruct.

    The model maintains an explicit `LECState` and exposes TEM-compatible
    generative and inference interfaces.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: LECSettings,
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self._config = config
        n_features, f_init = config.n_features, config.f_init
        n_freq = len(f_init)

        # Composable submodules (single responsibility each)
        self.filter = FrequencyFilter(f_init, config.filter)
        self.norm = FeatureNorm(config.norm)
        self.reconstruct = Reconstruction(n_features, config.reconstruction)
        self.w_f = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(n_freq)])

        self.reset_parameters()

    @property
    def config(self) -> LECSettings:
        """Return the LEC config."""
        return self._config

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
        return LECState(cells=x0, filtered=x0)

    def reset_state(  # ---------------------------------------------------------------------------
        self, state: LECState,  # TODO: define based in other modules reset_state
    ) -> LECState:  # fmt: skip
        """ """
        raise NotImplementedError(
            "LEC reset_state not implemented. Use init_state or implement reset logic here."
        )

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
        filtered = self.filter(c, state.filtered)
        normalized = self.norm(filtered)
        x_inf = next_cells = [torch.sigmoid(self.w_f[f]) * normalized[f] for f in range(self.n_freq)]
        return x_inf, state.new(cells=next_cells, filtered=filtered)


# =================================================================================================
__all__ = ["LECModel", "LECState"]
