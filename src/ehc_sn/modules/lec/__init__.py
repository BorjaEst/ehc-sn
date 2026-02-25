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
from typing import List, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.modules.lec.filter import FreqFilterSettings, FrequencyFilter
from ehc_sn.modules.lec.norm import FeatureNorm, FeatureNormSettings
from ehc_sn.modules.lec.reconstruction import Reconstruction, ReconstructionSettings
from ehc_sn.types import MultiScaleCode

__all__ = ["LECModel", "LECState"]


# =================================================================================================
class LECSettings(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """Settings for LEC modules."""

    filter: FreqFilterSettings = Field(
        default_factory=FreqFilterSettings,
        description="Feature Frequency filtering module settings.",
    )
    norm: FeatureNormSettings = Field(
        default_factory=FeatureNormSettings,
        description="Feature normalization module settings.",
    )
    reconstruction: ReconstructionSettings = Field(
        default_factory=ReconstructionSettings,
        description="Feature reconstruction module settings.",
    )


# =================================================================================================
@dataclass
class LECState:
    """Container for LEC state.

    Attributes:
        cells: Per-frequency LEC activations.
        filtered: Per-frequency unweighted filtered features.
    """

    cells: MultiScaleCode  # LEC cell activations per frequency
    filtered: MultiScaleCode  # Unweighted filtered features

    def new(
        self, cells: Optional[MultiScaleCode] = None, filtered: Optional[MultiScaleCode] = None
    ) -> "LECState":
        """Return a new state with updated fields.
        If a field is not provided, the value is reset to the initial value (zeros).

        Args:
            cells: Optional new LEC cell activations.
            filtered: Optional new filtered features.

        Returns:
            A new `LECState` instance.

        Notes:
            This method performs a shallow copy of the state fields. Use
            `detach()` when you need to carry state across iterations without
            keeping autograd history.
        """
        batch_size, device = self.cells[0].shape[0], self.cells[0].device
        copy, shape = self.__dict__.copy(), [v.shape[1] for v in self.cells]
        cells = cells or [torch.zeros((batch_size, n), device=device) for n in shape]
        filtered = filtered or [torch.zeros((batch_size, n), device=device) for n in shape]
        copy.update(cells=cells, filtered=filtered)
        return LECState(**copy)

    def detach(self) -> "LECState":
        """Return a detached copy.

        Returns:
            A detached copy of the current state.
        """
        return LECState(
            cells=[v.detach() for v in self.cells],
            filtered=[v.detach() for v in self.filtered],
        )


# =================================================================================================
class LECModel(nn.Module):
    """LEC orchestrator: filter + normalize + reconstruct.

    The model maintains an explicit `LECState` and exposes TEM-compatible
    generative and inference interfaces.
    """

    def __init__(
        self,
        n_features: int,  # Number of LEC features (compressed observation)
        f_init: List[float],  # Initial firing rates per frequency
        *,
        settings: Optional[LECSettings] = None,  # LEC settings
    ):
        super().__init__()
        self._settings = settings or LECSettings()
        self._n_c, self._n_freq = n_features, len(f_init)
        self._shape = [n_features] * self.n_freq

        # Composable submodules (single responsibility each)
        self.filter = FrequencyFilter(f_init, settings.filter)
        self.norm = FeatureNorm(settings.norm)
        self.reconstructor = Reconstruction(n_features, settings.reconstruction)

        # Frequency module specific scaling of filtered sensory experience
        self.w_f = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(self.n_freq)])

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> LECState:
        """Create an initial LEC state.

        Args:
            batch_size: Batch size for the returned state tensors.
            device: Optional device to place the returned tensors on.

        Returns:
            An initialized `LECState`.
        """
        x0 = [torch.zeros((batch_size, n), device=device) for n in self.shape]
        return LECState(cells=x0, filtered=x0)

    def set_runtime(self, *, _):
        """Set runtime hyperparameters.

        This module currently does not use runtime parameters.
        """
        pass

    @property
    def settings(self) -> LECSettings:
        """Return the LEC settings."""
        return self._settings

    @property
    def shape(self) -> List[int]:
        """Return per-frequency LEC activation sizes."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Return the number of frequency modules."""
        return self._n_freq

    def forward(self, *, _) -> Tuple[List[Tensor], LECState]:
        """Not implemented.

        Raises:
            NotImplementedError: Always. Use `generative` or `inference`.
        """
        raise NotImplementedError("LEC forward not implemented. Use generative() or inference().")

    def generative(self, x: List[Tensor]) -> Tensor:
        """Reconstruct sensory input from LEC features.

        Args:
            x: Per-frequency LEC features.

        Returns:
            A reconstruction of the sensory input.
        """
        return self.reconstructor(x)

    def inference(self, c: Tensor, state: LECState) -> Tuple[List[Tensor], LECState]:
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
