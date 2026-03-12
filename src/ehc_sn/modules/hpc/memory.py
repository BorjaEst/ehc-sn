from __future__ import annotations

"""HPC Hebbian memory write.

This module contains the Hebbian update used to write to the hippocampal memory
matrix.

The update is applied per batch element and optionally gated by a connectivity
mask (typically produced by `ehc_sn.utils.make_hebbian_write_mask`).
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.types import Device, Dtype


# =================================================================================================
class HebbianUpdateSettings(BaseModel, extra="forbid"):
    """Settings for Hebbian update modules."""

    clamp_min: float = Field(
        default=-1.0,
        description="Minimum clamp value for Hebbian memory weights.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum clamp value for Hebbian memory weights.",
    )


# =================================================================================================
@dataclass
class Runtime:
    """Runtime hyperparameters for Hebbian memory.

    These are set by the training loop (see `Model.set_runtime`) and are not
    part of the static config tree.

    Attributes:
        eta: Hebbian learning rate.
        hebbian_decay: Multiplicative decay applied to the memory before adding
            the new outer-product update.
    """

    eta: float = 0.5
    hebbian_decay: float = 0.9999


# =================================================================================================
class HebbianUpdate(nn.Module):
    """Hebbian write/update logic for the grounded-location memory matrix."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: HebbianUpdateSettings,
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """Initialize Hebbian update.

        Args:
            config: Hebbian update config (e.g., clamp range).
        """
        super().__init__()
        self._config = config
        self._runtime = Runtime()

    @property
    def config(self) -> HebbianUpdateSettings:
        """Return Hebbian update config."""
        return self._config

    @property
    def runtime(self) -> Runtime:
        """Return runtime hyperparameters."""
        return self._runtime

    def forward(  # -------------------------------------------------------------------------------
        self, memory: Tensor, p_inf: list[Tensor], p_gen: list[Tensor], *,
        mask: Optional[Tensor] = None,
    ) -> Tensor:  # fmt: skip
        """Apply a Hebbian write update.

        Args:
            memory: Current memory matrix of shape `(B, S, S)`.
            p_inf: Inferred grounded location flattened to shape `(B, S)`.
            p_gen: Generated/recalled grounded location flattened to shape
                `(B, S)`.
            mask: Optional write mask of shape `(S, S)` (or broadcastable to
                `(B, S, S)`) used to gate which synapses are updated.

        Returns:
            Updated memory matrix with decay and clamping applied.
        """
        eta, hebbian_decay = self.runtime.eta, self.runtime.hebbian_decay
        p_inf, p_gen = [torch.cat(p, dim=1) for p in (p_inf, p_gen)]
        update = torch.squeeze(torch.unsqueeze(p_inf + p_gen, 2) @ torch.unsqueeze(p_inf - p_gen, 1))
        update = update * mask.to(dtype=memory.dtype) if mask is not None else update
        return self.clamp_memory(hebbian_decay * memory + eta * update)

    def clamp_memory(self, m: Tensor) -> Tensor:
        """Clamp memory values for numerical stability and legacy parity."""
        return torch.clamp(m, min=self._config.clamp_min, max=self._config.clamp_max)


# =================================================================================================
__all__ = ["HebbianUpdateSettings", "Runtime", "HebbianUpdate"]
