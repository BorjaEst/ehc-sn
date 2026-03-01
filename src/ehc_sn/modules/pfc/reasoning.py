""" """

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, List, Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.transformer import TransformerBlockConfig, TransformerStack
from ehc_sn.types import Activation, Device, Dtype, Matrix, MemoryState, MultiScaleCode


# =================================================================================================
class ReasoningSettings(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """ """

    cortex: TransformerBlockConfig = Field(
        ...,
        description="Base transformer block config for reasoning modules.",
    )
    n_layers: int = Field(
        default=4,
        ge=1,
        description="Number of layers in the reasoning module.",
    )

    @property
    def layers(self) -> List[TransformerBlockConfig]:
        """Convenience property to construct the list of transformer block."""
        return [self.cortex for _ in range(self.n_layers)]

    n_cycles: int = Field(
        default=2,
        ge=1,
        description="Number of cycles to reason.",
    )


# =================================================================================================
@dataclass
class WorkingMemory:
    """ """

    z_H: Tensor  # Higher-level state tensor of shape [batch, seq_length, hidden_size].
    z_L: Tensor  # Lower-level state tensor of shape [batch, seq_length, hidden_size].

    def detach(self) -> "WorkingMemory":
        """Return a detached copy of the state."""
        return WorkingMemory(self.z_H.detach(), self.z_L.detach())


# =================================================================================================
class ReasoningModule(nn.Module):
    """"""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ReasoningSettings, device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config
        self.cortex = TransformerStack(config.layers, device=device, dtype=dtype)

    @property
    def config(self) -> ReasoningSettings:
        """ """
        return self._config


# =================================================================================================
class HighLvRModule(ReasoningModule):
    """High-level reasoning module (anterior dlPFC)."""

    def forward(  # ----------------------------------------------------------------------------------
        self, x: Tensor, memory: WorkingMemory,
    ) -> WorkingMemory:  # fmt: skip
        """ """
        z_H = self.cortex(memory.z_H, memory.z_L)
        return WorkingMemory(z_H, memory.z_L)


# =================================================================================================
class LowLvRModule(ReasoningModule):
    """Low-level reasoning module (posterior dlPFC)."""

    def forward(  # ----------------------------------------------------------------------------------
        self, x: Tensor, memory: WorkingMemory,
    ) -> WorkingMemory:  # fmt: skip
        """ """
        z_L = self.cortex(memory.z_L, memory.z_H + x)
        return WorkingMemory(memory.z_H, z_L)


# ==================================================================================================
def reasoning_gen(  # ----------------------------------------------------------------------------------
    x: Tensor, memory: WorkingMemory, high_module: HighLvRModule, low_module: LowLvRModule,
) -> Generator[WorkingMemory]:  # fmt: skip
    """ """
    for _ in range(high_module.config.n_cycles):
        for _ in range(low_module.config.n_cycles):
            memory = low_module(x, memory)
            yield memory
        memory = high_module(x, memory)
        yield memory
