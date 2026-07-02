"""Two-level recurrent reasoning primitives for the PFC module.

This file contains:
    - configuration objects for reasoning modules
    - working-memory state containers
    - high-level and low-level reasoning module implementations
    - helper functions to initialize/reset memory and to generate recurrent updates
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, cast

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.modules.transformer import TransformerBlockConfig, TransformerStack


# =============================================================================
class ReasoningSettings(BaseModel, extra="forbid"):
    """Settings for a single reasoning module (high or low level)."""

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
    def layers(self) -> list[TransformerBlockConfig]:
        """Convenience property to construct the list of transformer block."""
        return [self.cortex for _ in range(self.n_layers)]

    n_cycles: int = Field(
        default=4,
        ge=1,
        description="Number of cycles to reason.",
    )


# =============================================================================
@dataclass
class WorkingMemory:
    """Working memory state for the PFC reasoning stack.

    Attributes:
        z_H: High-level activation tensor of shape ``(B, S, D)``.
        z_L: Low-level activation tensor of shape ``(B, S, D)``.
    """

    z_H: Tensor  # Higher-level state tensor of shape [batch, seq_length, hidden_size].
    z_L: Tensor  # Lower-level state tensor of shape [batch, seq_length, hidden_size].

    def detach(self) -> "WorkingMemory":
        """Return a detached copy of the state."""
        return WorkingMemory(self.z_H.detach(), self.z_L.detach())


# =============================================================================
class ReasoningModule(nn.Module):
    """Base class for reasoning modules.

    Wraps a :class:`~ehp_sn.modules.transformer.TransformerStack` and exposes a
    persistent reset vector used to initialize and reset memory.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: ReasoningSettings,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self._config = config

        self.cortex = TransformerStack(
            config.layers, device=device, dtype=dtype
        )
        self.register_buffer(
            "reset_vector",
            torch.empty((config.cortex.hidden_size,)),
            persistent=True,
        )
        self.reset_vector = cast(Tensor, self.reset_vector)

    @property
    def config(self) -> ReasoningSettings:
        """Return the parsed settings for this reasoning module."""
        return self._config


# =============================================================================
class HighLvRModule(ReasoningModule):
    """High-level reasoning module (anterior dlPFC)."""

    def forward(  # -----------------------------------------------------------
        self,
        x: Tensor,
        memory: WorkingMemory,
    ) -> WorkingMemory:
        """Update the high-level state given current low-level state."""
        z_H = self.cortex(memory.z_H, memory.z_L)
        return WorkingMemory(z_H, memory.z_L)


# =============================================================================
class LowLvRModule(ReasoningModule):
    """Low-level reasoning module (posterior dlPFC)."""

    def forward(  # -----------------------------------------------------------
        self,
        x: Tensor,
        memory: WorkingMemory,
    ) -> WorkingMemory:
        """Update the low-level state given high-level state and inputs."""
        z_L = self.cortex(memory.z_L, memory.z_H + x)
        return WorkingMemory(memory.z_H, z_L)


# =============================================================================
def reasoning_gen(  # ---------------------------------------------------------
    x: Tensor,
    memory: WorkingMemory,
    high_module: HighLvRModule,
    low_module: LowLvRModule,
) -> Generator[WorkingMemory]:
    """Yield successive working-memory updates for one full reasoning episode.

    The schedule is nested: for each high-level cycle, run ``n_cycles`` low-level
    updates, then one high-level update.
    """
    for _ in range(high_module.config.n_cycles):
        for _ in range(low_module.config.n_cycles):
            memory = low_module(x, memory)
            yield memory
        memory = high_module(x, memory)
        yield memory


# =============================================================================
def init_memory(  # -----------------------------------------------------------
    batch_size: int,
    seq_length: int,
    high_module: HighLvRModule,
    low_module: LowLvRModule,
) -> WorkingMemory:
    """Initialize working memory using the modules' reset vectors."""
    return WorkingMemory(
        z_H=high_module.reset_vector.view(1, 1, -1)
        .expand(batch_size, seq_length, -1)
        .clone(),
        z_L=low_module.reset_vector.view(1, 1, -1)
        .expand(batch_size, seq_length, -1)
        .clone(),
    )


# =============================================================================
def reset_memory(  # ----------------------------------------------------------
    memory: WorkingMemory,
    reset_flag: Tensor,
    high_module: HighLvRModule,
    low_module: LowLvRModule,
) -> WorkingMemory:
    """Reset selected rows of working memory.

    Args:
        memory: Current working memory.
        reset_flag: Boolean tensor of shape ``(B,)``. True indicates the row is reset.
        high_module: High-level module providing its reset vector.
        low_module: Low-level module providing its reset vector.

    Returns:
        New working memory with reset rows replaced by reset vectors.
    """
    batch_size, seq_length, _ = memory.z_H.shape
    init_H = high_module.reset_vector.view(1, 1, -1).expand(
        batch_size, seq_length, -1
    )
    init_L = low_module.reset_vector.view(1, 1, -1).expand(
        batch_size, seq_length, -1
    )
    mask = reset_flag.view(-1, 1, 1)
    return WorkingMemory(
        z_H=torch.where(mask, init_H, memory.z_H),
        z_L=torch.where(mask, init_L, memory.z_L),
    )
