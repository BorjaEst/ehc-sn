"""Attractor dynamics and the concrete HPC attractor backend.

This module contains both:

- the low-level iterative attractor retrieval primitive over dense Hebbian
    memory matrices
- the concrete ``HPCAttractor`` implementation that wires attractor retrieval,
    Hebbian writes, and shared HPC orchestration together
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.hpc._base import HPCBase, HPCCommonSettings, HPCState
from ehc_sn.modules.hpc.memory import HebbianUpdate, HebbianUpdateSettings
from ehc_sn.types import Activation, Device, Dtype, MemoryEntry, MemoryState, OperationMode


# =================================================================================================
class AttractorSettings(BaseModel, extra="forbid"):
    """Settings for attractor dynamics modules."""

    kappa: float = Field(
        default=0.8,
        description="Hebbian retrieval decay term",
    )
    activation: Activation = Field(
        default="leaky_relu",
        frozen=True,
        description="Activation function for attractor dynamics.",
    )
    clamp_min: float = Field(
        default=-1.0,
        description="Minimum clamp value for attractor dynamics.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum clamp value for attractor dynamics.",
    )


# =================================================================================================
class AttractorNetwork(nn.Module):
    """Attractor retrieval dynamics (pattern completion) over a memory matrix.

    The network flattens the multi-frequency query code, applies iterative
    updates of the form:

        field = kappa * h + h @ M

    and uses stage masks to control which dimensions update at each stage.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, shape: list[int], config: AttractorSettings,
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """Initialize the attractor.

        Args:
            shape: Feature sizes per frequency module.
            config: Attractor config.
        """
        super().__init__()
        self._config = config

        self._shape, self._n_freq = list(shape), len(shape)
        self._activation_fn = utils.activation_from_str(self._config.activation)

    @property
    def config(self) -> AttractorSettings:
        """Return attractor config."""
        return self._config

    @property
    def shape(self) -> list[int]:
        """Return the per-frequency grounded-location shape."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Return the number of attractor frequency modules."""
        return self._n_freq

    def forward(  # -------------------------------------------------------------------------------
        self, p_query: list[Tensor], M: Tensor, *, 
        masks: Sequence[Tensor],
    ) -> list[Tensor]:  # fmt: skip
        """Run attractor retrieval.

        Args:
            p_query: Query grounded-location code (multi-scale), one tensor per
                frequency module with shape `(B, shape[f])`.
            M: Hebbian memory matrix of shape `(B, S, S)` where `S = sum(shape)`.
            masks: Stage masks used to gate which dimensions update at each
                iteration stage. Each mask is expected to be broadcastable
                to `h` (shape `(B, S)`).

        Returns:
            Retrieved grounded-location code (multi-scale), where each returned
            tensor has shape `(B, shape[f])`.
        """
        # Flatten query grounded locations across frequency modules.
        p, kappa = torch.cat(p_query, dim=1), self.config.kappa
        h = self.activation(p)

        # Ensure dtype consistency for numerical stability.
        masks = [m.to(dtype=h.dtype) for m in masks]
        if not masks:
            raise ValueError("masks must contain at least one stage mask.")
        M = M.to(dtype=h.dtype)

        for mask in masks:
            field = kappa * h + (h.unsqueeze(1) @ M).squeeze(1)
            h = (1 - mask) * h + mask * self.activation(field)

        # Re-split the grounded location into frequency modules.
        return list(torch.split(h, split_size_or_sections=self.shape, dim=1))

    def activation(  # ----------------------------------------------------------------------------
        self, p: Tensor,
    ) -> Tensor:  # fmt: skip
        """Apply the configured activation with clamping."""
        p = torch.clamp(p, min=self.config.clamp_min, max=self.config.clamp_max)
        return self._activation_fn(p)


# =================================================================================================
class HPCAttractorSettings(HPCCommonSettings):
    """Settings for the dense Hebbian attractor hippocampal module."""

    attractor: AttractorSettings = Field(
        default_factory=AttractorSettings,
        description="Attractor dynamics module config.",
    )
    memory: HebbianUpdateSettings = Field(
        default_factory=HebbianUpdateSettings,
        description="Hebbian update module config.",
    )


# =================================================================================================
class HPCAttractor(HPCBase):
    """Dense Hebbian-memory hippocampal module with attractor retrieval."""

    def __init__(  # ------------------------------------------------------------------------------
        self, n_stages: int, f_initial: list[float], config: HPCAttractorSettings,
        *, device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        super().__init__(config, device=device, dtype=dtype)

        masks = utils.update_to_masks(self.shape, update=utils.make_update_hierarchical(n_stages, self.n_freq))  # fmt: skip
        self.register_buffer("masks_hierarchical", masks, persistent=False)
        masks = utils.update_to_masks(self.shape, update=utils.make_update_full(n_stages, self.n_freq))
        self.register_buffer("masks_full", masks, persistent=False)

        mask = utils.make_hebbian_write_mask(n_stages, self.shape, f_initial)
        self.register_buffer("update_mask", mask, persistent=False)

        self.attractor_system = AttractorNetwork(self.shape, config.attractor, device=device, dtype=dtype)
        self.memory_system = HebbianUpdate(config.memory, device=device, dtype=dtype)

    @property
    def config(self) -> HPCAttractorSettings:
        """Return attractor hippocampal module config."""
        return super().config  # type: ignore[return-value]

    def init_memory(  # ---------------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> MemoryState:  # fmt: skip
        """Initialize dense Hebbian memory matrices."""
        feature_dim = sum(self.shape)
        g_cued = torch.zeros((batch_size, feature_dim, feature_dim), dtype=torch.float, device=device)
        x_cued = g_cued if self.config.common_memory else g_cued.clone()
        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def set_runtime(  # ---------------------------------------------------------------------------
        self, *, eta: float, hebbian_decay: float,
    ) -> None:  # fmt: skip
        """Apply runtime parameters to the Hebbian memory writer."""
        self.memory_system.runtime.eta = float(eta)
        self.memory_system.runtime.hebbian_decay = float(hebbian_decay)

    def recall(  # --------------------------------------------------------------------------------
        self, p_query: list[Tensor], state: HPCState, *, operation: OperationMode,
    ) -> list[Tensor]:  # fmt: skip
        """Retrieve grounded location via attractor dynamics."""
        memory = state.memory.for_operation(operation)
        if not isinstance(memory, torch.Tensor):
            raise TypeError("HPCAttractor expected dense Hebbian memory matrices.")
        masks = self.masks_hierarchical if operation == "generative" else self.masks_full
        return self.attractor_system(p_query, memory, masks=masks)

    def update(  # --------------------------------------------------------------------------------
        self, p_inf: list[Tensor], p_gen_gi: list[Tensor], p_xi: Optional[list[Tensor]], state: HPCState,
    ) -> HPCState:  # fmt: skip
        """Apply a Hebbian write to the dense memory matrices."""
        g_cued = state.memory.g_cued
        x_cued = state.memory.x_cued
        if not isinstance(g_cued, torch.Tensor) or not isinstance(x_cued, torch.Tensor):
            raise TypeError("HPCAttractor expected dense Hebbian memory matrices.")

        g_cued = self.memory_system(g_cued, p_inf, p_gen_gi, mask=self.update_mask)
        if self.config.common_memory:
            x_cued = g_cued
        elif p_xi is not None:
            x_cued = self.memory_system(x_cued, p_inf, p_xi)

        return HPCState(state.grounded_belief, _memory=MemoryState(g_cued=g_cued, x_cued=x_cued))

    def merge_memory_rows(  # ---------------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> MemoryEntry:  # fmt: skip
        """Merge dense memory rows during partial reset."""
        if not isinstance(current, torch.Tensor) or not isinstance(fresh, torch.Tensor):
            raise TypeError("HPCAttractor expected dense Hebbian memory matrices.")
        return utils.merge_rows(flag, current, fresh)


# =================================================================================================
__all__ = ["AttractorSettings", "AttractorNetwork", "HPCAttractorSettings", "HPCAttractor"]
