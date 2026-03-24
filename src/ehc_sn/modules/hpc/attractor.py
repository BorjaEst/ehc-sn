"""Attractor dynamics and the concrete HPC attractor backend.

This module owns dense-memory retrieval and dense backend assembly:

- ``AttractorNetwork`` for iterative attractor-based recall over dense memory
- ``HPCAttractor`` which plugs the attractor backend into the shared HPC
    contract defined in ``ehc_sn.modules.hpc._base``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.hpc._base import HPCBackendAdapter, HPCBase, HPCCommonSettings
from ehc_sn.modules.hpc.memory import HebbianMemoryWrite, HebbianMemoryWriteSettings, merge_dense_memory_rows
from ehc_sn.types import Activation, DenseMemoryStore, Device, Dtype, MemoryEntry, MemoryState, RetrievalRole


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
        self, config: AttractorSettings,
    ) -> None:  # fmt: skip
        """Initialize the attractor.

        Args:
            config: Attractor config.
        """
        super().__init__()
        self._config = config
        self._activation_fn = utils.activation_from_str(self._config.activation)

    @property
    def config(self) -> AttractorSettings:
        """Return attractor config."""
        return self._config

    def forward(  # -------------------------------------------------------------------------------
        self, p_query: Tensor, M: Tensor, *, masks: Sequence[Tensor],
    ) -> Tensor:  # fmt: skip
        """Run attractor retrieval.

        Args:
            p_query: Flattened query grounded-location code with shape `(B, S)`
                where `S = sum(shape)`.
            M: Hebbian memory matrix of shape `(B, S, S)` where `S = sum(shape)`.
            masks: Stage masks used to gate which dimensions update at each
                iteration stage. Each mask is expected to be broadcastable
                to `h` (shape `(B, S)`).

        Returns:
            Retrieved grounded-location code with shape `(B, S)`.
        """
        kappa = self.config.kappa
        h = self.activation(p_query)

        # Ensure dtype consistency for numerical stability.
        masks = [m.to(dtype=h.dtype) for m in masks]
        if not masks:
            raise ValueError("masks must contain at least one stage mask.")
        M = M.to(dtype=h.dtype)

        for mask in masks:
            field = kappa * h + (h.unsqueeze(1) @ M).squeeze(1)
            h = (1 - mask) * h + mask * self.activation(field)

        return h

    def activation(  # ----------------------------------------------------------------------------
        self, p: Tensor,
    ) -> Tensor:  # fmt: skip
        """Apply the configured activation with clamping."""
        p = torch.clamp(p, min=self.config.clamp_min, max=self.config.clamp_max)
        return self._activation_fn(p)


# =================================================================================================
class HPCAttractorSettings(HPCCommonSettings):
    """Settings for the dense Hebbian attractor hippocampal module."""

    retrieval: AttractorSettings = Field(
        default_factory=AttractorSettings,
        description="Attractor dynamics module config.",
    )
    write: HebbianMemoryWriteSettings = Field(
        default_factory=HebbianMemoryWriteSettings,
        description="Dense Hebbian write module config.",
    )


# =================================================================================================
class AttractorMemoryBackend(HPCBackendAdapter):
    """Backend helper for dense Hebbian attractor memory operations."""

    def __init__(  # ------------------------------------------------------------------------------
        self, retrieval_system: AttractorNetwork, write_system: HebbianMemoryWrite,
        *,
        masks_hierarchical: Tensor, masks_full: Tensor,
        update_mask: Tensor, feature_dim: int, common_memory: bool,
    ) -> None:  # fmt: skip
        """Initialize the attractor memory backend."""
        self._retrieval_system = retrieval_system
        self._write_system = write_system
        self._masks_hierarchical = masks_hierarchical
        self._masks_full = masks_full
        self._update_mask = update_mask
        self._feature_dim = feature_dim
        self._common_memory = common_memory

    def init_memory(  # ---------------------------------------------------------------------------
        self, batch_size: int, *,
        device: Optional[Device] = None,
    ) -> MemoryState:  # fmt: skip
        """Initialize dense Hebbian memory stores."""
        g_cued = DenseMemoryStore(matrix=torch.zeros((batch_size, self._feature_dim, self._feature_dim), dtype=torch.float, device=device))  # fmt: skip
        x_cued = g_cued if self._common_memory else g_cued.clone()
        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def set_runtime(  # ---------------------------------------------------------------------------
        self, *, eta: float, hebbian_decay: float,
    ) -> None:  # fmt: skip
        """Apply runtime parameters to the dense-memory write primitive."""
        self._write_system.runtime.eta = float(eta)
        self._write_system.runtime.hebbian_decay = float(hebbian_decay)

    def recall_flat(  # ---------------------------------------------------------------------------
        self, query: Tensor, memory: MemoryEntry, *, role: RetrievalRole,
    ) -> Tensor:  # fmt: skip
        """Retrieve a flattened grounded-location code via attractor dynamics."""
        store = self._expect_store(memory)
        masks = self._masks_hierarchical if role == "generative" else self._masks_full
        return self._retrieval_system(query, store.matrix, masks=masks)

    def update_memory(  # -------------------------------------------------------------------------
        self,
        memory: MemoryState,
        key: Tensor,
        g_value: Tensor,
        x_value: Optional[Tensor],
    ) -> MemoryState:
        """Apply the dense-memory write primitive to attractor stores."""
        g_cued = self._expect_store(memory.g_cued)
        x_cued = self._expect_store(memory.x_cued)

        g_cued = DenseMemoryStore(matrix=self._write_system(g_cued.matrix, key, g_value, mask=self._update_mask))
        if self._common_memory:
            x_cued = g_cued
        elif x_value is not None:
            x_cued = DenseMemoryStore(matrix=self._write_system(x_cued.matrix, key, x_value))

        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def merge_memory_rows(self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry) -> MemoryEntry:
        """Merge dense memory rows during partial reset."""
        current_store = self._expect_store(current)
        fresh_store = self._expect_store(fresh)
        return merge_dense_memory_rows(flag, current_store, fresh_store)

    @staticmethod
    def _expect_store(memory: MemoryEntry) -> DenseMemoryStore:
        """Validate that a memory entry is a dense Hebbian store."""
        if not isinstance(memory, DenseMemoryStore):
            raise TypeError("HPCAttractor expected dense Hebbian memory stores.")
        return memory


# =================================================================================================
class HPCAttractor(HPCBase):
    """Dense-memory hippocampal module with explicit retrieval/write/backend layers."""

    def __init__(  # ------------------------------------------------------------------------------
        self, n_stages: int, f_initial: list[float], config: HPCAttractorSettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        super().__init__(config, device=device, dtype=dtype)

        masks = utils.update_to_masks(self.shape, update=utils.make_update_hierarchical(n_stages, self.n_freq))  # fmt: skip
        self.register_buffer("masks_hierarchical", masks, persistent=False)
        masks = utils.update_to_masks(self.shape, update=utils.make_update_full(n_stages, self.n_freq))
        self.register_buffer("masks_full", masks, persistent=False)

        mask = utils.make_hebbian_write_mask(n_stages, self.shape, f_initial)
        self.register_buffer("update_mask", mask, persistent=False)

        self.retrieval_module = AttractorNetwork(config.retrieval)
        self.write_module = HebbianMemoryWrite(config.write, device=device, dtype=dtype)
        self._memory_backend = AttractorMemoryBackend(
            self.retrieval_module,
            self.write_module,
            masks_hierarchical=self.masks_hierarchical,
            masks_full=self.masks_full,
            update_mask=self.update_mask,
            feature_dim=sum(self.shape),
            common_memory=config.common_memory,
        )

    @property
    def config(self) -> HPCAttractorSettings:
        """Return attractor hippocampal module config."""
        return super().config  # type: ignore[return-value]

    @property
    def memory_backend(self) -> AttractorMemoryBackend:
        """Return the dense-memory backend helper."""
        return self._memory_backend


# =================================================================================================
__all__ = [
    "AttractorSettings", "AttractorNetwork", "AttractorMemoryBackend", "HPCAttractorSettings",
    "HPCAttractor",
]  # fmt: skip
