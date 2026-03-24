from __future__ import annotations

"""Write-side primitives and row-merge helpers for HPC memory backends.

This module owns memory-maintenance logic shared across hippocampal retrieval
backends:

- ``HebbianMemoryWrite`` for dense Hebbian-memory updates
- ``EpisodicMemoryWrite`` for append-only episodic key/value stores
- row-merge helpers used during partial reset

Retrieval primitives remain in the concrete backend modules.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.types import DenseMemoryStore, Device, Dtype, EpisodicMemoryStore


# =================================================================================================
class HebbianMemoryWriteSettings(BaseModel, extra="forbid"):
    """Settings for dense Hebbian-memory write modules."""

    clamp_min: float = Field(
        default=-1.0,
        description="Minimum clamp value for Hebbian memory weights.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum clamp value for Hebbian memory weights.",
    )


# =================================================================================================
class EpisodicMemoryWriteSettings(BaseModel, extra="forbid"):
    """Settings for append-only episodic-memory writes."""

    memory_capacity: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of episodic slots retained; unset keeps all stored steps.",
    )


# =================================================================================================
@dataclass
class HebbianMemoryRuntime:
    """Runtime hyperparameters for dense Hebbian memory.

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
class HebbianMemoryWrite(nn.Module):
    """Hebbian write/update logic for the grounded-location memory matrix."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: HebbianMemoryWriteSettings,
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """Initialize dense Hebbian-memory write.

        Args:
            config: Dense-memory write config (e.g., clamp range).
        """
        del device, dtype
        super().__init__()
        self._config = config
        self._runtime = HebbianMemoryRuntime()

    @property
    def config(self) -> HebbianMemoryWriteSettings:
        """Return dense-memory write config."""
        return self._config

    @property
    def runtime(self) -> HebbianMemoryRuntime:
        """Return runtime hyperparameters."""
        return self._runtime

    def _normalize_code(self, code: Tensor | list[Tensor], *, name: str) -> Tensor:
        """Return a flattened ``(B, S)`` code from flat or multiscale input."""
        if isinstance(code, Tensor):
            if code.ndim != 2:
                raise ValueError(f"{name} must be rank-2 `(B, S)`, got shape {tuple(code.shape)}.")
            return code

        if not code:
            raise ValueError(f"{name} must contain at least one tensor.")

        batch_size: int | None = None
        for index, tensor in enumerate(code):
            if tensor.ndim != 2:
                raise ValueError(f"{name}[{index}] must be rank-2 `(B, N)`, got shape {tuple(tensor.shape)}.")
            if batch_size is None:
                batch_size = int(tensor.shape[0])
            elif int(tensor.shape[0]) != batch_size:
                raise ValueError(f"All tensors in {name} must share the same batch size.")

        return torch.cat(code, dim=1)

    def forward(  # -------------------------------------------------------------------------------
        self, memory: Tensor, p_inf: Tensor | list[Tensor], p_gen: Tensor | list[Tensor], *,
        mask: Optional[Tensor] = None,
    ) -> Tensor:  # fmt: skip
        """Apply a dense Hebbian-memory write update.

        Args:
            memory: Current memory matrix of shape `(B, S, S)`.
            p_inf: Inferred grounded location, either already flattened to
                shape `(B, S)` or provided as a multiscale list of rank-2
                tensors that will be concatenated along feature dimension.
            p_gen: Generated/recalled grounded location, either already
                flattened to shape `(B, S)` or provided as a multiscale list of
                rank-2 tensors that will be concatenated along feature
                dimension.
            mask: Optional write mask of shape `(S, S)` (or broadcastable to
                `(B, S, S)`) used to gate which synapses are updated.

        Returns:
            Updated memory matrix with decay and clamping applied.
        """
        if memory.ndim != 3:
            raise ValueError(f"memory must be rank-3 `(B, S, S)`, got shape {tuple(memory.shape)}.")

        eta, hebbian_decay = self.runtime.eta, self.runtime.hebbian_decay
        p_inf = self._normalize_code(p_inf, name="p_inf")
        p_gen = self._normalize_code(p_gen, name="p_gen")

        batch_size, feature_dim = int(memory.shape[0]), int(memory.shape[1])
        if int(memory.shape[2]) != feature_dim:
            raise ValueError(
                f"memory must be square over feature dimension, got shape {tuple(memory.shape)}."
            )
        if int(p_inf.shape[0]) != batch_size or int(p_gen.shape[0]) != batch_size:
            raise ValueError("memory, p_inf, and p_gen must share the same batch size.")
        if int(p_inf.shape[1]) != feature_dim or int(p_gen.shape[1]) != feature_dim:
            raise ValueError(
                f"p_inf and p_gen must have width {feature_dim} to match memory, "
                f"got {int(p_inf.shape[1])} and {int(p_gen.shape[1])}."
            )

        update = (p_inf + p_gen).unsqueeze(2) @ (p_inf - p_gen).unsqueeze(1)
        update = update * mask.to(dtype=memory.dtype) if mask is not None else update
        return self.clamp_memory(hebbian_decay * memory + eta * update)

    def clamp_memory(self, m: Tensor) -> Tensor:
        """Clamp memory values for numerical stability and legacy parity."""
        return torch.clamp(m, min=self._config.clamp_min, max=self._config.clamp_max)


# =================================================================================================
class EpisodicMemoryWrite:
    """Append-only episodic-store allocation and write helpers."""

    def __init__(
        self,
        shape: list[int],
        config: EpisodicMemoryWriteSettings,
    ) -> None:
        """Initialize episodic-memory write helpers.

        Args:
            shape: Grounded-location feature size per frequency module.
            config: Append-write settings, including optional slot capacity.
        """
        self._shape = list(shape)
        self._config = config

    @property
    def config(self) -> EpisodicMemoryWriteSettings:
        """Return append-write settings."""
        return self._config

    @property
    def memory_capacity(self) -> Optional[int]:
        """Return the configured episodic-memory capacity."""
        return self.config.memory_capacity

    def init_store(
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> EpisodicMemoryStore:  # fmt: skip
        """Create an empty episodic store for the given batch size."""
        feature_dim = sum(self._shape)
        capacity = 0 if self.memory_capacity is None else int(self.memory_capacity)
        keys = torch.zeros((batch_size, capacity, feature_dim), dtype=torch.float, device=device)
        values = torch.zeros((batch_size, capacity, feature_dim), dtype=torch.float, device=device)
        valid_mask = torch.zeros((batch_size, capacity), dtype=torch.bool, device=device)
        return EpisodicMemoryStore(keys=keys, values=values, valid_mask=valid_mask)

    def append(
        self, store: EpisodicMemoryStore, key: Tensor, value: Tensor,
    ) -> EpisodicMemoryStore:  # fmt: skip
        """Append one key/value item per batch row, truncating oldest items if needed."""
        key = key.to(dtype=store.keys.dtype) if store.capacity > 0 else key.to(dtype=torch.float)
        value = value.to(dtype=store.values.dtype) if store.capacity > 0 else value.to(dtype=torch.float)

        keys = torch.cat((store.keys, key.unsqueeze(1)), dim=1)
        values = torch.cat((store.values, value.unsqueeze(1)), dim=1)
        valid_mask = torch.cat(
            (
                store.valid_mask,
                torch.ones((key.shape[0], 1), dtype=torch.bool, device=key.device),
            ),
            dim=1,
        )

        if self.memory_capacity is not None:
            capacity = int(self.memory_capacity)
            keys = keys[:, -capacity:, :]
            values = values[:, -capacity:, :]
            valid_mask = valid_mask[:, -capacity:]

        return EpisodicMemoryStore(keys=keys, values=values, valid_mask=valid_mask)


# =================================================================================================
def merge_dense_memory_rows(
    flag: Tensor,
    current: DenseMemoryStore,
    fresh: DenseMemoryStore,
) -> DenseMemoryStore:
    """Replace flagged dense-memory rows with fresh rows during partial reset."""
    return DenseMemoryStore(matrix=utils.merge_rows(flag, current.matrix, fresh.matrix))


# =================================================================================================
def merge_episodic_memory_rows(
    flag: Tensor,
    current: EpisodicMemoryStore,
    fresh: EpisodicMemoryStore,
) -> EpisodicMemoryStore:
    """Replace flagged episodic-memory rows with fresh rows during partial reset.

    ``flag`` follows the shared partial-reset convention: flagged rows are taken
    from ``fresh`` and unflagged rows are kept from ``current``.
    """
    target = max(current.capacity, fresh.capacity)
    current = _pad_store(current, target)
    fresh = _pad_store(fresh, target)
    return utils.merge_tree_rows(flag, current, fresh)


# =================================================================================================
def _pad_store(store: EpisodicMemoryStore, target_capacity: int) -> EpisodicMemoryStore:
    """Pad a store with empty slots up to ``target_capacity``."""
    if store.capacity >= target_capacity:
        return store

    pad = target_capacity - store.capacity
    key_padding = torch.zeros(
        (*store.keys.shape[:1], pad, store.keys.shape[2]),
        dtype=store.keys.dtype,
        device=store.keys.device,
    )
    value_padding = torch.zeros(
        (*store.values.shape[:1], pad, store.values.shape[2]),
        dtype=store.values.dtype,
        device=store.values.device,
    )
    mask_padding = torch.zeros(
        (*store.valid_mask.shape[:1], pad),
        dtype=store.valid_mask.dtype,
        device=store.valid_mask.device,
    )
    return EpisodicMemoryStore(
        keys=torch.cat((store.keys, key_padding), dim=1),
        values=torch.cat((store.values, value_padding), dim=1),
        valid_mask=torch.cat((store.valid_mask, mask_padding), dim=1),
    )


# =================================================================================================
__all__ = [
    "HebbianMemoryWriteSettings",
    "EpisodicMemoryWriteSettings",
    "HebbianMemoryRuntime",
    "HebbianMemoryWrite",
    "EpisodicMemoryWrite",
    "merge_dense_memory_rows",
    "merge_episodic_memory_rows",
]
