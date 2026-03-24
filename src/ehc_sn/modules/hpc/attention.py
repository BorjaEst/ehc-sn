"""Explicit episodic memory and the concrete HPC attention backend.

This module contains both:

- the low-level episodic key/value storage and masked softmax retrieval logic
- the concrete ``HPCAttention`` implementation that plugs that logic into the
    shared HPC contract defined in ``ehc_sn.modules.hpc._base``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.hpc._base import HPCBase, HPCCommonSettings, HPCState
from ehc_sn.types import Device, Dtype, MemoryEntry, MemoryState, OperationMode
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class AttentionSettings(BaseModel, extra="forbid"):
    """Settings for explicit episodic-memory retrieval."""

    memory_capacity: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum episodic memory slots retained by the softmax backend; unset keeps all slots.",
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Softmax temperature divisor applied after dot-product scaling.",
    )
    empty_retrieval: Literal["query", "zeros"] = Field(
        default="query",
        description="Fallback returned when no episodic slots are populated.",
    )


# =================================================================================================
@dataclass
class EpisodicMemoryStore(DetachMixin):
    """Fixed-shape episodic key/value store.

    Attributes:
        keys: Stored key vectors with shape ``(B, T, S)``.
        values: Stored value vectors with shape ``(B, T, S)``.
        valid_mask: Boolean mask of shape ``(B, T)`` marking populated slots.
    """

    keys: Tensor
    values: Tensor
    valid_mask: Tensor

    @property
    def capacity(self) -> int:
        """Return the current slot count carried by the store."""
        return int(self.keys.shape[1])


# =================================================================================================
class EpisodicAttention(nn.Module):
    """Explicit episodic-memory backend for HPC retrieval and writes."""

    def __init__(  # ------------------------------------------------------------------------------
        self, shape: list[int], config: AttentionSettings,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Initialize the explicit episodic-memory backend.

        Args:
            shape: Grounded-location feature size per frequency module.
            config: Episodic-memory retrieval settings.
        """
        super().__init__()
        self._shape = list(shape)
        self._config = config

    @property
    def config(self) -> AttentionSettings:
        """Return attention backend settings."""
        return self._config

    @property
    def memory_capacity(self) -> Optional[int]:
        """Return the configured episodic-memory capacity."""
        return self.config.memory_capacity

    def init_store(  # ----------------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> EpisodicMemoryStore:  # fmt: skip
        """Create an empty episodic store for the given batch size."""
        feature_dim = sum(self._shape)
        capacity = 0 if self.memory_capacity is None else int(self.memory_capacity)
        keys = torch.zeros((batch_size, capacity, feature_dim), dtype=torch.float, device=device)
        values = torch.zeros((batch_size, capacity, feature_dim), dtype=torch.float, device=device)
        valid_mask = torch.zeros((batch_size, capacity), dtype=torch.bool, device=device)
        return EpisodicMemoryStore(keys=keys, values=values, valid_mask=valid_mask)

    def append(  # --------------------------------------------------------------------------------
        self, store: EpisodicMemoryStore, key: Tensor, value: Tensor,
    ) -> EpisodicMemoryStore:  # fmt: skip
        """Append one key/value item per batch row, truncating oldest items if needed."""
        key = key.to(dtype=store.keys.dtype) if store.capacity > 0 else key.to(dtype=torch.float)
        value = value.to(dtype=store.values.dtype) if store.capacity > 0 else value.to(dtype=torch.float)

        keys = torch.cat((store.keys, key.unsqueeze(1)), dim=1)
        values = torch.cat((store.values, value.unsqueeze(1)), dim=1)
        valid_mask = torch.cat((store.valid_mask, torch.ones((key.shape[0], 1), dtype=torch.bool, device=key.device),), dim=1)  # fmt: skip

        if self.memory_capacity is not None:
            capacity = int(self.memory_capacity)
            keys = keys[:, -capacity:, :]
            values = values[:, -capacity:, :]
            valid_mask = valid_mask[:, -capacity:]

        return EpisodicMemoryStore(keys=keys, values=values, valid_mask=valid_mask)

    def recall(  # --------------------------------------------------------------------------------
        self, query: Tensor, store: EpisodicMemoryStore,
    ) -> Tensor:  # fmt: skip
        """Retrieve a value tensor of shape ``(B, S)`` by masked softmax attention."""
        if store.capacity == 0:
            return self._empty_fallback(query)

        query = query.to(dtype=store.keys.dtype)
        scale = math.sqrt(max(query.shape[1], 1)) * self.config.temperature
        logits = torch.einsum("bs,bts->bt", query, store.keys) / scale
        valid_mask = store.valid_mask
        any_valid = valid_mask.any(dim=1, keepdim=True)

        masked_logits = torch.where(valid_mask, logits, torch.full_like(logits, torch.finfo(logits.dtype).min))  # fmt: skip
        safe_logits = torch.where(any_valid, masked_logits, torch.zeros_like(masked_logits))
        weights = torch.softmax(safe_logits, dim=1)
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        retrieved = torch.einsum("bt,bts->bs", weights, store.values)

        fallback = self._empty_fallback(query)
        return torch.where(any_valid, retrieved, fallback)

    def _empty_fallback(  # -----------------------------------------------------------------------
        self, query: Tensor,
    ) -> Tensor:  # fmt: skip
        """Return the configured fallback when memory is empty."""
        if self.config.empty_retrieval == "zeros":
            return torch.zeros_like(query)
        return query


def merge_episodic_memory_rows(
    flag: Tensor,
    current: EpisodicMemoryStore,
    fresh: EpisodicMemoryStore,
) -> EpisodicMemoryStore:
    """Replace flagged batch rows with fresh episodic memory rows."""
    target = max(current.capacity, fresh.capacity)
    current = _pad_store(current, target)
    fresh = _pad_store(fresh, target)
    return utils.merge_tree_rows(flag, current, fresh)


def _pad_store(store: EpisodicMemoryStore, target_capacity: int) -> EpisodicMemoryStore:
    """Pad a store with empty slots up to ``target_capacity``."""
    if store.capacity >= target_capacity:
        return store
    pad = target_capacity - store.capacity
    k0 = torch.zeros((*store.keys.shape[:1], pad, store.keys.shape[2]), dtype=store.keys.dtype, device=store.keys.device)  # fmt: skip
    keys = torch.cat((store.keys, k0), dim=1)
    v0 = torch.zeros((*store.values.shape[:1], pad, store.values.shape[2]), dtype=store.values.dtype, device=store.values.device)  # fmt: skip
    values = torch.cat((store.values, v0), dim=1)
    m0 = torch.zeros((*store.valid_mask.shape[:1], pad), dtype=store.valid_mask.dtype, device=store.valid_mask.device)  # fmt: skip
    valid_mask = torch.cat((store.valid_mask, m0), dim=1)
    return EpisodicMemoryStore(keys=keys, values=values, valid_mask=valid_mask)


# =================================================================================================
class HPCAttentionSettings(HPCCommonSettings):
    """Settings for the episodic softmax-attention hippocampal module."""

    attention: AttentionSettings = Field(
        default_factory=AttentionSettings,
        description="Masked softmax retrieval settings for the episodic memory backend.",
    )

    # memory: HebbianUpdateSettings = Field  # TODO


# =================================================================================================
class HPCAttention(HPCBase):
    """Explicit episodic-memory hippocampal module with masked softmax retrieval."""

    def __init__(  # ------------------------------------------------------------------------------
        self, n_stages: int, f_initial: list[float], config: HPCAttentionSettings,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        del n_stages, f_initial
        super().__init__(config, device=device, dtype=dtype)
        self.attention_system = EpisodicAttention(self.shape, config.attention, device=device, dtype=dtype)

    @property
    def config(self) -> HPCAttentionSettings:
        """Return attention hippocampal module config."""
        return super().config  # type: ignore[return-value]

    def init_memory(  # ---------------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> MemoryState:  # fmt: skip
        """Initialize episodic memory stores."""
        g_cued = self.attention_system.init_store(batch_size, device=device)
        x_cued = (
            g_cued
            if self.config.common_memory
            else self.attention_system.init_store(batch_size, device=device)
        )
        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def set_runtime(  # ---------------------------------------------------------------------------
        self, *, eta: float, hebbian_decay: float,
    ) -> None:  # fmt: skip
        """Attention memory has no Hebbian runtime knobs in the current implementation."""
        del eta, hebbian_decay

    def recall(  # --------------------------------------------------------------------------------
        self, p_query: list[Tensor], state: HPCState, *, operation: OperationMode,
    ) -> list[Tensor]:  # fmt: skip
        """Retrieve grounded location via masked softmax over episodic memory."""
        memory = state.memory.for_operation(operation)
        if not isinstance(memory, EpisodicMemoryStore):
            raise TypeError("HPCAttention expected episodic memory stores.")
        recalled = self.attention_system.recall(torch.cat(p_query, dim=1), memory)
        return list(torch.split(recalled, split_size_or_sections=self.shape, dim=1))

    def update(  # --------------------------------------------------------------------------------
        self, p_inf: list[Tensor], p_gen_gi: list[Tensor], p_xi: Optional[list[Tensor]], state: HPCState,
    ) -> HPCState:  # fmt: skip
        """Append the current step into episodic memory stores."""
        g_cued = state.memory.g_cued
        x_cued = state.memory.x_cued
        if not isinstance(g_cued, EpisodicMemoryStore) or not isinstance(x_cued, EpisodicMemoryStore):
            raise TypeError("HPCAttention expected episodic memory stores.")

        key = torch.cat(p_inf, dim=1)
        g_cued = self.attention_system.append(g_cued, key, torch.cat(p_gen_gi, dim=1))
        if self.config.common_memory:
            x_cued = g_cued
        elif p_xi is not None:
            x_cued = self.attention_system.append(x_cued, key, torch.cat(p_xi, dim=1))

        return HPCState(state.grounded_belief, _memory=MemoryState(g_cued=g_cued, x_cued=x_cued))

    def merge_memory_rows(  # ---------------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> MemoryEntry:  # fmt: skip
        """Merge episodic memory rows during partial reset."""
        if not isinstance(current, EpisodicMemoryStore) or not isinstance(fresh, EpisodicMemoryStore):
            raise TypeError("HPCAttention expected episodic memory stores.")
        return merge_episodic_memory_rows(flag, current, fresh)


# =================================================================================================
__all__ = [
    "AttentionSettings", "EpisodicAttention", "EpisodicMemoryStore", "merge_episodic_memory_rows",
    "HPCAttentionSettings", "HPCAttention",
]  # fmt: skip
