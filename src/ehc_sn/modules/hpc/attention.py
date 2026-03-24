"""Explicit episodic memory and the concrete HPC attention backend.

This module owns episodic-memory retrieval and attention backend assembly:

- ``EpisodicRetrieval``: tensor-only masked-softmax recall over episodic slots
- ``HPCAttention``: assembly of the episodic backend behind the shared HPC
    contract defined in ``ehc_sn.modules.hpc._base``
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.modules.hpc._base import HPCBackendAdapter, HPCBase, HPCCommonSettings
from ehc_sn.modules.hpc.memory import EpisodicMemoryWrite, EpisodicMemoryWriteSettings, merge_episodic_memory_rows
from ehc_sn.types import Device, Dtype, EpisodicMemoryStore, MemoryEntry, MemoryState, RetrievalRole


# =================================================================================================
class AttentionSettings(BaseModel, extra="forbid"):
    """Settings for explicit episodic-memory retrieval."""

    temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Softmax temperature divisor applied after dot-product scaling.",
    )
    empty_retrieval: Literal["query", "zeros"] = Field(
        default="query",
        description="Fallback returned when no episodic slots are populated: pass through the query or return zeros.",
    )


# =================================================================================================
class EpisodicRetrieval(nn.Module):
    """Masked-softmax retrieval over explicit episodic memory slots.

    The episodic ....

    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: AttentionSettings,
    ) -> None:  # fmt: skip
        """Initialize episodic-memory retrieval.

        Args:
            config: Episodic-memory retrieval settings.
        """
        super().__init__()
        self._config = config

    @property
    def config(self) -> AttentionSettings:
        """Return episodic retrieval settings."""
        return self._config

    def forward(  # -------------------------------------------------------------------------------
        self, query: Tensor, keys: Tensor, values: Tensor, *, valid_mask: Tensor,
    ) -> Tensor:  # fmt: skip
        """Return a recalled tensor of shape ``(B, S)`` by masked softmax attention.

        Args:
            query: Flattened query tensor with shape ``(B, S)``.
            keys: Episodic memory keys with shape ``(B, T, S)``.
            values: Episodic memory values with shape ``(B, T, S)``.
            valid_mask: Boolean slot mask with shape ``(B, T)``.

        Returns:
            Recalled value tensor with shape ``(B, S)``. Rows with no valid
            slots fall back to ``query`` or zeros, according to
            ``config.empty_retrieval``.
        """
        query = query.to(dtype=keys.dtype)
        # Match standard dot-product attention scaling while preserving the
        # existing temperature control as an additional divisor.
        scale = math.sqrt(max(query.shape[1], 1)) * self.config.temperature
        logits = torch.einsum("bs,bts->bt", query, keys) / scale
        has_valid_slot = valid_mask.any(dim=1, keepdim=True)

        masked_logits = torch.where(valid_mask, logits, torch.full_like(logits, torch.finfo(logits.dtype).min))  # fmt: skip
        stabilized_logits = torch.where(has_valid_slot, masked_logits, torch.zeros_like(masked_logits))
        weights = torch.softmax(stabilized_logits, dim=1)
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        recalled = torch.einsum("bt,bts->bs", weights, values)

        fallback = self._empty_fallback(query)
        return torch.where(has_valid_slot, recalled, fallback)

    def _empty_fallback(  # -----------------------------------------------------------------------
        self, query: Tensor,
    ) -> Tensor:  # fmt: skip
        """Return the configured fallback when a row has no populated slots."""
        if self.config.empty_retrieval == "zeros":
            return torch.zeros_like(query)
        return query


# =================================================================================================
class HPCAttentionSettings(HPCCommonSettings):
    """Settings for the episodic softmax-attention hippocampal module."""

    retrieval: AttentionSettings = Field(
        default_factory=AttentionSettings,
        description="Masked softmax retrieval settings for the episodic memory backend.",
    )
    write: EpisodicMemoryWriteSettings = Field(
        default_factory=EpisodicMemoryWriteSettings,
        description="Settings for the episodic memory append-write system.",
    )


# =================================================================================================
class AttentionMemoryBackend(HPCBackendAdapter):
    """Backend adapter for episodic attention memory operations."""

    def __init__(  # ------------------------------------------------------------------------------
        self, retrieval_system: EpisodicRetrieval, write_system: EpisodicMemoryWrite,
        *,
        common_memory: bool,
    ) -> None:  # fmt: skip
        self._retrieval_system = retrieval_system
        self._write_system = write_system
        self._common_memory = common_memory

    def init_memory(  # ---------------------------------------------------------------------------
        self, batch_size: int, *,
        device: Optional[Device] = None,
    ) -> MemoryState:  # fmt: skip
        """Initialize episodic memory stores."""
        g_cued = self._write_system.init_store(batch_size, device=device)
        x_cued = g_cued if self._common_memory else self._write_system.init_store(batch_size, device=device)
        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def set_runtime(  # ---------------------------------------------------------------------------
        self, *, eta: float, hebbian_decay: float,
    ) -> None:  # fmt: skip
        """Attention memory has no runtime-adjustable write parameters."""
        del eta, hebbian_decay

    def recall_flat(  # ---------------------------------------------------------------------------
        self, query: Tensor, memory: MemoryEntry, *, role: RetrievalRole,
    ) -> Tensor:  # fmt: skip
        """Retrieve a flattened memory-space code by masked softmax attention.

        ``role`` is intentionally unused here. The caller already selects
        the role-specific memory entry via ``MemoryState.for_role(...)``,
        and this backend applies the same masked-softmax retrieval rule to both
        generative and inference recall.
        """
        del role
        store = self._expect_store(memory)
        return self._retrieval_system(query, store.keys, store.values, valid_mask=store.valid_mask)

    def update_memory(
        self,
        memory: MemoryState,
        key: Tensor,
        g_value: Tensor,
        x_value: Optional[Tensor],
    ) -> MemoryState:
        """Append a TEM step into episodic memory stores."""
        g_cued = self._expect_store(memory.g_cued)
        x_cued = self._expect_store(memory.x_cued)

        g_cued = self._write_system.append(g_cued, key, g_value)
        if self._common_memory:
            x_cued = g_cued
        elif x_value is not None:
            x_cued = self._write_system.append(x_cued, key, x_value)

        return MemoryState(g_cued=g_cued, x_cued=x_cued)

    def merge_memory_rows(self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry) -> MemoryEntry:
        """Merge episodic memory rows during partial reset."""
        return merge_episodic_memory_rows(flag, self._expect_store(current), self._expect_store(fresh))

    @staticmethod
    def _expect_store(memory: MemoryEntry) -> EpisodicMemoryStore:
        """Validate that a memory entry is an episodic store."""
        if not isinstance(memory, EpisodicMemoryStore):
            raise TypeError("HPCAttention expected episodic memory stores.")
        return memory


# =================================================================================================
class HPCAttention(HPCBase):
    """Explicit episodic-memory hippocampal module with masked softmax retrieval."""

    def __init__(  # ------------------------------------------------------------------------------
        self, n_stages: int, f_initial: list[float], config: HPCAttentionSettings,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        del n_stages, f_initial
        super().__init__(config, device=device, dtype=dtype)
        self.retrieval_module = EpisodicRetrieval(config.retrieval)
        self.write_module = EpisodicMemoryWrite(self.shape, config.write)
        self._memory_backend = AttentionMemoryBackend(
            self.retrieval_module,
            self.write_module,
            common_memory=config.common_memory,
        )

    @property
    def config(self) -> HPCAttentionSettings:
        """Return attention hippocampal module config."""
        return super().config  # type: ignore[return-value]

    @property
    def memory_backend(self) -> AttentionMemoryBackend:
        """Return the episodic-memory backend helper."""
        return self._memory_backend


# =================================================================================================
__all__ = [
    "AttentionSettings",
    "EpisodicRetrieval",
    "AttentionMemoryBackend",
    "HPCAttentionSettings",
    "HPCAttention",
]  # fmt: skip
