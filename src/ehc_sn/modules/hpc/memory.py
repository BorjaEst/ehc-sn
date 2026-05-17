"""Store settings and backend adapters for HPC memory representations.

This module defines the configuration models and runtime backends that connect
high-level HPC write and recall logic to concrete dense or factorized memory
stores.
"""

from __future__ import annotations

from typing import Optional, Protocol

import torch
from pydantic import BaseModel, Field, field_validator
from torch import Tensor
from torch import device as Device
from torch import dtype as DType
from torch import nn

from ehc_sn.modules.hpc.update import HebbianLayout, HebbianWriteRule
from ehc_sn.types import (
    DEFAULT_FACTOR_BANK_NAME,
    DenseMemoryStore,
    FactorMemoryStore,
    FactorSlotBank,
    MemoryEntry,
)


# =============================================================================
class LinearStoreSettings(BaseModel, extra="forbid"):
    """Marker settings for stores that only need linear-operator compatibility.

    The model is intentionally empty because dense Hebbian stores currently do
    not expose additional static store parameters beyond the write layout.
    """


# =============================================================================
class FactorStoreSettings(BaseModel, extra="forbid"):
    """Static configuration for explicit factor-memory stores.

    These settings control optional slot truncation and the allocation of
    auxiliary named banks alongside the default factor-memory bank.
    """

    memory_capacity: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of factor slots retained; unset keeps all "
        "stored steps.",
    )
    bank_names: tuple[str, ...] = Field(
        default=(),
        description="Optional additional named factor-memory banks allocated "
        "alongside the default bank.",
    )

    @field_validator("bank_names")
    @classmethod
    def _validate_bank_names(  # ----------------------------------------------
        cls,
        bank_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Validate explicit bank names for deterministic factor-store structure."""
        if len(set(bank_names)) != len(bank_names):
            raise ValueError("store.bank_names must be distinct.")
        if DEFAULT_FACTOR_BANK_NAME in bank_names:
            raise ValueError(
                "store.bank_names may not include the reserved bank name "
                f"{DEFAULT_FACTOR_BANK_NAME!r}."
            )
        return bank_names


# =============================================================================
class StoreBackend(Protocol):
    """Runtime backend contract for one memory-entry representation family."""

    def init_store(  # --------------------------------------------------------
        self,
        batch_size: int,
        *,
        device: Optional[Device] = None,
    ) -> MemoryEntry:
        """Return an empty memory entry for the configured representation."""

    def merge_rows(  # --------------------------------------------------------
        self,
        flag: Tensor,
        current: MemoryEntry,
        fresh: MemoryEntry,
    ) -> MemoryEntry:
        """Merge backend-specific rows during partial reset."""


class HebbianStoreBackend(StoreBackend, Protocol):
    """Backend contract for Hebbian store representations."""

    def apply_write(  # -------------------------------------------------------
        self,
        store: MemoryEntry,
        key: Tensor,
        value: Tensor,
        *,
        masked: bool,
    ) -> MemoryEntry:
        """Apply one Hebbian write step to the provided store."""


class AppendStoreBackend(StoreBackend, Protocol):
    """Backend contract for append-only factor-store representations."""

    def append_write(  # ------------------------------------------------------
        store: MemoryEntry,
        key: Tensor,
        value: Tensor,
        *,
        bank_name: Optional[str] = None,
    ) -> MemoryEntry:
        """Append one value atom to the provided store."""


# =============================================================================
class DenseHebbianStoreBackend(nn.Module):
    """Dense-store backend for Hebbian memory writes.

    The backend owns dense-store allocation, masked Hebbian updates, and row
    merges used by partial-reset training.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        write_rule: HebbianWriteRule,
        *,
        layout: HebbianLayout,
    ) -> None:
        """Bind dense Hebbian allocation, write, and merge behavior to one
        backend."""
        super().__init__()
        self._write_rule = write_rule
        self._feature_dim = int(layout.feature_dim)
        self.register_buffer("dense_mask", layout.dense_mask, persistent=False)

    def init_store(  # --------------------------------------------------------
        self,
        batch_size: int,
        *,
        device: Optional[Device] = None,
    ) -> DenseMemoryStore:
        """Allocate an empty dense Hebbian memory entry."""
        matrix = torch.zeros(
            (batch_size, self._feature_dim, self._feature_dim),
            dtype=torch.float,
            device=device,
        )
        return DenseMemoryStore(matrix=matrix)

    def merge_rows(  # --------------------------------------------------------
        self,
        flag: Tensor,
        current: MemoryEntry,
        fresh: MemoryEntry,
    ) -> DenseMemoryStore:
        """Merge dense Hebbian stores during partial reset."""
        if not isinstance(current, DenseMemoryStore) or not isinstance(
            fresh, DenseMemoryStore
        ):
            raise TypeError(
                "DenseHebbianStoreBackend expected dense memory stores."
            )
        return current.merged_rows(flag, fresh)

    def apply_write(  # -------------------------------------------------------
        self,
        store: MemoryEntry,
        key: Tensor,
        value: Tensor,
        *,
        masked: bool,
    ) -> DenseMemoryStore:
        """Apply a dense Hebbian update to the provided memory entry."""
        if not isinstance(store, DenseMemoryStore):
            raise TypeError(
                "DenseHebbianStoreBackend expected dense memory stores."
            )
        mask = self.dense_mask if masked else None
        return DenseMemoryStore(
            matrix=self._write_rule.apply_dense(
                store.matrix, key, value, mask=mask
            )
        )


# =============================================================================
class FactorHebbianStoreBackend:
    """Factor-store backend for Hebbian memory writes.

    This backend preserves Hebbian updates in factorized form when possible,
    materializing a dense operator only when post-clamp exactness requires it.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        write_rule: HebbianWriteRule,
        *,
        layout: HebbianLayout,
    ) -> None:
        """Bind factorized Hebbian allocation, write, and merge behavior to one
        backend."""
        self._write_rule = write_rule
        self._layout = layout
        self._feature_dim = int(layout.feature_dim)

    def init_store(  # --------------------------------------------------------
        self,
        batch_size: int,
        *,
        device: Optional[Device] = None,
    ) -> FactorMemoryStore:
        """Allocate an empty factor Hebbian memory entry."""
        return FactorMemoryStore(
            keys=torch.zeros(
                (batch_size, 0, self._feature_dim),
                dtype=torch.float,
                device=device,
            ),
            values=torch.zeros(
                (batch_size, 0, self._feature_dim),
                dtype=torch.float,
                device=device,
            ),
            valid_mask=torch.zeros(
                (batch_size, 0), dtype=torch.bool, device=device
            ),
            coefficients=torch.zeros(
                (batch_size, 0), dtype=torch.float, device=device
            ),
        )

    def merge_rows(  # --------------------------------------------------------
        self,
        flag: Tensor,
        current: MemoryEntry,
        fresh: MemoryEntry,
    ) -> FactorMemoryStore:
        """Merge factor Hebbian stores during partial reset."""
        if not isinstance(current, FactorMemoryStore) or not isinstance(
            fresh, FactorMemoryStore
        ):
            raise TypeError(
                "FactorHebbianStoreBackend expected factor memory stores."
            )
        return current.merged_rows(flag, fresh)

    def apply_write(  # -------------------------------------------------------
        self,
        store: MemoryEntry,
        key: Tensor,
        value: Tensor,
        *,
        masked: bool,
    ) -> FactorMemoryStore:
        """Apply a Hebbian update and keep the resulting store in factor form."""
        if not isinstance(store, FactorMemoryStore):
            raise TypeError(
                "FactorHebbianStoreBackend expected factor memory stores."
            )

        runtime = self._write_rule.runtime
        decayed = store.decayed(runtime.hebbian_decay)
        increment = self._layout.compile_factors(
            key, value, eta=runtime.eta, masked=masked
        )

        updated = decayed.concatenated(increment)
        dense_matrix = updated.to_dense()
        clamped = self._write_rule.clamp_memory(dense_matrix)
        if torch.equal(clamped, dense_matrix):
            return updated
        return FactorMemoryStore.from_dense(clamped)


# =============================================================================
class FactorAppendStoreBackend:
    """Factor-store backend for append-only episodic writes.

    The backend allocates explicit slot memories and delegates append semantics,
    novelty gating, and named-bank selection to the episodic write policy.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        write_system: object,
        *,
        feature_dim: int,
        settings: FactorStoreSettings,
    ) -> None:
        """Bind factor allocation, append, and merge behavior to one backend."""
        self._write_system = write_system
        self._feature_dim = int(feature_dim)
        self._settings = settings

    def init_store(  # --------------------------------------------------------
        self,
        batch_size: int,
        *,
        device: Optional[Device] = None,
    ) -> FactorMemoryStore:
        """Allocate an empty factor store using the configured store settings."""
        capacity = (
            0
            if self._settings.memory_capacity is None
            else int(self._settings.memory_capacity)
        )
        keys = torch.zeros(
            (batch_size, capacity, self._feature_dim),
            dtype=torch.float,
            device=device,
        )
        values = torch.zeros(
            (batch_size, capacity, self._feature_dim),
            dtype=torch.float,
            device=device,
        )
        valid_mask = torch.zeros(
            (batch_size, capacity), dtype=torch.bool, device=device
        )
        coefficients = torch.zeros(
            (batch_size, capacity), dtype=torch.float, device=device
        )
        banks = {
            name: FactorSlotBank(
                keys=keys.clone(),
                values=values.clone(),
                valid_mask=valid_mask.clone(),
                coefficients=coefficients.clone(),
            )
            for name in self._settings.bank_names
        }
        return FactorMemoryStore(
            keys=keys,
            values=values,
            valid_mask=valid_mask,
            coefficients=coefficients,
            banks=banks,
        )

    def merge_rows(  # --------------------------------------------------------
        self,
        flag: Tensor,
        current: MemoryEntry,
        fresh: MemoryEntry,
    ) -> FactorMemoryStore:
        """Merge append-only factor stores during partial reset."""
        if not isinstance(current, FactorMemoryStore) or not isinstance(
            fresh, FactorMemoryStore
        ):
            raise TypeError(
                "FactorAppendStoreBackend expected factor memory stores."
            )
        return current.merged_rows(flag, fresh)

    def append_write(  # ------------------------------------------------------
        self,
        store: MemoryEntry,
        key: Tensor,
        value: Tensor,
        *,
        bank_name: Optional[str] = None,
    ) -> FactorMemoryStore:
        """Append a factor-memory atom to the provided memory entry."""
        if not isinstance(store, FactorMemoryStore):
            raise TypeError(
                "FactorAppendStoreBackend expected factor memory stores."
            )
        return self._write_system.append(
            store,
            key,
            value,
            bank_name=bank_name,
            capacity=self._settings.memory_capacity,
        )


# =============================================================================
__all__ = [
    "AppendStoreBackend",
    "FactorAppendStoreBackend",
    "FactorStoreSettings",
    "DenseHebbianStoreBackend",
    "FactorHebbianStoreBackend",
    "HebbianStoreBackend",
    "LinearStoreSettings",
    "StoreBackend",
]
