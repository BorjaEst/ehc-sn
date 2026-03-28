"""Store settings, backends, and compatibility validation for HPC memory modules."""

from __future__ import annotations

from typing import Callable, Literal, Optional, Protocol

import torch
from pydantic import BaseModel, Field, field_validator
from torch import Tensor, nn

from ehc_sn.types import DEFAULT_FACTOR_BANK_NAME, DenseMemoryStore, Device, FactorMemoryStore, FactorSlotBank, MemoryEntry


# =================================================================================================
class LinearStoreSettings(BaseModel, extra="forbid"):
    """Settings for Hebbian stores that must expose a linear operator."""

    representation: Literal["dense", "factor"] = Field(
        default="dense",
        description="Internal representation used to store the Hebbian operator.",
    )


# =================================================================================================
class FactorStoreSettings(BaseModel, extra="forbid"):
    """Settings for explicit factor-memory stores."""

    memory_capacity: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of factor slots retained; unset keeps all stored steps.",
    )
    bank_names: tuple[str, ...] = Field(
        default=(),
        description="Optional additional named factor-memory banks allocated alongside the default bank.",
    )

    @field_validator("bank_names")
    @classmethod
    def _validate_bank_names(cls, bank_names: tuple[str, ...]) -> tuple[str, ...]:
        """Validate explicit bank names for deterministic factor-store structure."""
        if len(set(bank_names)) != len(bank_names):
            raise ValueError("store.bank_names must be distinct.")
        if DEFAULT_FACTOR_BANK_NAME in bank_names:
            raise ValueError(f"store.bank_names may not include the reserved bank name {DEFAULT_FACTOR_BANK_NAME!r}.")
        return bank_names


# =================================================================================================
class StoreBackend(Protocol):
    """Behavior-bearing backend for one memory-entry representation family."""

    supports_linear_view: bool
    supports_factor_view: bool

    def init_store(  # ---------------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> MemoryEntry:  # fmt: skip
        """Return an empty memory entry for the configured representation."""

    def merge_rows(  # ---------------------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> MemoryEntry:  # fmt: skip
        """Merge backend-specific rows during partial reset."""


class HebbianStoreBackend(StoreBackend, Protocol):
    """Backend contract for Hebbian store representations."""

    def apply_write(  # --------------------------------------------------------------------------
        self, store: MemoryEntry, key: Tensor, value: Tensor, *, masked: bool,
    ) -> MemoryEntry:  # fmt: skip
        """Apply one Hebbian write step to the provided store."""


class AppendStoreBackend(StoreBackend, Protocol):
    """Backend contract for append-only factor-store representations."""

    def append_write(  # -------------------------------------------------------------------------
        self, store: MemoryEntry, key: Tensor, value: Tensor, *, bank_name: Optional[str] = None,
    ) -> MemoryEntry:  # fmt: skip
        """Append one value atom to the provided store."""


# =================================================================================================
class DenseHebbianStoreBackend(nn.Module):
    """Dense-store backend for Hebbian memory writes."""

    supports_linear_view = True
    supports_factor_view = False

    def __init__(self, write_system: nn.Module, *, feature_dim: int, update_mask: Tensor) -> None:
        """Bind dense Hebbian allocation, write, and merge behavior to one backend."""
        super().__init__()
        self._write_system = write_system
        self._feature_dim = int(feature_dim)
        self.register_buffer("update_mask", update_mask, persistent=False)

    def init_store(  # ---------------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> DenseMemoryStore:  # fmt: skip
        """Allocate an empty dense Hebbian memory entry."""
        matrix = torch.zeros((batch_size, self._feature_dim, self._feature_dim), dtype=torch.float, device=device)
        return DenseMemoryStore(matrix=matrix)

    def merge_rows(  # ---------------------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> DenseMemoryStore:  # fmt: skip
        """Merge dense Hebbian stores during partial reset."""
        if not isinstance(current, DenseMemoryStore) or not isinstance(fresh, DenseMemoryStore):
            raise TypeError("DenseHebbianStoreBackend expected dense memory stores.")
        return current.merged_rows(flag, fresh)

    def apply_write(  # --------------------------------------------------------------------------
        self, store: MemoryEntry, key: Tensor, value: Tensor, *, masked: bool,
    ) -> DenseMemoryStore:  # fmt: skip
        """Apply a dense Hebbian update to the provided memory entry."""
        if not isinstance(store, DenseMemoryStore):
            raise TypeError("DenseHebbianStoreBackend expected dense memory stores.")
        mask = self.update_mask if masked else None
        return DenseMemoryStore(matrix=self._write_system(store.matrix, key, value, mask=mask))


# =================================================================================================
class FactorHebbianStoreBackend:
    """Factor-store backend for Hebbian memory writes."""

    supports_linear_view = True
    supports_factor_view = True

    def __init__(  # ------------------------------------------------------------------------------
        self, write_system: nn.Module, *,
        feature_dim: int,
        n_stages: int,
        shape: list[int],
        f_initial: list[float],
        compile_hebbian_factors: Callable[..., FactorMemoryStore],
        compile_masked_hebbian_factors: Callable[..., FactorMemoryStore],
    ) -> None:  # fmt: skip
        """Bind factorized Hebbian allocation, write, and merge behavior to one backend."""
        self._write_system = write_system
        self._feature_dim = int(feature_dim)
        self._n_stages = int(n_stages)
        self._shape = list(shape)
        self._f_initial = list(f_initial)
        self._compile_hebbian_factors = compile_hebbian_factors
        self._compile_masked_hebbian_factors = compile_masked_hebbian_factors

    def init_store(  # ---------------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> FactorMemoryStore:  # fmt: skip
        """Allocate an empty factor Hebbian memory entry."""
        return FactorMemoryStore(
            keys=torch.zeros((batch_size, 0, self._feature_dim), dtype=torch.float, device=device),
            values=torch.zeros((batch_size, 0, self._feature_dim), dtype=torch.float, device=device),
            valid_mask=torch.zeros((batch_size, 0), dtype=torch.bool, device=device),
            coefficients=torch.zeros((batch_size, 0), dtype=torch.float, device=device),
        )

    def merge_rows(  # ---------------------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> FactorMemoryStore:  # fmt: skip
        """Merge factor Hebbian stores during partial reset."""
        if not isinstance(current, FactorMemoryStore) or not isinstance(fresh, FactorMemoryStore):
            raise TypeError("FactorHebbianStoreBackend expected factor memory stores.")
        return current.merged_rows(flag, fresh)

    def apply_write(  # --------------------------------------------------------------------------
        self, store: MemoryEntry, key: Tensor, value: Tensor, *, masked: bool,
    ) -> FactorMemoryStore:  # fmt: skip
        """Apply a Hebbian update and keep the resulting store in factor form."""
        if not isinstance(store, FactorMemoryStore):
            raise TypeError("FactorHebbianStoreBackend expected factor memory stores.")

        runtime = self._write_system.runtime
        decayed = store.decayed(runtime.hebbian_decay)
        if masked:
            increment = self._compile_masked_hebbian_factors(
                key,
                value,
                eta=runtime.eta,
                n_stages=self._n_stages,
                shape=self._shape,
                f_initial=self._f_initial,
            )
        else:
            increment = self._compile_hebbian_factors(key, value, eta=runtime.eta)

        updated = decayed.concatenated(increment)
        dense_matrix = updated.to_dense()
        clamped = self._write_system.clamp_memory(dense_matrix)
        if torch.equal(clamped, dense_matrix):
            return updated
        return FactorMemoryStore.from_dense(clamped)


# =================================================================================================
class FactorAppendStoreBackend:
    """Factor-store backend for append-only episodic writes."""

    supports_linear_view = True
    supports_factor_view = True

    def __init__(self, write_system: object, *, feature_dim: int, settings: FactorStoreSettings) -> None:
        """Bind factor allocation, append, and merge behavior to one backend."""
        self._write_system = write_system
        self._feature_dim = int(feature_dim)
        self._settings = settings

    def init_store(  # ---------------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> FactorMemoryStore:  # fmt: skip
        """Allocate an empty factor store using the configured store settings."""
        capacity = 0 if self._settings.memory_capacity is None else int(self._settings.memory_capacity)
        keys = torch.zeros((batch_size, capacity, self._feature_dim), dtype=torch.float, device=device)
        values = torch.zeros((batch_size, capacity, self._feature_dim), dtype=torch.float, device=device)
        valid_mask = torch.zeros((batch_size, capacity), dtype=torch.bool, device=device)
        coefficients = torch.zeros((batch_size, capacity), dtype=torch.float, device=device)
        banks = {
            name: FactorSlotBank(
                keys=keys.clone(),
                values=values.clone(),
                valid_mask=valid_mask.clone(),
                coefficients=coefficients.clone(),
            )
            for name in self._settings.bank_names
        }
        return FactorMemoryStore(keys=keys, values=values, valid_mask=valid_mask, coefficients=coefficients, banks=banks)

    def merge_rows(  # ---------------------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> FactorMemoryStore:  # fmt: skip
        """Merge append-only factor stores during partial reset."""
        if not isinstance(current, FactorMemoryStore) or not isinstance(fresh, FactorMemoryStore):
            raise TypeError("FactorAppendStoreBackend expected factor memory stores.")
        return current.merged_rows(flag, fresh)

    def append_write(  # -------------------------------------------------------------------------
        self, store: MemoryEntry, key: Tensor, value: Tensor, *, bank_name: Optional[str] = None,
    ) -> FactorMemoryStore:  # fmt: skip
        """Append a factor-memory atom to the provided memory entry."""
        if not isinstance(store, FactorMemoryStore):
            raise TypeError("FactorAppendStoreBackend expected factor memory stores.")
        return self._write_system.append(store, key, value, bank_name=bank_name, capacity=self._settings.memory_capacity)


# =================================================================================================
def build_hebbian_store_backend(  # ---------------------------------------------------------------
    write_system: nn.Module, *,
    store: LinearStoreSettings,
    feature_dim: int,
    update_mask: Tensor,
    n_stages: int,
    shape: list[int],
    f_initial: list[float],
    compile_hebbian_factors: Callable[..., FactorMemoryStore],
    compile_masked_hebbian_factors: Callable[..., FactorMemoryStore],
) -> HebbianStoreBackend:  # fmt: skip
    """Construct one behavior-bearing backend for Hebbian memory."""
    if store.representation == "factor":
        return FactorHebbianStoreBackend(
            write_system,
            feature_dim=feature_dim,
            n_stages=n_stages,
            shape=shape,
            f_initial=f_initial,
            compile_hebbian_factors=compile_hebbian_factors,
            compile_masked_hebbian_factors=compile_masked_hebbian_factors,
        )

    return DenseHebbianStoreBackend(write_system, feature_dim=feature_dim, update_mask=update_mask)


def build_factor_store_backend(  # ----------------------------------------------------------------
    write_system: object, *, store: FactorStoreSettings, feature_dim: int,
) -> AppendStoreBackend:  # fmt: skip
    """Construct one behavior-bearing backend for episodic factor memory."""
    return FactorAppendStoreBackend(write_system, feature_dim=feature_dim, settings=store)


def validate_attractor_store_backend(backend: StoreBackend) -> None:
    """Validate that one backend can support attractor retrieval."""
    if not backend.supports_linear_view:
        raise TypeError("Attractor modules require a store backend that exposes a linear-memory view.")


def validate_attention_store_backend(backend: StoreBackend) -> None:
    """Validate that one backend can support factor-memory retrieval."""
    if not backend.supports_factor_view:
        raise TypeError("Attention modules require a store backend that exposes factor-memory banks.")


# =================================================================================================
__all__ = [
    "AppendStoreBackend", "FactorAppendStoreBackend", "FactorStoreSettings",
    "DenseHebbianStoreBackend", "FactorHebbianStoreBackend", "HebbianStoreBackend",
    "LinearStoreSettings", "StoreBackend",
    "build_factor_store_backend", "build_hebbian_store_backend",
    "validate_attractor_store_backend", "validate_attention_store_backend",
]  # fmt: skip
