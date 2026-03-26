"""Store representations and exact operator helpers for HPC memory modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch
from torch import Tensor

from ehc_sn import utils
from ehc_sn.types import DEFAULT_FACTOR_BANK_NAME, DenseMemoryStore, Device, FactorMemoryStore, FactorSlotBank, MemoryEntry


# =================================================================================================
class HebbianStoreApplier(Protocol):
    """Store-specific Hebbian write strategy used by attractor implementations."""

    def apply(  # -------------------------------------------------------------------------------
        self, store: MemoryEntry, key: Tensor, value: Tensor, *, masked: bool,
    ) -> MemoryEntry:  # fmt: skip
        """Return the updated store after one Hebbian write step."""


class AppendStoreApplier(Protocol):
    """Store-specific append-write strategy used by attention implementations."""

    def apply(  # -------------------------------------------------------------------------------
        self, store: MemoryEntry, key: Tensor, value: Tensor,
    ) -> MemoryEntry:  # fmt: skip
        """Return the updated store after one append write step."""


class MemoryStoreFactory(Protocol):
    """Constructor-time store factory used by init/reset paths."""

    def init_store(  # ---------------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> MemoryEntry:  # fmt: skip
        """Return an empty store for the configured representation."""


class MemoryResetStrategy(Protocol):
    """Store-specific row merge strategy used during partial reset."""

    def merge_rows(  # ---------------------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> MemoryEntry:  # fmt: skip
        """Return the merged store after applying the partial-reset mask."""


# =================================================================================================
@dataclass(frozen=True)
class HebbianStoreComponents:
    """Constructor-time bundle for store representation strategies."""

    store_factory: MemoryStoreFactory
    store_applier: HebbianStoreApplier
    reset_strategy: MemoryResetStrategy


@dataclass(frozen=True)
class AppendStoreComponents:
    """Constructor-time bundle for append-write store strategies."""

    store_factory: MemoryStoreFactory
    store_applier: AppendStoreApplier
    reset_strategy: MemoryResetStrategy


# =================================================================================================
class DenseMemoryStoreFactory:
    """Allocate empty dense Hebbian stores."""

    def __init__(  # ------------------------------------------------------------------------------
        self, *, feature_dim: int,
    ) -> None:  # fmt: skip
        """Initialize a dense-store factory for the given flattened feature width."""
        self._feature_dim = int(feature_dim)

    def init_store(  # ---------------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> DenseMemoryStore:  # fmt: skip
        """Allocate an empty dense memory matrix for each batch row."""
        matrix = torch.zeros(
            (batch_size, self._feature_dim, self._feature_dim),
            dtype=torch.float,
            device=device,
        )
        return DenseMemoryStore(matrix=matrix)


class FactorMemoryStoreFactory:
    """Allocate empty exact factor stores."""

    def __init__(  # ------------------------------------------------------------------------------
        self, *, feature_dim: int,
    ) -> None:  # fmt: skip
        """Initialize a factor-store factory for the given flattened feature width."""
        self._feature_dim = int(feature_dim)

    def init_store(  # ---------------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> FactorMemoryStore:  # fmt: skip
        """Allocate an empty factor store with zero populated atoms."""
        return FactorMemoryStore(
            keys=torch.zeros((batch_size, 0, self._feature_dim), dtype=torch.float, device=device),
            values=torch.zeros((batch_size, 0, self._feature_dim), dtype=torch.float, device=device),
            valid_mask=torch.zeros((batch_size, 0), dtype=torch.bool, device=device),
            coefficients=torch.zeros((batch_size, 0), dtype=torch.float, device=device),
        )


# =================================================================================================
class DenseMemoryResetStrategy:
    """Partial-reset row merge for dense Hebbian stores."""

    def merge_rows(  # ---------------------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> DenseMemoryStore:  # fmt: skip
        """Merge dense stores by replacing flagged rows with fresh rows."""
        if not isinstance(current, DenseMemoryStore) or not isinstance(fresh, DenseMemoryStore):
            raise TypeError("DenseMemoryResetStrategy expected dense memory stores.")
        return merge_dense_memory_rows(flag, current, fresh)


class FactorMemoryResetStrategy:
    """Partial-reset row merge for exact factor stores."""

    def merge_rows(  # ---------------------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> FactorMemoryStore:  # fmt: skip
        """Merge factor stores by replacing flagged rows with fresh rows."""
        if not isinstance(current, FactorMemoryStore) or not isinstance(fresh, FactorMemoryStore):
            raise TypeError("FactorMemoryResetStrategy expected factor memory stores.")
        return merge_factor_memory_rows(flag, current, fresh)


# =================================================================================================
def merge_dense_memory_rows(  # ------------------------------------------------------------------
    flag: Tensor, current: DenseMemoryStore, fresh: DenseMemoryStore,
) -> DenseMemoryStore:  # fmt: skip
    """Replace flagged dense-memory rows with fresh rows during partial reset."""
    return DenseMemoryStore(matrix=utils.merge_rows(flag, current.matrix, fresh.matrix))


def merge_factor_memory_rows(  # -----------------------------------------------------------------
    flag: Tensor, current: FactorMemoryStore, fresh: FactorMemoryStore,
) -> FactorMemoryStore:  # fmt: skip
    """Replace flagged factor-memory rows with fresh rows during partial reset.

    ``flag`` follows the shared partial-reset convention: flagged rows are taken
    from ``fresh`` and unflagged rows are kept from ``current``.
    """
    capacities = _bank_capacities(current, fresh)
    current = _pad_store(current, capacities)
    fresh = _pad_store(fresh, capacities)
    return utils.merge_tree_rows(flag, current, fresh)


merge_episodic_memory_rows = merge_factor_memory_rows
"""Backward-compatible alias for factor-memory row merging."""


# =================================================================================================
def _pad_store(  # --------------------------------------------------------------------------------
    store: FactorMemoryStore, target_capacity: dict[str, int],
) -> FactorMemoryStore:  # fmt: skip
    """Pad each bank in a store up to the requested capacities."""
    current_banks = _store_banks(store)
    padded = {name: _pad_bank(_get_bank(current_banks, name, store=store), target_capacity[name]) for name in target_capacity}
    return _store_from_banks(padded)


# =================================================================================================
def factor_memory_to_dense(  # --------------------------------------------------------------------
    store: FactorMemoryStore,
) -> Tensor:  # fmt: skip
    """Materialize a factor store into the equivalent dense operator.

    The codebase uses row-vector retrieval (`h @ M`). Each factor atom therefore
    contributes `gamma * key^T value`, so that `(h @ M)` yields
    `sum_i gamma_i (h · key_i) value_i`.
    """
    dense: Tensor | None = None
    for bank in _store_banks(store).values():
        coefficients = bank.coefficient_tensor() * bank.valid_mask.to(dtype=bank.values.dtype)
        contribution = torch.einsum("bt,btk,bts->bks", coefficients, bank.keys, bank.values)
        dense = contribution if dense is None else dense + contribution
    if dense is None:
        raise ValueError("Factor memory store must contain at least one bank.")
    return dense


def apply_factor_memory(  # -----------------------------------------------------------------------
    query: Tensor, store: FactorMemoryStore,
) -> Tensor:  # fmt: skip
    """Apply a factor-store operator to a flattened row-vector query.

    This is the linear, non-softmax operator induced by the factor atoms and is
    exact with respect to ``factor_memory_to_dense(store)``.
    """
    return store.apply(query)


def dense_memory_to_factor(  # --------------------------------------------------------------------
    matrix: Tensor,
) -> FactorMemoryStore:  # fmt: skip
    """Return an exact factor-store representation of a dense operator.

    The factorization uses the standard basis for the key vectors, which yields
    one atom per source row and reconstructs the dense operator exactly under
    the row-vector convention used by the codebase.
    """
    if matrix.ndim != 3:
        raise ValueError(f"matrix must be rank-3 `(B, S, S)`, got shape {tuple(matrix.shape)}.")
    batch_size, rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"matrix must be square over feature dimension, got shape {tuple(matrix.shape)}.")

    basis = torch.eye(rows, dtype=matrix.dtype, device=matrix.device).unsqueeze(0).expand(batch_size, -1, -1)
    valid_mask = torch.ones((batch_size, rows), dtype=torch.bool, device=matrix.device)
    coefficients = torch.ones((batch_size, rows), dtype=matrix.dtype, device=matrix.device)
    return FactorMemoryStore(keys=basis, values=matrix, valid_mask=valid_mask, coefficients=coefficients)


def decay_factor_memory(  # -----------------------------------------------------------------------
    store: FactorMemoryStore, decay: float,
) -> FactorMemoryStore:  # fmt: skip
    """Scale factor coefficients to apply dense-memory decay exactly."""
    scaled = {
        name: FactorSlotBank(
            keys=bank.keys,
            values=bank.values,
            valid_mask=bank.valid_mask,
            coefficients=bank.coefficient_tensor() * float(decay),
        )
        for name, bank in _store_banks(store).items()
    }
    return _store_from_banks(scaled)


def concat_factor_memory(  # ----------------------------------------------------------------------
    current: FactorMemoryStore,
    fresh: FactorMemoryStore,
    *,
    capacity: Optional[int] = None,
) -> FactorMemoryStore:  # fmt: skip
    """Concatenate two factor stores along the atom axis.

    Both stores must share batch size and feature width. When ``capacity`` is
    provided, the oldest atoms are truncated from the left.
    """
    merged: dict[str, FactorSlotBank] = {}
    bank_names = sorted(set(_store_banks(current)) | set(_store_banks(fresh)))
    for name in bank_names:
        current_bank = _get_bank(_store_banks(current), name, store=current)
        fresh_bank = _get_bank(_store_banks(fresh), name, store=fresh)
        _validate_bank_pair(current_bank, fresh_bank)

        keys = torch.cat((current_bank.keys, fresh_bank.keys.to(dtype=current_bank.keys.dtype)), dim=1)
        values = torch.cat((current_bank.values, fresh_bank.values.to(dtype=current_bank.values.dtype)), dim=1)
        valid_mask = torch.cat((current_bank.valid_mask, fresh_bank.valid_mask.to(dtype=current_bank.valid_mask.dtype)), dim=1)
        coefficients = torch.cat(
            (current_bank.coefficient_tensor(), fresh_bank.coefficient_tensor().to(dtype=current_bank.values.dtype)),
            dim=1,
        )

        if capacity is not None:
            limit = int(capacity)
            keys = keys[:, -limit:, :]
            values = values[:, -limit:, :]
            valid_mask = valid_mask[:, -limit:]
            coefficients = coefficients[:, -limit:]

        merged[name] = FactorSlotBank(keys=keys, values=values, valid_mask=valid_mask, coefficients=coefficients)

    return _store_from_banks(merged)


def _store_banks(store: FactorMemoryStore) -> dict[str, FactorSlotBank]:
    """Return all factor banks, including the legacy default bank."""
    return {DEFAULT_FACTOR_BANK_NAME: store.default_bank(), **store.banks}


def _store_from_banks(banks: dict[str, FactorSlotBank]) -> FactorMemoryStore:
    """Rebuild a factor-memory store from a bank mapping."""
    default_bank = banks[DEFAULT_FACTOR_BANK_NAME]
    named_banks = {name: bank for name, bank in banks.items() if name != DEFAULT_FACTOR_BANK_NAME}
    return FactorMemoryStore(
        keys=default_bank.keys,
        values=default_bank.values,
        valid_mask=default_bank.valid_mask,
        coefficients=default_bank.coefficients,
        banks=named_banks,
    )


def _bank_capacities(*stores: FactorMemoryStore) -> dict[str, int]:
    """Return the maximum capacity required for each bank across stores."""
    capacities: dict[str, int] = {}
    for store in stores:
        for name, bank in _store_banks(store).items():
            capacities[name] = max(capacities.get(name, 0), bank.capacity)
    return capacities


def _get_bank(banks: dict[str, FactorSlotBank], name: str, *, store: FactorMemoryStore) -> FactorSlotBank:
    """Return a bank or an empty compatible bank when it is absent."""
    bank = banks.get(name)
    if bank is not None:
        return bank
    reference = store.default_bank()
    return FactorSlotBank(
        keys=torch.zeros((reference.keys.shape[0], 0, reference.keys.shape[2]), dtype=reference.keys.dtype, device=reference.keys.device),
        values=torch.zeros(
            (reference.values.shape[0], 0, reference.values.shape[2]), dtype=reference.values.dtype, device=reference.values.device
        ),
        valid_mask=torch.zeros((reference.valid_mask.shape[0], 0), dtype=reference.valid_mask.dtype, device=reference.valid_mask.device),
        coefficients=torch.zeros(
            (reference.valid_mask.shape[0], 0), dtype=reference.coefficient_tensor().dtype, device=reference.valid_mask.device
        ),
    )


def _pad_bank(bank: FactorSlotBank, target_capacity: int) -> FactorSlotBank:
    """Pad one bank with empty slots up to ``target_capacity``."""
    coefficients = bank.coefficient_tensor()
    if bank.capacity >= target_capacity:
        return FactorSlotBank(
            keys=bank.keys,
            values=bank.values,
            valid_mask=bank.valid_mask,
            coefficients=coefficients,
        )

    pad = target_capacity - bank.capacity
    key_padding = torch.zeros(
        (*bank.keys.shape[:1], pad, bank.keys.shape[2]),
        dtype=bank.keys.dtype,
        device=bank.keys.device,
    )
    value_padding = torch.zeros(
        (*bank.values.shape[:1], pad, bank.values.shape[2]),
        dtype=bank.values.dtype,
        device=bank.values.device,
    )
    mask_padding = torch.zeros(
        (*bank.valid_mask.shape[:1], pad),
        dtype=bank.valid_mask.dtype,
        device=bank.valid_mask.device,
    )
    coefficient_padding = torch.zeros(
        (*coefficients.shape[:1], pad),
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    return FactorSlotBank(
        keys=torch.cat((bank.keys, key_padding), dim=1),
        values=torch.cat((bank.values, value_padding), dim=1),
        valid_mask=torch.cat((bank.valid_mask, mask_padding), dim=1),
        coefficients=torch.cat((coefficients, coefficient_padding), dim=1),
    )


def _validate_bank_pair(current: FactorSlotBank, fresh: FactorSlotBank) -> None:
    """Validate that two banks can be concatenated safely."""
    if current.keys.ndim != 3 or fresh.keys.ndim != 3:
        raise ValueError("factor stores must use rank-3 key tensors `(B, T, S)`.")
    if current.keys.shape[0] != fresh.keys.shape[0]:
        raise ValueError("factor stores must share batch size.")
    if current.keys.shape[2] != fresh.keys.shape[2]:
        raise ValueError("factor stores must share feature width.")


# =================================================================================================
__all__ = [
    "AppendStoreApplier", "HebbianStoreApplier", "MemoryStoreFactory", "MemoryResetStrategy",
    "AppendStoreComponents", "HebbianStoreComponents",
    "DenseMemoryStoreFactory", "FactorMemoryStoreFactory",
    "DenseMemoryResetStrategy", "FactorMemoryResetStrategy",
    "factor_memory_to_dense", "dense_memory_to_factor",
    "apply_factor_memory", "concat_factor_memory", "decay_factor_memory",
    "merge_dense_memory_rows", "merge_episodic_memory_rows", "merge_factor_memory_rows",
]  # fmt: skip
