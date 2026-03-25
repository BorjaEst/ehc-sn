from __future__ import annotations

"""Store representations and exact operator helpers for HPC memory backends."""

from dataclasses import dataclass
from typing import Optional, Protocol

import torch
from torch import Tensor

from ehc_sn import utils
from ehc_sn.types import DenseMemoryStore, Device, FactorMemoryStore, MemoryEntry


# =================================================================================================
class HebbianStoreApplier(Protocol):
    """Store-specific Hebbian write strategy used by attractor backends."""

    def apply(  # -------------------------------------------------------------------------------
        self, store: MemoryEntry, key: Tensor, value: Tensor, *, masked: bool,
    ) -> MemoryEntry:  # fmt: skip
        """Return the updated store after one Hebbian write step."""


# =================================================================================================
class AppendStoreApplier(Protocol):
    """Store-specific append-write strategy used by attention backends."""

    def apply(  # -------------------------------------------------------------------------------
        self, store: MemoryEntry, key: Tensor, value: Tensor,
    ) -> MemoryEntry:  # fmt: skip
        """Return the updated store after one append write step."""


# =================================================================================================
class MemoryStoreFactory(Protocol):
    """Constructor-time store factory used by backend init/reset paths."""

    def init_store(  # ---------------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> MemoryEntry:  # fmt: skip
        """Return an empty store for the configured representation."""


# =================================================================================================
class MemoryResetStrategy(Protocol):
    """Store-specific row merge strategy used during partial reset."""

    def merge_rows(  # ---------------------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> MemoryEntry:  # fmt: skip
        """Return the merged store after applying the partial-reset mask."""


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
        self._feature_dim = int(feature_dim)

    def init_store(  # ---------------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> DenseMemoryStore:  # fmt: skip
        return DenseMemoryStore(matrix=torch.zeros((batch_size, self._feature_dim, self._feature_dim), dtype=torch.float, device=device))


# =================================================================================================
class FactorMemoryStoreFactory:
    """Allocate empty exact factor stores."""

    def __init__(  # ------------------------------------------------------------------------------
        self, *, feature_dim: int,
    ) -> None:  # fmt: skip
        self._feature_dim = int(feature_dim)

    def init_store(  # ---------------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> FactorMemoryStore:  # fmt: skip
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
        if not isinstance(current, DenseMemoryStore) or not isinstance(fresh, DenseMemoryStore):
            raise TypeError("DenseMemoryResetStrategy expected dense memory stores.")
        return merge_dense_memory_rows(flag, current, fresh)


# =================================================================================================
class FactorMemoryResetStrategy:
    """Partial-reset row merge for exact factor stores."""

    def merge_rows(  # ---------------------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> FactorMemoryStore:  # fmt: skip
        if not isinstance(current, FactorMemoryStore) or not isinstance(fresh, FactorMemoryStore):
            raise TypeError("FactorMemoryResetStrategy expected factor memory stores.")
        return merge_factor_memory_rows(flag, current, fresh)


# =================================================================================================
# =================================================================================================
def merge_dense_memory_rows(  # ------------------------------------------------------------------
    flag: Tensor,
    current: DenseMemoryStore,
    fresh: DenseMemoryStore,
) -> DenseMemoryStore:  # fmt: skip
    """Replace flagged dense-memory rows with fresh rows during partial reset."""
    return DenseMemoryStore(matrix=utils.merge_rows(flag, current.matrix, fresh.matrix))


# =================================================================================================
def merge_factor_memory_rows(  # -----------------------------------------------------------------
    flag: Tensor,
    current: FactorMemoryStore,
    fresh: FactorMemoryStore,
) -> FactorMemoryStore:  # fmt: skip
    """Replace flagged factor-memory rows with fresh rows during partial reset.

    ``flag`` follows the shared partial-reset convention: flagged rows are taken
    from ``fresh`` and unflagged rows are kept from ``current``.
    """
    target = max(current.capacity, fresh.capacity)
    current = _pad_store(current, target)
    fresh = _pad_store(fresh, target)
    return utils.merge_tree_rows(flag, current, fresh)


merge_episodic_memory_rows = merge_factor_memory_rows
"""Backward-compatible alias for factor-memory row merging."""


# =================================================================================================
def _pad_store(  # --------------------------------------------------------------------------------
    store: FactorMemoryStore, target_capacity: int,
) -> FactorMemoryStore:  # fmt: skip
    """Pad a store with empty slots up to ``target_capacity``."""
    coefficients = store.coefficient_tensor()
    if store.capacity >= target_capacity:
        return FactorMemoryStore(
            keys=store.keys,
            values=store.values,
            valid_mask=store.valid_mask,
            coefficients=coefficients,
        )

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
    coefficient_padding = torch.zeros(
        (*coefficients.shape[:1], pad),
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    return FactorMemoryStore(
        keys=torch.cat((store.keys, key_padding), dim=1),
        values=torch.cat((store.values, value_padding), dim=1),
        valid_mask=torch.cat((store.valid_mask, mask_padding), dim=1),
        coefficients=torch.cat((coefficients, coefficient_padding), dim=1),
    )


# =================================================================================================
def factor_memory_to_dense(  # --------------------------------------------------------------------
    store: FactorMemoryStore,
) -> Tensor:  # fmt: skip
    """Materialize a factor store into the equivalent dense operator.

    The codebase uses row-vector retrieval (`h @ M`). Each factor atom therefore
    contributes `gamma * key^T value`, so that `(h @ M)` yields
    `sum_i gamma_i (h · key_i) value_i`.
    """
    coefficients = store.coefficient_tensor() * store.valid_mask.to(dtype=store.values.dtype)
    return torch.einsum("bt,btk,bts->bks", coefficients, store.keys, store.values)


def apply_factor_memory(  # -----------------------------------------------------------------------
    query: Tensor, store: FactorMemoryStore,
) -> Tensor:  # fmt: skip
    """Apply a factor-store operator to a flattened row-vector query.

    This is the linear, non-softmax operator induced by the factor atoms and is
    exact with respect to `factor_memory_to_dense(store)`.
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
    return FactorMemoryStore(
        keys=store.keys,
        values=store.values,
        valid_mask=store.valid_mask,
        coefficients=store.coefficient_tensor() * float(decay),
    )


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
    if current.keys.ndim != 3 or fresh.keys.ndim != 3:
        raise ValueError("factor stores must use rank-3 key tensors `(B, T, S)`.")
    if current.keys.shape[0] != fresh.keys.shape[0]:
        raise ValueError("factor stores must share batch size.")
    if current.keys.shape[2] != fresh.keys.shape[2]:
        raise ValueError("factor stores must share feature width.")

    keys = torch.cat((current.keys, fresh.keys.to(dtype=current.keys.dtype)), dim=1)
    values = torch.cat((current.values, fresh.values.to(dtype=current.values.dtype)), dim=1)
    valid_mask = torch.cat((current.valid_mask, fresh.valid_mask.to(dtype=current.valid_mask.dtype)), dim=1)
    coefficients = torch.cat(
        (current.coefficient_tensor(), fresh.coefficient_tensor().to(dtype=current.values.dtype)),
        dim=1,
    )

    if capacity is not None:
        limit = int(capacity)
        keys = keys[:, -limit:, :]
        values = values[:, -limit:, :]
        valid_mask = valid_mask[:, -limit:]
        coefficients = coefficients[:, -limit:]

    return FactorMemoryStore(keys=keys, values=values, valid_mask=valid_mask, coefficients=coefficients)


# =================================================================================================
__all__ = [
    "AppendStoreApplier",
    "AppendStoreComponents",
    "DenseMemoryStoreFactory",
    "DenseMemoryResetStrategy",
    "FactorMemoryStoreFactory",
    "FactorMemoryResetStrategy",
    "HebbianStoreApplier",
    "HebbianStoreComponents",
    "MemoryResetStrategy",
    "MemoryStoreFactory",
    "apply_factor_memory",
    "concat_factor_memory",
    "decay_factor_memory",
    "dense_memory_to_factor",
    "factor_memory_to_dense",
    "merge_dense_memory_rows",
    "merge_episodic_memory_rows",
    "merge_factor_memory_rows",
]
