"""Write systems and store-application strategies for hippocampal memory modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.modules.hpc import memory
from ehc_sn.modules.hpc.memory import (
    AppendStoreComponents,
    DenseMemoryResetStrategy,
    DenseMemoryStoreFactory,
    FactorMemoryResetStrategy,
    FactorMemoryStoreFactory,
    HebbianStoreComponents,
)
from ehc_sn.types import DenseMemoryStore, Device, Dtype, FactorMemoryStore, MemoryEntry


# =================================================================================================
class HebbianMemoryWriteSettings(BaseModel, extra="forbid"):
    """Settings for dense Hebbian-memory write modules."""

    emit_store: Literal["dense", "factor"] = Field(
        default="dense",
        description="Internal storage form used after each Hebbian write update.",
    )
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
    """Settings for append-only factor-memory writes."""

    memory_capacity: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of factor slots retained; unset keeps all stored steps.",
    )
    policy: Literal["append_all", "append_if_novel"] = Field(
        default="append_if_novel",
        description="Write policy applied to factor-memory insertion.",
    )
    novelty_similarity: Literal["cosine", "dot"] = Field(
        default="cosine",
        description="Similarity metric used by novelty-gated factor writes.",
    )
    novelty_threshold: float = Field(
        default=0.95,
        description="Similarity threshold above which a candidate is treated as already stored.",
    )


# =================================================================================================
@dataclass
class HebbianMemoryRuntime:
    """Runtime hyperparameters for dense Hebbian memory."""

    eta: float = 0.5
    hebbian_decay: float = 0.9999


class HebbianMemoryWrite(nn.Module):
    """Hebbian write/update logic for grounded-location memory."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: HebbianMemoryWriteSettings,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Initialize dense Hebbian write logic and mutable runtime parameters."""
        del device, dtype
        super().__init__()
        self._config = config
        self._runtime = HebbianMemoryRuntime()

    @property
    def config(self) -> HebbianMemoryWriteSettings:
        """Return static Hebbian write settings."""
        return self._config

    @property
    def runtime(self) -> HebbianMemoryRuntime:
        """Return mutable runtime hyperparameters for Hebbian updates."""
        return self._runtime

    def _normalize_code(  # ----------------------------------------------------------------------
        self, code: Tensor | list[Tensor], *, name: str,
    ) -> Tensor:  # fmt: skip
        """Normalize flattened or multi-scale codes into a rank-2 tensor."""
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
        """Apply one Hebbian update step to a dense memory operator."""
        if memory.ndim != 3:
            raise ValueError(f"memory must be rank-3 `(B, S, S)`, got shape {tuple(memory.shape)}.")

        eta, hebbian_decay = self.runtime.eta, self.runtime.hebbian_decay
        p_inf = self._normalize_code(p_inf, name="p_inf")
        p_gen = self._normalize_code(p_gen, name="p_gen")

        batch_size, feature_dim = int(memory.shape[0]), int(memory.shape[1])
        if int(memory.shape[2]) != feature_dim:
            raise ValueError(f"memory must be square over feature dimension, got shape {tuple(memory.shape)}.")
        if int(p_inf.shape[0]) != batch_size or int(p_gen.shape[0]) != batch_size:
            raise ValueError("memory, p_inf, and p_gen must share the same batch size.")
        if int(p_inf.shape[1]) != feature_dim or int(p_gen.shape[1]) != feature_dim:
            raise ValueError(f"p_inf and p_gen must have width {feature_dim} to match memory, got {int(p_inf.shape[1])} and {int(p_gen.shape[1])}.")  # fmt: skip

        update = (p_inf + p_gen).unsqueeze(2) @ (p_inf - p_gen).unsqueeze(1)
        if mask is not None:
            update = update * mask.to(device=memory.device, dtype=memory.dtype)
        return self.clamp_memory(hebbian_decay * memory + eta * update)

    def clamp_memory(self, memory: Tensor) -> Tensor:
        """Clamp dense memory weights to the configured numeric range."""
        return torch.clamp(memory, min=self._config.clamp_min, max=self._config.clamp_max)


class DenseHebbianStoreApplier(nn.Module):
    """Apply Hebbian writes directly to dense memory stores."""

    def __init__(self, write_system: HebbianMemoryWrite, *, update_mask: Tensor) -> None:
        """Bind dense Hebbian updates to a shared write system and optional mask."""
        super().__init__()
        self._write_system = write_system
        self.register_buffer("update_mask", update_mask, persistent=False)

    def apply(  # -------------------------------------------------------------------------------
        self, store: MemoryEntry, key: Tensor, value: Tensor, *, masked: bool,
    ) -> DenseMemoryStore:  # fmt: skip
        """Apply a dense Hebbian update to the provided memory entry."""
        if not isinstance(store, DenseMemoryStore):
            raise TypeError("DenseHebbianStoreApplier expected dense memory stores.")
        mask = self.update_mask if masked else None
        return DenseMemoryStore(matrix=self._write_system(store.matrix, key, value, mask=mask))


# =================================================================================================
class FactorHebbianStoreApplier:
    """Apply Hebbian writes while storing the result as exact factor atoms."""

    def __init__(  # ------------------------------------------------------------------------------
        self, write_system: HebbianMemoryWrite, *,
        n_stages: int, shape: list[int], f_initial: list[float],
    ) -> None:  # fmt: skip
        """Bind factorized Hebbian updates to the configured hierarchy metadata."""
        self._write_system = write_system
        self._n_stages = int(n_stages)
        self._shape = list(shape)
        self._f_initial = list(f_initial)

    def apply(  # -------------------------------------------------------------------------------
        self, store: MemoryEntry, key: Tensor, value: Tensor, *, masked: bool,
    ) -> FactorMemoryStore:  # fmt: skip
        """Apply a Hebbian update and keep the resulting store in factor form."""
        if not isinstance(store, FactorMemoryStore):
            raise TypeError("FactorHebbianStoreApplier expected factor memory stores.")

        runtime = self._write_system.runtime
        decayed = memory.decay_factor_memory(store, runtime.hebbian_decay)
        if masked:
            increment = compile_masked_hebbian_factors(key, value, eta=runtime.eta, n_stages=self._n_stages, shape=self._shape, f_initial=self._f_initial)  # fmt: skip
        else:
            increment = compile_hebbian_factors(key, value, eta=runtime.eta)

        updated = memory.concat_factor_memory(decayed, increment)
        dense_matrix = memory.factor_memory_to_dense(updated)
        clamped = self._write_system.clamp_memory(dense_matrix)
        if torch.equal(clamped, dense_matrix):
            return updated
        return memory.dense_memory_to_factor(clamped)


# =================================================================================================
def build_hebbian_store_components(  # ------------------------------------------------------------
    write_system: HebbianMemoryWrite, *,
    emit_store: Literal["dense", "factor"], feature_dim: int, update_mask: Tensor,
    n_stages: int, shape: list[int], f_initial: list[float],
) -> HebbianStoreComponents:  # fmt: skip
    """Construct store factory, applier, and reset strategy for Hebbian memory."""
    if emit_store == "factor":
        return HebbianStoreComponents(
            store_factory=FactorMemoryStoreFactory(feature_dim=feature_dim),
            store_applier=FactorHebbianStoreApplier(write_system, n_stages=n_stages, shape=shape, f_initial=f_initial),
            reset_strategy=FactorMemoryResetStrategy(),
        )

    return HebbianStoreComponents(
        store_factory=DenseMemoryStoreFactory(feature_dim=feature_dim),
        store_applier=DenseHebbianStoreApplier(write_system, update_mask=update_mask),
        reset_strategy=DenseMemoryResetStrategy(),
    )


# =================================================================================================
class EpisodicMemoryWrite:
    """Append-only factor-store allocation and write helpers."""

    def __init__(self, shape: list[int], config: EpisodicMemoryWriteSettings) -> None:
        """Initialize append-only episodic memory with the configured capacity policy."""
        self._shape = list(shape)
        self._config = config

    @property
    def config(self) -> EpisodicMemoryWriteSettings:
        """Return static episodic-write settings."""
        return self._config

    @property
    def memory_capacity(self) -> Optional[int]:
        """Return the maximum retained atom count, if one is configured."""
        return self.config.memory_capacity

    def init_store(  # ---------------------------------------------------------------------------
        self, batch_size: int, *,
        device: Optional[Device] = None,
    ) -> FactorMemoryStore:  # fmt: skip
        """Allocate an empty episodic factor store for the given batch size."""
        feature_dim = sum(self._shape)
        capacity = 0 if self.memory_capacity is None else int(self.memory_capacity)
        keys = torch.zeros((batch_size, capacity, feature_dim), dtype=torch.float, device=device)
        values = torch.zeros((batch_size, capacity, feature_dim), dtype=torch.float, device=device)
        valid_mask = torch.zeros((batch_size, capacity), dtype=torch.bool, device=device)
        coefficients = torch.zeros((batch_size, capacity), dtype=torch.float, device=device)
        return FactorMemoryStore(keys=keys, values=values, valid_mask=valid_mask, coefficients=coefficients)

    def append(  # -------------------------------------------------------------------------------
        self, store: FactorMemoryStore, key: Tensor, value: Tensor,
    ) -> FactorMemoryStore:  # fmt: skip
        """Append a factor-memory atom, optionally skipping rows deemed non-novel."""
        key = key.to(dtype=store.keys.dtype) if store.capacity > 0 else key.to(dtype=torch.float)
        value = value.to(dtype=store.values.dtype) if store.capacity > 0 else value.to(dtype=torch.float)

        keep_row = torch.ones((key.shape[0],), dtype=torch.bool, device=key.device)
        if self.config.policy == "append_if_novel":
            keep_row = ~self._already_stored(store, key, value)
            if not keep_row.any():
                return store

        keys = torch.cat((store.keys, key.unsqueeze(1)), dim=1)
        values = torch.cat((store.values, value.unsqueeze(1)), dim=1)
        coefficients = torch.cat((store.coefficient_tensor(), keep_row.unsqueeze(1).to(dtype=value.dtype)), dim=1)
        valid_mask = torch.cat((store.valid_mask, keep_row.unsqueeze(1)), dim=1)

        if self.memory_capacity is not None:
            capacity = int(self.memory_capacity)
            keys = keys[:, -capacity:, :]
            values = values[:, -capacity:, :]
            coefficients = coefficients[:, -capacity:]
            valid_mask = valid_mask[:, -capacity:]

        return FactorMemoryStore(keys=keys, values=values, valid_mask=valid_mask, coefficients=coefficients)

    def _already_stored(  # ----------------------------------------------------------------------
        self, store: FactorMemoryStore, key: Tensor, value: Tensor,
    ) -> Tensor:  # fmt: skip
        """Return which batch rows already contain a sufficiently similar atom."""
        if store.capacity == 0:
            return torch.zeros((key.shape[0],), dtype=torch.bool, device=key.device)

        combined_query = torch.cat((key, value), dim=1)
        combined_store = torch.cat((store.keys, store.values), dim=2)
        if self.config.novelty_similarity == "cosine":
            combined_query = torch.nn.functional.normalize(combined_query, dim=1)
            combined_store = torch.nn.functional.normalize(combined_store, dim=2)

        similarity = torch.einsum("bd,btd->bt", combined_query, combined_store)
        invalid_fill = torch.full_like(similarity, torch.finfo(similarity.dtype).min)
        similarity = torch.where(store.valid_mask, similarity, invalid_fill)
        return similarity.max(dim=1).values >= self.config.novelty_threshold


class FactorAppendStoreFactory:
    """Allocate empty factor stores through the configured append-write system."""

    def __init__(self, write_system: EpisodicMemoryWrite) -> None:
        """Bind the append-store factory to an episodic write system."""
        self._write_system = write_system

    def init_store(  # ---------------------------------------------------------------------------
        self, batch_size: int, *,
        device: Optional[Device] = None,
    ) -> FactorMemoryStore:  # fmt: skip
        """Allocate an empty factor store using the bound episodic write system."""
        return self._write_system.init_store(batch_size, device=device)


class FactorAppendStoreApplier:
    """Apply append writes to factor memory stores."""

    def __init__(self, write_system: EpisodicMemoryWrite) -> None:
        """Bind append-store updates to an episodic write system."""
        self._write_system = write_system

    def apply(  # -------------------------------------------------------------------------------
        self, store: MemoryEntry, key: Tensor, value: Tensor,
    ) -> FactorMemoryStore:  # fmt: skip
        """Append a factor-memory atom to the provided memory entry."""
        if not isinstance(store, FactorMemoryStore):
            raise TypeError("FactorAppendStoreApplier expected factor memory stores.")
        return self._write_system.append(store, key, value)


# =================================================================================================
def build_factor_store_components(  # -------------------------------------------------------------
    write_system: EpisodicMemoryWrite,
) -> AppendStoreComponents:  # fmt: skip
    """Construct store factory, applier, and reset strategy for episodic memory."""
    return AppendStoreComponents(
        store_factory=FactorAppendStoreFactory(write_system),
        store_applier=FactorAppendStoreApplier(write_system),
        reset_strategy=FactorMemoryResetStrategy(),
    )


# =================================================================================================
def compile_hebbian_factors(  # ------------------------------------------------------------------
    p_inf: Tensor, p_gen: Tensor, *, eta: float,
) -> FactorMemoryStore:  # fmt: skip
    """Compile one dense Hebbian update step into a single factor-memory atom."""
    if p_inf.ndim != 2 or p_gen.ndim != 2:
        raise ValueError("p_inf and p_gen must both be rank-2 `(B, S)` tensors.")
    if p_inf.shape != p_gen.shape:
        raise ValueError(f"p_inf and p_gen must share shape, got {tuple(p_inf.shape)} and {tuple(p_gen.shape)}.")

    a_t = p_inf + p_gen
    b_t = p_inf - p_gen
    batch_size = int(p_inf.shape[0])
    coefficients = torch.full((batch_size, 1), float(eta), dtype=a_t.dtype, device=a_t.device)
    valid_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=a_t.device)
    return FactorMemoryStore(keys=a_t.unsqueeze(1), values=b_t.unsqueeze(1), valid_mask=valid_mask, coefficients=coefficients)


# =================================================================================================
def hebbian_block_pairs(  # ----------------------------------------------------------------------
    n_stages: int, shape: list[int], f_initial: list[float],
) -> list[tuple[int, int, slice, slice]]:  # fmt: skip
    """Return the block pairs allowed by the hierarchical Hebbian update mask."""

    n_freq = len(shape)
    if len(f_initial) != n_freq:
        raise ValueError(f"Expected f_initial length {n_freq}, got {len(f_initial)}.")

    offsets = [0]
    for width in shape:
        offsets.append(offsets[-1] + int(width))

    def block_slice(index: int) -> slice:
        return slice(offsets[index], offsets[index + 1])

    block_pairs: list[tuple[int, int, slice, slice]] = []
    for f_from in range(n_freq):
        constrained_from = f_from < int(n_stages)
        for f_to in range(n_freq):
            constrained_to = f_to < int(n_stages)
            same_type = constrained_from == constrained_to
            allow = (not same_type) or (float(f_initial[f_from]) <= float(f_initial[f_to]))
            if allow:
                block_pairs.append((f_from, f_to, block_slice(f_from), block_slice(f_to)))
    return block_pairs


# =================================================================================================
def compile_masked_hebbian_factors(  # ------------------------------------------------------------
    p_inf: Tensor, p_gen: Tensor, *,
    eta: float, n_stages: int, shape: list[int], f_initial: list[float],
) -> FactorMemoryStore:  # fmt: skip
    """Compile a masked hierarchical Hebbian update into factor-memory atoms."""
    if p_inf.ndim != 2 or p_gen.ndim != 2:
        raise ValueError("p_inf and p_gen must both be rank-2 `(B, S)` tensors.")
    if p_inf.shape != p_gen.shape:
        raise ValueError(f"p_inf and p_gen must share shape, got {tuple(p_inf.shape)} and {tuple(p_gen.shape)}.")

    feature_dim = sum(shape)
    if int(p_inf.shape[1]) != feature_dim:
        raise ValueError(f"Expected flattened code width {feature_dim}, got {int(p_inf.shape[1])}.")

    block_pairs = hebbian_block_pairs(n_stages, shape, f_initial)
    batch_size = int(p_inf.shape[0])
    atom_count = len(block_pairs)
    a_t = p_inf + p_gen
    b_t = p_inf - p_gen

    keys = torch.zeros((batch_size, atom_count, feature_dim), dtype=a_t.dtype, device=a_t.device)
    values = torch.zeros((batch_size, atom_count, feature_dim), dtype=b_t.dtype, device=b_t.device)
    valid_mask = torch.ones((batch_size, atom_count), dtype=torch.bool, device=a_t.device)
    coefficients = torch.full((batch_size, atom_count), float(eta), dtype=a_t.dtype, device=a_t.device)

    for atom_index, (_, _, row_slice, col_slice) in enumerate(block_pairs):
        keys[:, atom_index, row_slice] = a_t[:, row_slice]
        values[:, atom_index, col_slice] = b_t[:, col_slice]

    return FactorMemoryStore(keys=keys, values=values, valid_mask=valid_mask, coefficients=coefficients)


# =================================================================================================
__all__ = [
    "DenseHebbianStoreApplier", "FactorAppendStoreApplier", "FactorAppendStoreFactory",
    "FactorHebbianStoreApplier",
    "EpisodicMemoryWrite", "EpisodicMemoryWriteSettings",
    "HebbianMemoryWrite", "HebbianMemoryWriteSettings", "HebbianMemoryRuntime",
    "build_factor_store_components", "build_hebbian_store_components",
    "compile_hebbian_factors", "compile_masked_hebbian_factors", "hebbian_block_pairs",
]  # fmt: skip
