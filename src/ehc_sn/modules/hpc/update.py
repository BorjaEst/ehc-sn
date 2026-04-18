"""Write rules and learning-layout helpers for hippocampal memory modules.

This module implements the dense Hebbian update used by attractor memory, the
append-only episodic writer used by factor memory, and the shared layout logic
that keeps dense and factorized Hebbian updates consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.types import DEFAULT_FACTOR_BANK_NAME, FactorMemoryStore, FactorSlotBank


# =================================================================================================
class HebbianWriteSettings(BaseModel, extra="forbid"):
    """Static configuration for dense Hebbian-memory updates."""

    clamp_min: float = Field(
        default=-1.0,
        description="Minimum clamp value for Hebbian memory weights.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum clamp value for Hebbian memory weights.",
    )


# =================================================================================================
class EpisodicWriteSettings(BaseModel, extra="forbid"):
    """Static configuration for append-only factor-memory writes."""

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
    write_bank: str = Field(
        default=DEFAULT_FACTOR_BANK_NAME,
        description="Name of the bank updated by append writes.",
    )


# =================================================================================================
@dataclass
class HebbianWriteRuntime:
    """Mutable runtime hyperparameters for dense Hebbian memory."""

    eta: float = 0.5
    hebbian_decay: float = 0.9999


# =================================================================================================
HebbianBlockPair = tuple[int, int, slice, slice]
"""One allowed block pair in the hierarchical Hebbian write layout."""


# =================================================================================================
@dataclass(frozen=True)
class HebbianLayout:
    """Canonical masked Hebbian write layout shared by dense and factor stores.

    Attributes:
        dense_mask: Dense connectivity mask with shape ``(S, S)``.
        block_pairs: Allowed block-pair metadata used to compile masked factor
            increments without re-deriving the write policy.
        feature_dim: Flattened code width ``S = sum(shape)``.
    """

    dense_mask: Tensor
    block_pairs: tuple[HebbianBlockPair, ...]
    feature_dim: int

    def compile_factors(  # -----------------------------------------------------------------------
        self, p_inf: Tensor, p_gen: Tensor, *, eta: float, masked: bool,
    ) -> FactorMemoryStore:  # fmt: skip
        """Compile one Hebbian increment into factor-memory atoms.

        When ``masked`` is true, one atom is emitted per allowed block pair in
        the hierarchical write layout. Otherwise a single unmasked factor atom
        represents the whole update.
        """
        _validate_hebbian_codes(p_inf, p_gen, feature_dim=self.feature_dim)
        if not masked:
            return _compile_hebbian_factors_unmasked(p_inf, p_gen, eta=eta)

        batch_size = int(p_inf.shape[0])
        atom_count = len(self.block_pairs)
        a_t = p_inf + p_gen
        b_t = p_inf - p_gen

        keys = torch.zeros((batch_size, atom_count, self.feature_dim), dtype=a_t.dtype, device=a_t.device)
        values = torch.zeros((batch_size, atom_count, self.feature_dim), dtype=b_t.dtype, device=b_t.device)
        valid_mask = torch.ones((batch_size, atom_count), dtype=torch.bool, device=a_t.device)
        coefficients = torch.full((batch_size, atom_count), float(eta), dtype=a_t.dtype, device=a_t.device)

        for atom_index, (_, _, row_slice, col_slice) in enumerate(self.block_pairs):
            keys[:, atom_index, row_slice] = a_t[:, row_slice]
            values[:, atom_index, col_slice] = b_t[:, col_slice]

        return FactorMemoryStore(keys=keys, values=values, valid_mask=valid_mask, coefficients=coefficients)


# =================================================================================================
class HebbianWriteRule(Protocol):
    """Explicit collaborator contract required by Hebbian store backends."""

    @property
    def runtime(self) -> HebbianWriteRuntime:
        """Return mutable runtime hyperparameters for Hebbian updates."""

    def apply_dense(
        self, memory: Tensor, p_inf: Tensor | list[Tensor], p_gen: Tensor | list[Tensor], *,
        mask: Optional[Tensor] = None,
    ) -> Tensor:  # fmt: skip
        """Apply one dense Hebbian update step to a memory operator."""

    def clamp_memory(self, memory: Tensor) -> Tensor:
        """Clamp one dense memory tensor to the configured numeric range."""


# =================================================================================================
def _hebbian_allow_matrix(  # ---------------------------------------------------------------------
    n_stages: int, shape: list[int], f_initial: list[float],
) -> Tensor:  # fmt: skip
    """Return module-level Hebbian connectivity for the configured layout."""
    n_freq = len(shape)
    if len(f_initial) != n_freq:
        raise ValueError(f"Expected f_initial length {n_freq}, got {len(f_initial)}.")
    if not (0 <= int(n_stages) <= n_freq):
        raise ValueError(f"n_stages must be in [0, {n_freq}], got {n_stages}.")

    module = torch.arange(n_freq)
    constrained = module < int(n_stages)
    same_type = constrained[:, None] == constrained[None, :]
    frequencies = torch.as_tensor(f_initial, dtype=torch.float)
    low_to_high = frequencies[:, None] <= frequencies[None, :]
    return (~same_type) | low_to_high


# =================================================================================================
def _hebbian_offsets(  # --------------------------------------------------------------------------
    shape: list[int],
) -> list[int]:  # fmt: skip
    """Return cumulative feature offsets for a multi-frequency code shape."""
    offsets = [0]
    for width in shape:
        offsets.append(offsets[-1] + int(width))
    return offsets


# =================================================================================================
def build_hebbian_layout(  # ----------------------------------------------------------------------
    n_stages: int, shape: list[int], f_initial: list[float],
) -> HebbianLayout:  # fmt: skip
    """Return the canonical masked Hebbian write layout for one HPC configuration."""
    allow = _hebbian_allow_matrix(n_stages, shape, f_initial)
    offsets = _hebbian_offsets(shape)
    feature_dim = offsets[-1]

    block_pairs: list[HebbianBlockPair] = []
    for row_index in range(len(shape)):
        row_slice = slice(offsets[row_index], offsets[row_index + 1])
        for col_index in range(len(shape)):
            if bool(allow[row_index, col_index]):
                col_slice = slice(offsets[col_index], offsets[col_index + 1])
                block_pairs.append((row_index, col_index, row_slice, col_slice))

    widths = torch.as_tensor(shape, dtype=torch.long)
    module_ids = torch.arange(len(shape)).repeat_interleave(widths)
    dense_mask = allow[module_ids[:, None], module_ids[None, :]].to(dtype=torch.float)
    return HebbianLayout(dense_mask=dense_mask, block_pairs=tuple(block_pairs), feature_dim=feature_dim)


# =================================================================================================
def _validate_hebbian_codes(  # -------------------------------------------------------------------
    p_inf: Tensor, p_gen: Tensor, *, feature_dim: Optional[int] = None,
) -> None:  # fmt: skip
    """Validate one pair of flattened Hebbian codes."""
    if p_inf.ndim != 2 or p_gen.ndim != 2:
        raise ValueError("p_inf and p_gen must both be rank-2 `(B, S)` tensors.")
    if p_inf.shape != p_gen.shape:
        raise ValueError(f"p_inf and p_gen must share shape, got {tuple(p_inf.shape)} and {tuple(p_gen.shape)}.")
    if feature_dim is not None and int(p_inf.shape[1]) != feature_dim:
        raise ValueError(f"Expected flattened code width {feature_dim}, got {int(p_inf.shape[1])}.")


# =================================================================================================
def _compile_hebbian_factors_unmasked(  # ---------------------------------------------------------
    p_inf: Tensor, p_gen: Tensor, *, eta: float
) -> FactorMemoryStore:  # fmt: skip
    """Compile one unmasked Hebbian increment into a single factor-memory atom."""
    _validate_hebbian_codes(p_inf, p_gen)
    a_t = p_inf + p_gen
    b_t = p_inf - p_gen
    batch_size = int(p_inf.shape[0])
    coefficients = torch.full((batch_size, 1), float(eta), dtype=a_t.dtype, device=a_t.device)
    valid_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=a_t.device)
    return FactorMemoryStore(keys=a_t.unsqueeze(1), values=b_t.unsqueeze(1), valid_mask=valid_mask, coefficients=coefficients)


# =================================================================================================
class HebbianWrite(nn.Module):
    """Dense Hebbian write/update logic for grounded-location memory.

    The update uses the TEM-style outer-product form built from inferred and
    generative grounded-location codes.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: HebbianWriteSettings,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Initialize dense Hebbian write logic and mutable runtime parameters."""
        del device, dtype
        super().__init__()
        self._config = config
        self._runtime = HebbianWriteRuntime()

    @property
    def config(self) -> HebbianWriteSettings:
        """Return static Hebbian write settings."""
        return self._config

    @property
    def runtime(self) -> HebbianWriteRuntime:
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
        return self.apply_dense(memory, p_inf, p_gen, mask=mask)

    def apply_dense(  # ---------------------------------------------------------------------------
        self, memory: Tensor, p_inf: Tensor | list[Tensor], p_gen: Tensor | list[Tensor], *,
        mask: Optional[Tensor] = None,
    ) -> Tensor:  # fmt: skip
        """Apply one dense Hebbian update step to a memory operator."""
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

    def clamp_memory(  # --------------------------------------------------------------------------
        self, memory: Tensor,
    ) -> Tensor:  # fmt: skip
        """Clamp dense memory weights to the configured numeric range."""
        return torch.clamp(memory, min=self._config.clamp_min, max=self._config.clamp_max)


# =================================================================================================
class EpisodicWrite:
    """Append-only factor-store write policy with optional novelty gating."""

    def __init__(self, config: EpisodicWriteSettings) -> None:
        """Initialize append-only episodic write policy."""
        self._config = config

    @property
    def config(self) -> EpisodicWriteSettings:
        """Return static episodic-write settings."""
        return self._config

    def append(  # -------------------------------------------------------------------------------
        self, store: FactorMemoryStore, key: Tensor, value: Tensor, *,
        bank_name: Optional[str] = None, capacity: Optional[int] = None,
    ) -> FactorMemoryStore:  # fmt: skip
        """Append a factor-memory atom, optionally skipping non-novel rows.

        Args:
            store: Factor-memory store to update.
            key: Flattened key tensor with shape ``(B, S)``.
            value: Flattened value tensor with shape ``(B, S)``.
            bank_name: Optional explicit bank override.
            capacity: Optional maximum number of retained slots.

        Returns:
            A new ``FactorMemoryStore`` reflecting the requested append policy.
        """
        target_bank_name = self.config.write_bank if bank_name is None else bank_name
        current_bank = store.bank(target_bank_name, fallback_to_default=(target_bank_name == DEFAULT_FACTOR_BANK_NAME))

        key = key.to(dtype=current_bank.keys.dtype) if current_bank.capacity > 0 else key.to(dtype=torch.float)
        value = value.to(dtype=current_bank.values.dtype) if current_bank.capacity > 0 else value.to(dtype=torch.float)

        keep_row = torch.ones((key.shape[0],), dtype=torch.bool, device=key.device)
        if self.config.policy == "append_if_novel":
            keep_row = ~self._already_stored(current_bank, key, value)
            if not keep_row.any():
                return store

        keys = torch.cat((current_bank.keys, key.unsqueeze(1)), dim=1)
        values = torch.cat((current_bank.values, value.unsqueeze(1)), dim=1)
        coefficients = torch.cat((current_bank.coefficient_tensor(), keep_row.unsqueeze(1).to(dtype=value.dtype)), dim=1)
        valid_mask = torch.cat((current_bank.valid_mask, keep_row.unsqueeze(1)), dim=1)

        if capacity is not None:
            limit = int(capacity)
            keys = keys[:, -limit:, :]
            values = values[:, -limit:, :]
            coefficients = coefficients[:, -limit:]
            valid_mask = valid_mask[:, -limit:]

        updated_bank = FactorSlotBank(keys=keys, values=values, valid_mask=valid_mask, coefficients=coefficients)
        if target_bank_name == DEFAULT_FACTOR_BANK_NAME:
            return FactorMemoryStore(
                keys=updated_bank.keys,
                values=updated_bank.values,
                valid_mask=updated_bank.valid_mask,
                coefficients=updated_bank.coefficients,
                banks={name: bank.clone() for name, bank in store.banks.items()},
            )

        banks = {name: bank.clone() for name, bank in store.banks.items()}
        banks[target_bank_name] = updated_bank
        return FactorMemoryStore(
            keys=store.keys,
            values=store.values,
            valid_mask=store.valid_mask,
            coefficients=store.coefficients,
            banks=banks,
        )

    def _already_stored(  # ----------------------------------------------------------------------
        self, store: FactorSlotBank, key: Tensor, value: Tensor,
    ) -> Tensor:  # fmt: skip
        """Return which batch rows already contain a sufficiently similar atom.

        Similarity is computed over concatenated key-value vectors using the
        configured metric.
        """
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


# =================================================================================================
def compile_hebbian_factors(  # ------------------------------------------------------------------
    p_inf: Tensor, p_gen: Tensor, *, eta: float,
) -> FactorMemoryStore:  # fmt: skip
    """Compile one dense Hebbian update step into a single factor-memory atom."""
    return _compile_hebbian_factors_unmasked(p_inf, p_gen, eta=eta)


# =================================================================================================
def hebbian_block_pairs(  # ----------------------------------------------------------------------
    n_stages: int, shape: list[int], f_initial: list[float],
) -> list[tuple[int, int, slice, slice]]:  # fmt: skip
    """Return the block pairs allowed by the hierarchical Hebbian update mask."""
    return list(build_hebbian_layout(n_stages, shape, f_initial).block_pairs)


# =================================================================================================
def compile_masked_hebbian_factors(  # ------------------------------------------------------------
    p_inf: Tensor, p_gen: Tensor, *,
    eta: float, n_stages: int, shape: list[int], f_initial: list[float],
) -> FactorMemoryStore:  # fmt: skip
    """Compile a masked hierarchical Hebbian update into factor-memory atoms."""
    layout = build_hebbian_layout(n_stages, shape, f_initial)
    return layout.compile_factors(p_inf, p_gen, eta=eta, masked=True)


# =================================================================================================
__all__ = [
    "EpisodicWrite", "EpisodicWriteSettings",
    "HebbianBlockPair", "HebbianLayout", "HebbianWriteRule",
    "HebbianWrite", "HebbianWriteSettings", "HebbianWriteRuntime",
    "build_hebbian_layout", "compile_hebbian_factors", "compile_masked_hebbian_factors", "hebbian_block_pairs",
]  # fmt: skip
