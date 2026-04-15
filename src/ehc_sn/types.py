"""Core type definitions used across TEM.

This module is intentionally dependency-light (no imports from other
`ehc_sn` modules) and provides shared aliases and small dataclasses used
throughout the codebase.

Conventions:
    - Multi-scale codes are `list[Tensor]` (one tensor per frequency module).
    - Many modules use `B` for batch size and `S = sum(shape)` for flattened
      multi-scale size.

Longer background notes live in `docs/foundations.md`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Optional, Sequence, TypeAlias

import numpy as np
import torch
from torch import Tensor

# =============================================================================
# Mathematical Primitives
# =============================================================================

Vector = Tensor
"""A 1D or batched 2D tensor representing a neural population code.

Shape conventions:
    - Unbatched: (n_cells,)
    - Batched: (batch_size, n_cells)
"""

Matrix = Tensor
"""A 2D tensor representing connection weights or transformations.

Shape conventions:
    - Hebbian weights: (n_cells_pre, n_cells_post)
    - Projection matrix: (input_dim, output_dim)
"""


# =============================================================================
# Multi-Scale Representations
# =============================================================================

MultiScaleCode = list[Vector]
"""Hierarchical representation across multiple frequency modules.

In TEM, both abstract locations (g) and grounded locations (p) are
represented as multi-scale codes, where each frequency module operates
at a different spatial scale.

Structure:
    - Length: `n_freq` (one element per frequency module)
    - Each element: tensor of shape `(B, n_cells_f)`
"""

AbstractLocation = MultiScaleCode
"""Abstract spatial representation (g) from transition dynamics.

The abstract location encodes position in a factorized, multi-scale
representation. It is updated through path integration and refined
through sensory inference.

Notes:
    This is the abstract (grid-like) code used by MEC dynamics.
"""


GroundedLocation = MultiScaleCode
"""Grounded place cell representation (p) from memory retrieval.

The grounded location is retrieved from Hebbian memory using the abstract
location as a query. It represents discrete place cell activations that
are tied to specific environmental features.

Notes:
    This is the grounded (place-like) code used by HPC memory.
"""


Location: TypeAlias = GroundedLocation | AbstractLocation
"""Generic location code, either grounded (p) or abstract (g)."""


@dataclass(frozen=True)
class LocationBelief:
    """Probabilistic abstract location estimate with uncertainty quantification.

    Represents a Gaussian estimate of abstract location across multiple
    frequency modules. Each frequency has independent mean and uncertainty.

    Attributes:
        mean: Predicted abstract location per frequency [List of (B, n_g[f])]
        uncertainty: Prediction uncertainty (sigma) per frequency [List of (B, n_g[f])]

    Notes:
        The uncertainty is commonly used for inverse-variance fusion.
    """

    mean: Location
    uncertainty: Optional[MultiScaleCode]


Observation = Tensor
"""Ground-truth sensory observation (o) from the environment.

The sensory observation represents the actual sensory input received
at a given timestep, distinct from model-generated sensory predictions.
Observations are provided as single tensors (typically one-hot or encoded)
and are processed internally into multi-scale representations.

Shape:
    - Batched: `(B, n_observations)`

Notes:
    Observations are typically one-hot or encoded features.
"""

Action = Optional[int]
"""Ground-truth action signal (a) associated with a timestep.

The action represents the agent's motor command (or discrete action index)
that drives state transitions in the environment and therefore informs TEM's
transition/path-integration dynamics.

Shape:
    - 

Notes:
    Actions drive the transition/path-integration dynamics.
"""

LocationLabel = Tensor
"""Ground-truth environment location label for supervision and evaluation.

This represents the environment-provided notion of "true" location (e.g., a
grid cell index in a discrete maze). It is primarily used for auxiliary
supervision, diagnostics, and plotting, and is not required for TEM's core
generative/inference loops.

Shape:
    - Time-major batched (common in DataModule): (T, B)
    - Batched per-step: (batch_size,)
    - Unbatched per-step: () or (1,)

Recommended dtype:
    - Discrete location indices: integer type (e.g., torch.long)

Notes:
    The dataloader may zero-out location labels when they are not requested
    (see `WalkBatch`), so consumers should treat this signal as optional.
"""

# =============================================================================
# Memory Structures
# =============================================================================

MemoryWriteKind: TypeAlias = Literal["hebbian", "append"]
"""Canonical write-strategy labels for HPC memory presets."""

MemoryStoreKind: TypeAlias = Literal["dense", "factor"]
"""Canonical store-representation labels for HPC memory presets."""

MemoryReadKind: TypeAlias = Literal["attractor_n", "attention_n"]
"""Canonical read-strategy labels for HPC memory presets."""


@dataclass(frozen=True)
class HPCPresetSignature:
    """Public signature of a canonical HPC family.

    The signature makes the write / store / read tuple explicit without
    exposing the whole product space as a public configuration surface.
    """

    write: MemoryWriteKind
    store: MemoryStoreKind
    read: MemoryReadKind


@dataclass(frozen=True)
class LinearMemoryView:
    """Capability view exposing a linear memory operator.

    The callable implements the row-vector map ``h -> h @ M`` for batched
    queries ``h`` with shape ``(B, S)``.
    """

    apply: Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class FactorSlotBank:
    """One explicit factor-memory bank.

    Attributes:
        coefficients: Per-atom coefficients with shape ``(B, T)``. When unset,
            all atoms are treated as having coefficient 1.0.
        keys: Stored key vectors with shape ``(B, T, S)``.
        values: Stored value vectors with shape ``(B, T, S)``.
        valid_mask: Boolean mask of shape ``(B, T)`` marking populated slots.
    """

    keys: Tensor
    values: Tensor
    valid_mask: Tensor
    coefficients: Optional[Tensor] = None

    @property
    def capacity(self) -> int:
        """Return the current slot count carried by this bank."""
        return int(self.keys.shape[1])

    def coefficient_tensor(self) -> Tensor:
        """Return coefficients, defaulting missing entries to ones."""
        if self.coefficients is None:
            return torch.ones_like(self.valid_mask, dtype=self.values.dtype)
        return self.coefficients.to(dtype=self.values.dtype)

    def apply(self, query: Tensor) -> Tensor:
        """Apply the exact linear operator represented by this factor bank."""
        if query.ndim != 2:
            raise ValueError(f"query must be rank-2 `(B, S)`, got shape {tuple(query.shape)}.")
        coefficients = self.coefficient_tensor() * self.valid_mask.to(dtype=self.values.dtype)
        scores = torch.einsum("bs,bts->bt", query.to(dtype=self.keys.dtype), self.keys)
        return torch.einsum("bt,bt,bts->bs", scores, coefficients, self.values)

    def clone(self) -> "FactorSlotBank":
        """Return a cloned factor bank preserving tensor semantics."""
        coefficients = None if self.coefficients is None else self.coefficients.clone()
        return FactorSlotBank(
            keys=self.keys.clone(),
            values=self.values.clone(),
            valid_mask=self.valid_mask.clone(),
            coefficients=coefficients,
        )


DEFAULT_FACTOR_BANK_NAME = "default"
"""Canonical name of the legacy factor-memory bank."""


@dataclass(frozen=True)
class FactorMemoryView:
    """Capability view exposing an explicit factor-memory bank."""

    keys: Tensor
    values: Tensor
    valid_mask: Tensor
    coefficients: Optional[Tensor] = None
    banks: dict[str, FactorSlotBank] = field(default_factory=dict)

    def default_bank(self) -> FactorSlotBank:
        """Return the legacy default factor bank."""
        return FactorSlotBank(
            keys=self.keys,
            values=self.values,
            valid_mask=self.valid_mask,
            coefficients=self.coefficients,
        )

    def bank(self, name: str, *, fallback_to_default: bool = False) -> FactorSlotBank:
        """Return a named bank, optionally falling back to the default bank."""
        if name == DEFAULT_FACTOR_BANK_NAME:
            return self.default_bank()
        bank = self.banks.get(name)
        if bank is not None:
            return bank
        if fallback_to_default:
            return self.default_bank()
        raise KeyError(f"Factor-memory bank '{name}' is not available.")


HebbianMemory = list[Matrix]
"""Attractor network connection weights for memory storage.

Structure:
    - Single memory: [M_g_cued]
    - Dual memory: [M_g_cued, M_x_cued]
    
Where:
    - M_g_cued: Memory retrieved from abstract-location/grid cues.
    - M_x_cued: Memory retrieved from sensory cues (optional).

Shape:
    Each matrix: [batch_size, sum(n_p), sum(n_p)]

Notes:
    Detailed write dynamics are documented in the HPC memory module and in
    `docs/foundations.md`.
"""


@dataclass
class DenseMemoryStore:
    """Dense Hebbian-memory store wrapper.

    Attributes:
        matrix: Dense memory tensor with shape ``(B, S, S)``.
    """

    matrix: Tensor

    def as_linear_view(self) -> LinearMemoryView:
        """Return the linear-operator capability supported by dense memory."""
        return LinearMemoryView(apply=lambda query: (query.unsqueeze(1) @ self.matrix.to(dtype=query.dtype)).squeeze(1))

    def as_factor_view(self) -> FactorMemoryView:
        """Raise because dense stores do not expose explicit factor slots."""
        raise TypeError("Dense memory stores do not expose factor-memory views.")

    def merged_rows(self, flag: Tensor, fresh: "DenseMemoryStore") -> "DenseMemoryStore":
        """Return a new store where flagged batch rows are replaced from ``fresh``."""
        return DenseMemoryStore(matrix=_merge_rows(flag, self.matrix, fresh.matrix))

    def clone(self) -> "DenseMemoryStore":
        """Return a cloned dense-memory store preserving tensor semantics."""
        return DenseMemoryStore(matrix=self.matrix.clone())

    @property
    def kind(self) -> MemoryStoreKind:
        """Return the canonical store kind for dense memory."""
        return "dense"


@dataclass
class FactorMemoryStore:
    """Explicit factor-memory store.

    Attributes:
        coefficients: Per-atom coefficients with shape ``(B, T)``. When unset,
            all atoms are treated as having coefficient 1.0.
        keys: Stored key vectors with shape ``(B, T, S)``.
        values: Stored value vectors with shape ``(B, T, S)``.
        valid_mask: Boolean mask of shape ``(B, T)`` marking populated slots.
    """

    keys: Tensor
    values: Tensor
    valid_mask: Tensor
    coefficients: Optional[Tensor] = None
    banks: dict[str, FactorSlotBank] = field(default_factory=dict)

    def as_linear_view(self) -> LinearMemoryView:
        """Return the exact linear operator induced by the stored factors."""
        return LinearMemoryView(apply=lambda query: self.apply(query))

    def as_factor_view(self) -> FactorMemoryView:
        """Return the explicit factor-bank capability used by attention recall."""
        return FactorMemoryView(
            keys=self.keys,
            values=self.values,
            valid_mask=self.valid_mask,
            coefficients=self.coefficients,
            banks={name: bank.clone() for name, bank in self.banks.items()},
        )

    def default_bank(self) -> FactorSlotBank:
        """Return the legacy default factor bank."""
        return FactorSlotBank(
            keys=self.keys,
            values=self.values,
            valid_mask=self.valid_mask,
            coefficients=self.coefficients,
        )

    def bank(self, name: str, *, fallback_to_default: bool = False) -> FactorSlotBank:
        """Return a named bank, optionally falling back to the default bank."""
        if name == DEFAULT_FACTOR_BANK_NAME:
            return self.default_bank()
        bank = self.banks.get(name)
        if bank is not None:
            return bank
        if fallback_to_default:
            return self.default_bank()
        raise KeyError(f"Factor-memory bank '{name}' is not available.")

    @property
    def bank_names(self) -> tuple[str, ...]:
        """Return explicit named banks excluding the legacy default bank."""
        return tuple(self.banks.keys())

    @property
    def capacity(self) -> int:
        """Return the current slot count carried by the store."""
        return int(self.keys.shape[1])

    @property
    def kind(self) -> MemoryStoreKind:
        """Return the canonical store kind for factor memory."""
        return "factor"

    def coefficient_tensor(self) -> Tensor:
        """Return coefficients, defaulting missing entries to ones."""
        return self.default_bank().coefficient_tensor()

    def apply(self, query: Tensor, *, bank_name: Optional[str] = None) -> Tensor:
        """Apply the exact linear operator represented by the stored factors."""
        if bank_name is not None:
            return self.bank(bank_name, fallback_to_default=True).apply(query)

        banks = [self.default_bank(), *self.banks.values()]
        return sum((bank.apply(query) for bank in banks), torch.zeros_like(query, dtype=self.values.dtype))

    def merged_rows(self, flag: Tensor, fresh: "FactorMemoryStore") -> "FactorMemoryStore":
        """Return a new store where flagged batch rows are replaced from ``fresh``.

        Both stores are padded bank-by-bank to a common slot capacity before the
        row merge so named-bank alignment remains well defined.
        """
        capacities = _bank_capacities(self, fresh)
        current = _pad_factor_store(self, capacities)
        fresh = _pad_factor_store(fresh, capacities)
        merged_banks = {
            name: FactorSlotBank(
                keys=_merge_rows(flag, current_bank.keys, fresh_bank.keys),
                values=_merge_rows(flag, current_bank.values, fresh_bank.values),
                valid_mask=_merge_rows(flag, current_bank.valid_mask, fresh_bank.valid_mask),
                coefficients=_merge_rows(flag, current_bank.coefficient_tensor(), fresh_bank.coefficient_tensor()),
            )
            for name, current_bank in _store_banks(current).items()
            for fresh_bank in (_get_bank(_store_banks(fresh), name, store=fresh),)
        }
        return _store_from_banks(merged_banks)

    def to_dense(self) -> Tensor:
        """Materialize the exact dense linear operator represented by this store."""
        dense: Tensor | None = None
        for bank in _store_banks(self).values():
            coefficients = bank.coefficient_tensor() * bank.valid_mask.to(dtype=bank.values.dtype)
            contribution = torch.einsum("bt,btk,bts->bks", coefficients, bank.keys, bank.values)
            dense = contribution if dense is None else dense + contribution
        if dense is None:
            raise ValueError("Factor memory store must contain at least one bank.")
        return dense

    @classmethod
    def from_dense(cls, matrix: Tensor) -> "FactorMemoryStore":
        """Return an exact factor-store representation of a dense operator."""
        if matrix.ndim != 3:
            raise ValueError(f"matrix must be rank-3 `(B, S, S)`, got shape {tuple(matrix.shape)}.")
        batch_size, rows, cols = matrix.shape
        if rows != cols:
            raise ValueError(f"matrix must be square over feature dimension, got shape {tuple(matrix.shape)}.")

        basis = torch.eye(rows, dtype=matrix.dtype, device=matrix.device).unsqueeze(0).expand(batch_size, -1, -1)
        valid_mask = torch.ones((batch_size, rows), dtype=torch.bool, device=matrix.device)
        coefficients = torch.ones((batch_size, rows), dtype=matrix.dtype, device=matrix.device)
        return cls(keys=basis, values=matrix, valid_mask=valid_mask, coefficients=coefficients)

    def decayed(self, decay: float) -> "FactorMemoryStore":
        """Return a new store with every factor coefficient scaled by ``decay``."""
        scaled_banks = {
            name: FactorSlotBank(
                keys=bank.keys,
                values=bank.values,
                valid_mask=bank.valid_mask,
                coefficients=bank.coefficient_tensor() * float(decay),
            )
            for name, bank in _store_banks(self).items()
        }
        return _store_from_banks(scaled_banks)

    def concatenated(self, fresh: "FactorMemoryStore", *, capacity: Optional[int] = None) -> "FactorMemoryStore":
        """Return a new store containing the atoms of ``self`` followed by ``fresh``.

        When ``capacity`` is provided, the oldest atoms are truncated from the
        left after concatenation.
        """
        merged: dict[str, FactorSlotBank] = {}
        bank_names = sorted(set(_store_banks(self)) | set(_store_banks(fresh)))
        for name in bank_names:
            current_bank = _get_bank(_store_banks(self), name, store=self)
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

    def clone(self) -> "FactorMemoryStore":
        """Return a cloned factor-memory store preserving tensor semantics."""
        coefficients = None if self.coefficients is None else self.coefficients.clone()
        return FactorMemoryStore(
            keys=self.keys.clone(),
            values=self.values.clone(),
            valid_mask=self.valid_mask.clone(),
            coefficients=coefficients,
            banks={name: bank.clone() for name, bank in self.banks.items()},
        )


EpisodicMemoryStore = FactorMemoryStore
"""Backward-compatible alias for the current TEM-t explicit factor store."""


MemoryEntry: TypeAlias = DenseMemoryStore | FactorMemoryStore
"""Single backend-specific memory entry.

    Supported variants:
        - Dense Hebbian-memory store exposing ``matrix`` with shape ``(B, S, S)``.
        - Explicit factor-memory store with tensors ``keys`` and ``values``
          shaped ``(B, T, S)``, ``valid_mask`` shaped ``(B, T)``, and optional
                    ``coefficients`` shaped ``(B, T)``, plus optional named banks with
                    the same tensor conventions.
"""


@dataclass
class MemoryState:
    """Complete HPC memory state at a given timestep.

    Attributes:
        g_cued: Retrieval entry used when recall is indexed by abstract-location
            or grid-side cues. This corresponds to the paper's main memory M.
        x_cued: Retrieval entry used when recall is indexed by sensory cues.
            This corresponds to the optional inference memory M_x.

    Notes:
        Dense attractor memory uses a dense-memory store, while explicit
        attention memory uses per-slot key/value stores.
    """

    g_cued: MemoryEntry
    x_cued: MemoryEntry

    def for_role(self, role: Literal["generative", "inference"]) -> MemoryEntry:
        """Return the memory entry selected by the retrieval role."""
        if role == "generative":
            return self.g_cued
        if role == "inference":
            return self.x_cued
        raise ValueError(f"Invalid retrieval role '{role}'. Expected 'generative' or 'inference'.")


def _merge_rows(flag: Tensor, current: Tensor, fresh: Tensor) -> Tensor:
    """Return a tensor where flagged batch rows are taken from ``fresh``."""
    row_flag = flag.to(dtype=torch.bool, device=current.device).view(-1)
    if current.shape[0] != row_flag.shape[0] or fresh.shape[0] != row_flag.shape[0]:
        raise ValueError("flag, current, and fresh must share the same batch size.")
    broadcast_shape = (row_flag.shape[0],) + (1,) * (current.ndim - 1)
    row_flag = row_flag.view(broadcast_shape)
    return torch.where(row_flag, fresh.to(dtype=current.dtype), current)


def _store_banks(store: FactorMemoryStore) -> dict[str, FactorSlotBank]:
    """Return all banks including the legacy default bank."""
    return {DEFAULT_FACTOR_BANK_NAME: store.default_bank(), **store.banks}


def _store_from_banks(banks: dict[str, FactorSlotBank]) -> FactorMemoryStore:
    """Rebuild a factor store from one complete bank mapping."""
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
    """Return the maximum slot capacity required for each bank across stores."""
    capacities: dict[str, int] = {}
    for store in stores:
        for name, bank in _store_banks(store).items():
            capacities[name] = max(capacities.get(name, 0), bank.capacity)
    return capacities


def _get_bank(banks: dict[str, FactorSlotBank], name: str, *, store: FactorMemoryStore) -> FactorSlotBank:
    """Return one bank or an empty compatible bank when it is absent."""
    bank = banks.get(name)
    if bank is not None:
        return bank
    reference = store.default_bank()
    coefficient_dtype = reference.coefficient_tensor().dtype
    return FactorSlotBank(
        keys=torch.zeros((reference.keys.shape[0], 0, reference.keys.shape[2]), dtype=reference.keys.dtype, device=reference.keys.device),
        values=torch.zeros(
            (reference.values.shape[0], 0, reference.values.shape[2]), dtype=reference.values.dtype, device=reference.values.device
        ),
        valid_mask=torch.zeros((reference.valid_mask.shape[0], 0), dtype=reference.valid_mask.dtype, device=reference.valid_mask.device),
        coefficients=torch.zeros((reference.valid_mask.shape[0], 0), dtype=coefficient_dtype, device=reference.valid_mask.device),
    )


def _pad_factor_store(store: FactorMemoryStore, target_capacity: dict[str, int]) -> FactorMemoryStore:
    """Pad each bank in ``store`` up to the requested capacities."""
    current_banks = _store_banks(store)
    padded_banks = {name: _pad_factor_bank(_get_bank(current_banks, name, store=store), target_capacity[name]) for name in target_capacity}
    return _store_from_banks(padded_banks)


def _pad_factor_bank(bank: FactorSlotBank, target_capacity: int) -> FactorSlotBank:
    """Pad one factor bank with empty slots up to ``target_capacity``."""
    coefficients = bank.coefficient_tensor()
    if bank.capacity >= target_capacity:
        return FactorSlotBank(
            keys=bank.keys,
            values=bank.values,
            valid_mask=bank.valid_mask,
            coefficients=coefficients,
        )

    pad = target_capacity - bank.capacity
    key_padding = torch.zeros((*bank.keys.shape[:1], pad, bank.keys.shape[2]), dtype=bank.keys.dtype, device=bank.keys.device)
    value_padding = torch.zeros((*bank.values.shape[:1], pad, bank.values.shape[2]), dtype=bank.values.dtype, device=bank.values.device)
    mask_padding = torch.zeros((*bank.valid_mask.shape[:1], pad), dtype=bank.valid_mask.dtype, device=bank.valid_mask.device)
    coefficient_padding = torch.zeros((*coefficients.shape[:1], pad), dtype=coefficients.dtype, device=coefficients.device)
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


# =============================================================================
# Data Flow
# =============================================================================


@dataclass(frozen=True)
class SensoryPrediction:
    """Composed sensory observation prediction with values and logits.

    Attributes:
        values: Predicted sensory observation (probabilities or activations)
        logits: Pre-softmax logits corresponding to the prediction

    Notes:
        Both values and logits are kept for loss computation.
    """

    values: MultiScaleCode
    logits: MultiScaleCode


@dataclass(frozen=True)
class LocationInference:
    """Composed latent location prediction with abstract and grounded codes.

    Attributes:
        abstract: Abstract location code (g) representing position in a
                 factorized multi-scale representation
        grounded: Grounded location code (p) representing discrete place cell
                 activations tied to environmental features

    Notes:
        This bundles abstract and grounded codes produced by inference.
    """

    abstract: AbstractLocation
    grounded: GroundedLocation


@dataclass
class StepInput:
    """Input data for a single TEM iteration.

    Attributes:
        observation: Sensory observation (o) for the current timestep
        action: Action taken at the previous timestep (or None for initial step)
        location_info: Environment metadata (e.g., shiny object locations)

    Notes:
        This is a convenience container used by data/rollout utilities.
    """

    observation: MultiScaleCode
    action: Optional[int]
    location_info: dict[str, Any]


@dataclass
class Trajectory:
    """A sequence of inputs forming a walk or episode.

    Attributes:
        steps: Sequence of StepInput objects
        metadata: Optional trajectory-level metadata (e.g., environment ID)

    Notes:
        This is a convenience container used by data/rollout utilities.
    """

    steps: Sequence[StepInput]
    metadata: Optional[dict[str, Any]] = None

    def __len__(self) -> int:
        """Return the number of steps in the trajectory."""
        return len(self.steps)

    def __getitem__(self, index: int) -> StepInput:
        """Get a specific step from the trajectory."""
        return self.steps[index]

    def __iter__(self):
        """Iterate over steps in the trajectory."""
        return iter(self.steps)


# =============================================================================
# Batch Processing
# =============================================================================

WalkSample: TypeAlias = tuple[Observation, Action, Location]
"""Single unbatched walk sample.

This is the item-level return type used by the map-style walk dataset.

Tuple elements:
    observations: Float tensor of shape (T, n_o).
    actions: Integer tensor of shape (T,).
    locations: Integer tensor of shape (T,).

Notes:
    - The TEM DataModule collates a list of `WalkSample` into a time-major `WalkBatch`.
    - The time dimension is always first (time-major), which simplifies truncated BPTT.
"""

WalkBatch: TypeAlias = tuple[Observation, Action, Location]
"""Single time-major batch of walks.

Tuple elements:
    observations: Float tensor of shape (T, B, n_o).
    actions: Integer tensor of shape (T, B).
    locations: Integer tensor of shape (T, B).

Notes:
    `locations` is auxiliary and may be zeroed out by the dataloader's collate
    function when location labels are not required.
"""


BatchedCode = Vector
"""A batched multi-scale code with shape (batch_size, n_cells).

Note: Individual MultiScaleCode elements are already batched.
This type emphasizes that batching occurs at the Vector level.
"""

BatchedMemory = Matrix
"""A batched memory matrix with shape (batch_size, n_cells_pre, n_cells_post).

Note: In practice, memory is typically shared across a batch rather than
per-sample, so this represents the global memory state.
"""

Reduction: TypeAlias = Literal["none", "sum", "mean"]
Scalar: TypeAlias = int | float | Tensor


@dataclass
class Prediction:
    """Predicted observations and logits."""

    prediction: Observation
    logits: Tensor


# TODO
Activation = Literal["leaky_relu", "sigmoid", "none"]
ProjectionKind = Literal["identity", "linear", "tiling", "low_rank"]
ProjectionBridge = Literal["auto", "aligned", "broadcast"]
ProjectionEndpointKind = Literal["flat", "multiscale", "token_sequence"]
ProjectionMode: TypeAlias = ProjectionKind
InitStrategy = Literal["identity", "random"]
Channels: TypeAlias = dict[str, np.ndarray]
Batch: TypeAlias = dict[str, Tensor]  # Generic batch type, can be specialized as needed
StepBatchSource = Iterator[Batch]

RetrievalRole = Literal["generative", "inference"]
"""Semantic role used to select the cue-indexed HPC retrieval path."""
