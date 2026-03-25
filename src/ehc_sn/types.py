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
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, TypeAlias

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
class FactorMemoryView:
    """Capability view exposing an explicit factor-memory bank."""

    keys: Tensor
    values: Tensor
    valid_mask: Tensor
    coefficients: Optional[Tensor] = None


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
        )

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
        if self.coefficients is None:
            return torch.ones_like(self.valid_mask, dtype=self.values.dtype)
        return self.coefficients.to(dtype=self.values.dtype)

    def apply(self, query: Tensor) -> Tensor:
        """Apply the exact linear operator represented by the stored factors."""
        if query.ndim != 2:
            raise ValueError(f"query must be rank-2 `(B, S)`, got shape {tuple(query.shape)}.")
        coefficients = self.coefficient_tensor() * self.valid_mask.to(dtype=self.values.dtype)
        scores = torch.einsum("bs,bts->bt", query.to(dtype=self.keys.dtype), self.keys)
        return torch.einsum("bt,bt,bts->bs", scores, coefficients, self.values)

    def clone(self) -> "FactorMemoryStore":
        """Return a cloned factor-memory store preserving tensor semantics."""
        coefficients = None if self.coefficients is None else self.coefficients.clone()
        return FactorMemoryStore(
            keys=self.keys.clone(),
            values=self.values.clone(),
            valid_mask=self.valid_mask.clone(),
            coefficients=coefficients,
        )


EpisodicMemoryStore = FactorMemoryStore
"""Backward-compatible alias for the current TEM-t explicit factor store."""


MemoryEntry: TypeAlias = DenseMemoryStore | FactorMemoryStore
"""Single backend-specific memory entry.

    Supported variants:
        - Dense Hebbian-memory store exposing ``matrix`` with shape ``(B, S, S)``.
        - Explicit factor-memory store with tensors ``keys`` and ``values``
          shaped ``(B, T, S)``, ``valid_mask`` shaped ``(B, T)``, and optional
          ``coefficients`` shaped ``(B, T)``.
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
    location_info: Dict[str, Any]


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
    metadata: Optional[Dict[str, Any]] = None

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
ProjectionMode = Literal["identity", "tiling", "low_rank", "random"]
InitStrategy = Literal["identity", "random"]


Device = torch.device
Dtype = torch.dtype
Channels = dict[str, np.ndarray]
Batch = Dict[str, Tensor]  # Generic batch type, can be specialized as needed


RetrievalRole = Literal["generative", "inference"]
"""Semantic role used to select the cue-indexed HPC retrieval path."""
