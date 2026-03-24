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

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol, Sequence, TypeAlias, runtime_checkable

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


@runtime_checkable
class SupportsExplicitMemory(Protocol):
    """Protocol for backend memory stores with explicit key/value slots.

    Attributes:
        keys: Stored key tensor of shape ``(B, T, S)``.
        values: Stored value tensor of shape ``(B, T, S)``.
        valid_mask: Boolean validity mask of shape ``(B, T)``.
    """

    keys: Tensor
    values: Tensor
    valid_mask: Tensor


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


MemoryEntry: TypeAlias = Matrix | SupportsExplicitMemory
"""Single backend-specific memory entry.

    Supported variants:
        - Dense Hebbian memory matrix with shape ``(B, S, S)``.
        - Explicit key/value memory store with tensors ``keys`` and ``values``
          shaped ``(B, T, S)`` and ``valid_mask`` shaped ``(B, T)``.
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
        Dense attractor memory uses matrices, while explicit attention memory
        uses per-slot key/value stores.
    """

    g_cued: MemoryEntry
    x_cued: MemoryEntry

    def for_operation(self, operation: Literal["generative", "inference"]) -> MemoryEntry:
        """Return the memory entry selected by HPC operation role."""
        if operation == "generative":
            return self.g_cued
        if operation == "inference":
            return self.x_cued
        raise ValueError(f"Invalid operation '{operation}'. Expected 'generative' or 'inference'.")


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


OperationMode = Literal["generative", "inference"]
