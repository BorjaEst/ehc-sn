"""Equations for the EHC-SN model."""

from typing import List

import numpy as np
from ehc_sn.utils import CognitiveMap
from numpy.typing import NDArray

# pylint: disable=non-ascii-name
# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name

# Type alias for integer space arrays
Observation = NDArray[np.float64]  # Observation
Velocity = NDArray[np.float64]  # Velocity
Item = NDArray[np.float64]  # Navigation Item
Sequence = NDArray[np.float64]  # Navigation Sequence
Mixing = NDArray[np.float64]  # Mixing probabilities
PrioritizedMap = CognitiveMap  # Prioritized cognitive map
Map = CognitiveMap  # Cognitive map


LOG_LIMIT = -1e12  # Logarithm limit for numerical stability
LNP_LIMIT = -1e10  # Logarithm probability limit


# Custom.
def get_observation(x: Item, i: int) -> Observation:
    """Return the observation code for item."""
    return (x * np.eye(x.size))[i]


# Eq. (1)
def get_item(Ξ: List[Observation]) -> Item:
    """Return the hidden code for item."""
    return np.array(Ξ).sum(axis=0)


# Eq. (2)
def get_sequence(X: List[Item], δ: float = 0.7) -> Sequence:
    """Return the hidden code for sequence."""
    T = len(X)  # Number of items
    discounted = [x * δ ** (T - t) for t, x in enumerate(X, 1)]
    return np.array(discounted).sum(axis=0)


# Eq. (3)
def p_sequence(y: Sequence, Θ: list[Map], z: Mixing) -> float:
    """Return the probability of a sequence in a mixing."""
    p_dist = [p(y, θ) * z_i for θ, z_i in zip(Θ, z)]
    return np.array(p_dist).sum(axis=0)


# Eq. (4)
def p(y: Sequence, θ: Map) -> float:
    """Return the probability of a sequence in a map."""
    p_dist = θ.values**y
    return np.array(p_dist).prod(axis=0)


# Eq. (5)
def lnp(y: Sequence, θ: Map) -> float:
    """Return the log-likelihood of a sequence in a map."""
    p_dist = y @ np.maximum(np.log(θ.values), LOG_LIMIT)
    result = np.array(p_dist).sum(axis=0)  # Nan protection
    return result if result > LNP_LIMIT else -np.inf


# Eq. (6)
def mixing(y: Sequence, Θ: list[Map], z: Mixing, τ: float = 0.9) -> Mixing:
    """Return the mixing probability values."""
    _z = [z_i**τ * p(y, θ) ** (1 - τ) for θ, z_i in zip(Θ, z)]
    return np.array(_z, dtype=np.float64)


# Eq. (7)
def lnz(y: Sequence, Θ: list[Map], z: Mixing, τ: float = 0.9) -> Mixing:
    """Return ln of mixing probability values."""
    _lnz = [τ * np.log(z_i) + (1 - τ) * lnp(y, θ) for θ, z_i in zip(Θ, z)]
    return np.array(_lnz, dtype=np.float64)


# Eq. (8)
def item(ξ: Observation, y: Sequence, θ: Map) -> Item:
    """Return the hidden code for item."""
    # ξ here is a noisy prediction about the observation
    return ξ * θ.values - y


# Eq. (9)
def observation(x: Item) -> Observation:
    """Return the predicted observation code."""
    return (x * np.eye(x.size))[x.argmax()]


# Eq. (10)
def sequence(ξ: Observation, y: Sequence, θ: Map, δ: float = 0.9) -> Sequence:
    """Return the predicted sequence code."""
    return δ * y + (1 - δ) * θ.values + ξ


# Eq. (11)
def π_update(π_k: float, z_k: float, γ: float = 0.1) -> float:
    """Return mixing hyperparameters."""
    return (1 - γ) * π_k + z_k


# Eq. (12)
def ρ_update(ρ_k: list[float], z_k: float, y: Sequence) -> list[float]:
    """Return map hyperparameters."""
    return [float(ρ_ki) for ρ_ki in ρ_k + z_k * y]
