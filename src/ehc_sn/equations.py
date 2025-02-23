"""Equations for the EHC-SN model."""

import numpy as np
from ehc_sn import utils
from numpy import typing as npt

# pylint: disable=non-ascii-name
# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name

# Type aliases for the EHC-SN model
Array = npt.NDArray[np.floating]  # Alias for array of floats
Observation = npt.NDArray[np.floating]  # Observation
Velocity = npt.NDArray[np.floating]  # Velocity
Item = npt.NDArray[np.floating]  # Navigation Item
Sequence = npt.NDArray[np.floating]  # Navigation Sequence
Mixing = npt.NDArray[np.floating]  # Mixing probabilities
Map = npt.NDArray[np.floating]  # Navigation Map

# Constant limits for numerical stability
LOG_LIMIT = -1e12  # Logarithm limit for numerical stability
LNP_LIMIT = -1e10  # Logarithm probability limit


# Custom.
def get_observation(x: Item, i: int) -> Observation:
    """Return the observation code for item."""
    return (x * np.eye(x.size))[i]


# Eq. (1)
def get_item(Ξ: list[Observation]) -> Item:
    """Return the hidden code for item."""
    return np.array(Ξ).sum(axis=0)


# Eq. (2)
def get_sequence(X: list[Item], δ: float) -> Sequence:
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
    p_dist = θ**y
    return np.array(p_dist).prod(axis=0)


# Eq. (5)
def lnp(y: Sequence, θ: Map) -> float:
    """Return the log-likelihood of a sequence in a map."""
    p_dist = y @ np.maximum(np.log(θ), LOG_LIMIT)
    result = np.array(p_dist).sum(axis=0)  # Nan protection
    return result if result > LNP_LIMIT else -np.inf


# Eq. (6)
def mixing(y: Sequence, Θ: list[Map], z: Mixing, τ: float) -> Mixing:
    """Return the mixing probability values."""
    _z = [z_i**τ * p(y, θ) ** (1 - τ) for θ, z_i in zip(Θ, z)]
    return np.array(_z, dtype=np.float64)


# Eq. (7)
def lnz(y: Sequence, Θ: list[Map], z: Mixing, τ: float) -> Mixing:
    """Return ln of mixing probability values."""
    _lnz = [τ * np.log(z_i) + (1 - τ) * lnp(y, θ) for θ, z_i in zip(Θ, z)]
    return np.array(_lnz, dtype=np.float64)


# Eq. (8)
def item(ξ: Observation, y: Sequence, θ: Map, v: Velocity, c: float) -> Item:
    """Return the hidden code for item."""
    μ, Σ = ξ + c * v, v * np.eye(v.size)  # Noise parameters
    ξn = np.random.multivariate_normal(μ, Σ)  # Noisy observation
    return ξn * θ - y


# Eq. (9)
def observation(x: Item) -> Observation:
    """Return the predicted observation code."""
    return (x * np.eye(x.size))[x.argmax()]


# Eq. (10)
def sequence(ξ: Observation, y: Sequence, θ: Map, δ: float) -> Sequence:
    """Return the predicted sequence code."""
    return δ * y + (1 - δ) * θ + ξ


# Eq. (11)
def π_update(π_k: float, z_k: float, γ: float) -> float:
    """Return mixing hyperparameters."""
    return (1 - γ) * π_k + z_k


# Eq. (12)
def ρ_update(ρ_k: Array, z_k: float, y: Sequence) -> Array:
    """Return map hyperparameters."""
    return ρ_k + z_k * y


# Eq. (13)
def pmap_update(θ_k: Map, ξ: Observation, x: Item, λ: float) -> Map:
    """Return the updated map."""
    ξ_index = ξ != 0  # Index of the observations
    δ = utils.kronecker_delta(ξ_index, np.log(x))
    return (1 - λ) * θ_k - δ
