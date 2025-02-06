from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import numpy.typing as npt
from ehc_sn.utils import CognitiveMap, kron_delta

# pylint: disable=non-ascii-name

# Type alias for integer space arrays
Observation = npt.NDArray[np.float64]  # Observation
Velocity = npt.NDArray[np.float64]  # Velocity
Item = npt.NDArray[np.float64]  # Navigation Item
Trajectory = npt.NDArray[np.float64]  # Navigation Trajectory
# Mixing = npt.NDArray[np.float64]  # Mixing probabilities
PrioritizedMap = CognitiveMap  # Prioritized cognitive map
PriorMixing = npt.NDArray[np.float64]  # Prioritized mixing probabilities
PriorMaps = List[npt.NDArray[np.float64]]  # Prioritized cognitive maps

Mixing = Dict[CognitiveMap, np.float64]  # Mixing probabilities
Map = CognitiveMap  # Cognitive map


# Custom.
def get_observation(x: Item, i: int) -> Observation:  # Eq. (0)
    """Return the observation code for item."""
    return (x * np.eye(x.size))[i]


# Eq. (1)
def get_item(Ξ: List[Observation]) -> Item:
    """Return the hidden code for item."""
    return np.array(Ξ).sum(axis=0)


# Eq. (2)
def get_trajectory(X: List[Item], δ: float = 0.7) -> Trajectory:
    """Return the hidden code for trajectory."""
    T = len(X)  # Number of items
    discounted = [x * δ ** (T - t) for t, x in enumerate(X, 1)]
    return np.array(discounted).sum(axis=0)


# Eq. (3)
def p_trajectory(y: Trajectory, Θ: Mixing) -> float:
    """Return the probability of a trajectory in a mixing."""
    p_dist = [p(y, θ) * z_i for θ, z_i in Θ.items()]
    return np.array(p_dist).sum(axis=0)


# Eq. (4)
def p(y: Trajectory, θ: Map) -> float:
    """Return the probability of a trajectory in a map."""
    p_dist = θ.values**y
    return np.array(p_dist).prod(axis=0)


# Eq. (5)
def lnp(y: Trajectory, θ: Map) -> float:
    """Return the log-likelihood of a trajectory in a map."""
    p_dist = y @ np.log(θ.values)
    return np.array(p_dist).sum(axis=0)


# Eq. (6)
def z(Θ: Mixing, y: Trajectory, τ: float = 0.9) -> list[float]:
    """Return the mixing probability values."""
    return [z**τ * p(y, θ) ** (1 - τ) for θ, z in Θ.items()]


# Eq. (7)
def lnz(Θ: Mixing, y: Trajectory, τ: float = 0.9) -> list[float]:
    """Return ln of mixing probability values."""
    return [τ * np.log(z) + (1 - τ) * lnp(y, θ) for θ, z in Θ.items()]


# Eq. (8)
def item(ξ: Observation, y: Trajectory, θ: Map) -> Item:
    """Return the hidden code for item."""
    # ξ here is a noisy prediction about the observation
    return ξ * θ.values - y


# Eq. (9)
def observation(x: Item) -> Observation:
    """Return the predicted observation code."""
    return (x * np.eye(x.size))[x.argmax()]
