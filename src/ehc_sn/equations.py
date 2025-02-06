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
def observation(x: Item, i: int) -> Observation:  # Eq. (0)
    """Return the observation code for item."""
    return (x * np.eye(x.size))[i]


# Eq. (1)
def item(Ξ: List[Observation]) -> Item:
    """Return the hidden code for item."""
    return np.array(Ξ).sum(axis=0)


# Eq. (2)
def trajectory(X: List[Item], δ: float = 0.7) -> Trajectory:
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
def p(y: Trajectory, θ: Map, γ: float = 1.0) -> float:
    """Return the probability of a trajectory in a map."""
    p_dist = θ.values**y
    return γ * np.array(p_dist).prod(axis=0)
