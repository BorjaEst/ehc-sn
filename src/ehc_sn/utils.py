"""Utility functions and classes for EHC-SN"""

from typing import Tuple

import numpy as np
import numpy.typing as npt


class CognitiveMap:  # pylint: disable=too-few-public-methods
    """Neural Network class for EHC-SN"""

    def __init__(self, θ: npt.NDArray[np.float64]):
        # Initialize the synaptic weights (structural parameters)
        self.θ: npt.NDArray[np.float64] = θ  # Cat(ρ) == ρ

    def _likelihood(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Note ln[p(y|Θ_k)] actually proportional to y·ln[θ_k]
        return y @ np.log(self.θ)  # Eq. (5)

    def __call__(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Normalized likelihood of trajectory given a map."""
        return np.exp(self._likelihood(y))

    def __mul__(self, other) -> npt.NDArray[np.float64]:
        """Element-wise multiplication."""
        return self.θ * other

    def __add__(self, other) -> npt.NDArray[np.float64]:
        """Element-wise addition."""
        return self.θ + other

    @property
    def params(self) -> npt.NDArray[np.float64]:
        """Return the structural parameters."""
        return self.θ


def kronecker(ξ: np.int64, N: int) -> npt.NDArray[np.bool_]:
    """Return the Kronecker delta matrix."""
    Δ = np.zeros([N] * 2, dtype=np.bool_)
    Δ[ξ, ξ] = True
    return Δ


def rand_episode(
    start: Tuple[int, int], size: Tuple[int, int], T: int
) -> npt.NDArray[np.float64]:
    """Return a random episode."""
    base_episode = np.zeros([T, *size], dtype=np.float64)
    position = np.array(start)
    for t in range(T):
        base_episode[t, *position] = 1.0
        position += np.random.choice([-1, 1], size=2) * (np.random.rand(2) < 0.5)
        position = np.clip(position, [0, 0], np.array(size) - 1)
    return base_episode
