"""Utility functions and classes for EHC-SN"""

import numpy as np
import numpy.typing as npt


class NeuralNetwork:
    """Neural Network class for EHC-SN"""

    def __init__(self, size):
        self._w = np.random.rand(size)  # Initialize weights

    @property
    def map(self):
        """Cognitive map representation"""  # w_i = ln[θ_i]
        return np.exp(self._w)

    def __likelihood(self, y: npt.NDArray[np.float32]) -> float:
        # Note ln[p(y|Θ_k)] actually proportional to y·ln[θ_k] = y·w
        return y @ self._w  # Eq. (5)

    def __call__(self, y: npt.NDArray[np.float32]) -> float:
        """Normalized likelihood of trajectory given a map."""
        return np.exp(self.__likelihood(y))


def kronecker(ξ, N):
    """Return the Kronecker delta matrix."""
    Δ = np.zeros([N] * 2)
    Δ[ξ, ξ] = 1
    return Δ
