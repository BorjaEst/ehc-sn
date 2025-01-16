"""Utility functions and classes for EHC-SN"""

import numpy as np
import numpy.typing as npt


class NeuralNetwork:
    """Neural Network class for EHC-SN"""

    def __init__(self, θ: npt.NDArray[np.float64]):
        # Initialize the synaptic weights (structural parameters)
        self.θ: npt.NDArray[np.float64] = θ  # Cat(ρ) == ρ

    @property
    def map(self):
        """Cognitive map representation"""  # w_i = ln[θ_i]
        return np.exp(self.θ)

    def __likelihood(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Note ln[p(y|Θ_k)] actually proportional to y·ln[θ_k] = y·w
        return y @ self.θ  # Eq. (5)

    def __call__(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Normalized likelihood of trajectory given a map."""
        return np.exp(self.__likelihood(y))


def kronecker(ξ: int, N: int) -> npt.NDArray[np.bool_]:
    """Return the Kronecker delta matrix."""
    Δ = np.zeros([N] * 2, dtype=np.bool_)
    Δ[ξ, ξ] = True
    return Δ
