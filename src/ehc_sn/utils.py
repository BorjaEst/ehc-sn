"""Utility functions and classes for EHC-SN"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class NeuralNetwork:
    """Neural Network class for EHC-SN"""

    def __init__(self, shape):
        self.shape = shape  # Store the shape
        self._w = np.random.rand(np.prod(shape))  # Initialize weights

    @property
    def map(self):
        """Cognitive map representation"""  # w_i = ln[θ_i]
        return np.exp(self._w).reshape(self.shape)

    def __likelihood(self, y: npt.NDArray[np.float32]) -> float:
        # Note ln[p(y|Θ_k)] actually proportional to y·ln[θ_k] = y·w
        return y.flatten() @ self._w  # Eq. (5)

    def __call__(self, y: npt.NDArray[np.float32]) -> float:
        """Normalized likelihood of trajectory given a map."""
        return np.exp(self.__likelihood(y))

    def plot(self, ax: plt.Axes):
        """Plot the cognitive map"""
        ax.imshow(self.map, cmap="viridis")
        ax.set_title("Cognitive Map")
