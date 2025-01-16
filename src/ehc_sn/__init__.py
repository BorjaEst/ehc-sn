"""Sequential Navigation (SN) module for the Entorhinal–Hippocampal circuit (EHC)"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ehc_sn.utils import NeuralNetwork, kronecker

# Type alias for integer space arrays
Observation = npt.NDArray[np.float64]  # Observation
Velocity = npt.NDArray[np.float64]  # Velocity
Item = npt.NDArray[np.float64]  # Navigation Item
Trajectory = npt.NDArray[np.float64]  # Navigation Trajectory
Map = NeuralNetwork  # Cognitive Map
MapSet = dict[NeuralNetwork, float]  # Set of cognitive maps
Mixing = npt.NDArray[np.float64]  # Mixing probabilities


@dataclass
class HGModelParams:
    """The param set class example for model configuration."""

    δ: float = 0.7  # Discount factor for trajectory
    τ: float = 0.9  # Exponential decay for mixing categorical distribution
    c: float = 0.4  # Velocity rate for item code update

    def validate(self):
        """Validate the parameters."""
        if not isinstance(self.δ, float) or not 0 <= self.δ <= 1:
            raise ValueError(f"δ: {self.δ}. Must be a float between 0 and 1.")
        if not isinstance(self.τ, float) or not 0 <= self.τ <= 1:
            raise ValueError(f"τ: {self.τ}. Must be a float between 0 and 1.")
        if not isinstance(self.c, float) or not 0 <= self.c <= 1:
            raise ValueError(f"c: {self.c}. Must be a float between 0 and 1.")


def get_trajectory(items: List[Item], δ: float = 0.7) -> Trajectory:
    """Return the hidden code for trajectory."""  # Eq. (2)
    T = len(items)  # Number of items
    discounted = [x * δ ** (T - t) for t, x in enumerate(items, 1)]
    return np.array(discounted).sum(axis=0)


def p_trajectory(y: Trajectory, Θ: MapSet) -> float:
    """Return the probability of a trajectory."""  # Eq. (3)
    p_dist = [θ(y) * p for θ, p in Θ.items()]
    return np.array(p_dist).sum(axis=0)


class HierarchicalGenerativeModel:
    """Hierarchical generative model for sequential navigation."""

    def __init__(self, α: List[float], N: int, parameters: HGModelParams):
        # Store and validate the parameters
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer.")
        self.parameters = parameters  # Automatically validated
        # Initialize mixing probabilities and structural parameters
        self.π = np.random.dirichlet(α)  # Belief degree for each map
        self.ρ = [np.random.dirichlet(np.ones(N)) for _ in range(len(α))]
        # Initialize private and auxiliary variables
        self._ξ: Observation = np.zeros(N, dtype=np.float32)
        self._v: Velocity = np.zeros(N, dtype=np.float32)

    @property
    def parameters(self) -> HGModelParams:
        """Return the parameters of the model."""
        return self.__parameters

    @parameters.setter
    def parameters(self, params: HGModelParams):
        """Set the parameters of the model."""
        if not isinstance(params, HGModelParams):
            raise ValueError("parameters must be instance of HGModelParams.")
        params.validate()  # Validate the parameters
        self.__parameters = params

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the model (k, N)."""
        return len(self.π), len(self.ρ[0])

    # @property
    # def cognitive_maps(self) -> MapSet:
    #     """Return a dictionary of cognitive maps."""
    #     return {NeuralNetwork(θ): z for z, θ in zip(self.π, self.ρ)}

    # @property
    # def best(self) -> Tuple[Map, float]:
    #     """Return the best cognitive map."""
    #     return max(self.cognitive_maps.items(), key=lambda x: x[1])

    def inference(  # pylint: disable=too-many-arguments
        self,
        ξ: Observation,
        x: Item,
        y: Trajectory,
        Θ: List[Map],
        z: Optional[Mixing] = None,
    ) -> Tuple[Item, Trajectory, Mixing, np.int64]:
        """Inference function, returns predicted next item and trajectory."""
        # Update mixing probabilities or use the provided ones
        z = self.π if z is None else self._estimate_mixing(z, y, Θ)
        k = np.argmax(z)  # Get the best map index
        x = self._estimate_item(Θ[k], y)  # Predict item code
        self._v, self._ξ = ξ - self._ξ, ξ  # Update v(t) and ξ(t)
        y = self._estimate_trajectory(Θ[k], ξ, y)  # Update the trajectory
        return x, y, z, k

    def _estimate_mixing(self, z: Mixing, y: Trajectory, Θ: List[Map]) -> Mixing:
        """Estimate posterior mixing probabilities."""  # Eq. (6) and Eq. (7)
        τ = self.parameters.τ  # Extract exponential decay for mixing
        return np.array([z_i**τ * θ(y) ** (1 - τ) for θ, z_i in zip(Θ, z)])

    def _estimate_item(self, θ: Map, y: Trajectory) -> Item:
        """Estimate the posterior hidden item code."""  # Eq. (8) and Eq. (9)
        c = self.parameters.c  # Extract velocity rate for item code
        μ, Σ = self._ξ + c * self._v, self._v  # Using ξ(t-1) and v(t-1)
        return np.random.normal(μ, Σ) * θ.map - y  # Random movement probability

    def _estimate_trajectory(self, θ: Map, ξ: Observation, y: Trajectory) -> Trajectory:
        """Estimate the posterior sequence code."""  # Eq. (10)
        δ = self.parameters.δ  # Extract parameters
        return δ * y + (1 - δ) * θ.map + ξ

    def _estimate_observation(self, x: Item) -> Observation:
        """Estimate the posterior observation code."""
        return np.argmax(x, axis=0)
