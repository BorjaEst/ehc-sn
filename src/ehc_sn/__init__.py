"""Sequential Navigation (SN) module for the Entorhinal–Hippocampal circuit (EHC)"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ehc_sn.utils import CognitiveMap, kronecker

# Type alias for integer space arrays
Observation = npt.NDArray[np.float64]  # Observation
Velocity = npt.NDArray[np.float64]  # Velocity
Item = npt.NDArray[np.float64]  # Navigation Item
Trajectory = npt.NDArray[np.float64]  # Navigation Trajectory
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


def get_trajectory(items: List[Item], δ: float = 0.7) -> Trajectory:  # Eq. (2)
    """Return the hidden code for trajectory."""
    T = len(items)  # Number of items
    discounted = [x * δ ** (T - t) for t, x in enumerate(items, 1)]
    return np.array(discounted).sum(axis=0)


def prob_trajectory(  # Eq. (3)
    y: Trajectory, Θ: List[CognitiveMap], z: Mixing
) -> float:
    """Return the probability of a trajectory."""
    p_dist = [θ(y) * z_i for θ, z_i in zip(Θ, z)]
    return np.array(p_dist).sum(axis=0)


def pred_observation(x: Item) -> Observation:  # Eq. (9)
    """Return the predicted observation code."""
    return (x * np.eye(x.size))[x.argmax()]


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
        ξ: Observation,  # ξ(t-1)
        x: Item,  # x(t-1)  TODO: This variable seems to not be used
        y: Trajectory,  # y(t-1)
        Θ: List[CognitiveMap],
        z: Optional[Mixing] = None,
    ) -> Tuple[Item, Trajectory, Mixing, np.int64]:
        """Inference function, returns predicted next item and trajectory."""
        # Update mixing probabilities or use the provided ones
        z = self.π if z is None else self._estimate_mixing(z, y, Θ)
        k = np.argmax(z)  # Get the best map index
        x = self._estimate_item(ξ, y, Θ[k])  # Predict item code
        # TODO: Check: ξ = pred_observation(x)  # Predict the next observation
        y = self._estimate_trajectory(ξ, y, Θ[k])  # Update the trajectory
        return x, y, z, k

    def _estimate_mixing(  # Eq. (6) and Eq. (7)
        self, z: Mixing, y: Trajectory, Θ: List[CognitiveMap]
    ) -> Mixing:
        """Estimate posterior mixing probabilities."""
        τ = self.parameters.τ  # Extract exponential decay for mixing
        return np.array([z_i**τ * θ(y) ** (1 - τ) for θ, z_i in zip(Θ, z)])

    def _estimate_item(  # Eq. (8) and Eq. (9)
        self, ξ: Observation, y: Trajectory, θ: CognitiveMap
    ) -> Item:
        """Estimate the posterior hidden item code."""
        c = self.parameters.c  # Extract velocity rate for item code
        v = ξ - self._ξ  # velocity v(t-1) = ξ(t-1) - ξ(t-2)
        μ, Σ = ξ + c * v, v  # Using ξ(t-1) and v(t-1)
        self._ξ = ξ  # Update the observation code
        return θ * np.random.normal(μ, abs(Σ)) - y  # Random movement probability

    def _estimate_trajectory(  # Eq. (10)
        self, ξ: Observation, y: Trajectory, θ: CognitiveMap
    ) -> Trajectory:
        """Estimate the posterior sequence code."""
        δ = self.parameters.δ  # Extract parameters
        return δ * y + θ * (1 - δ) + ξ

    def learning(
        self,
        episode: List[List[Observation]],
        γ: float = 0.01,
        λ: float = 0.1,
    ) -> Tuple[
        npt.NDArray[np.float64],
        List[npt.NDArray[np.float64]],
        List[CognitiveMap],
    ]:
        """Learning function for the model."""
        _, N = self.shape  # Get the number of items
        z = None  # Initialize mixing probabilities
        for sequence in episode:
            Θ = [CognitiveMap(θ) for θ in self.ρ]  # Initialize cognitive maps
            x, y = np.zeros(N), np.zeros(N)
            for ξ in sequence:
                x, y, z, k = self.inference(ξ, x, y, Θ, z)
                self.π[k] = (1 - γ) * self.π[k] + z[k]  # Eq. (11)
                self.ρ[k] = self.ρ[k] + z[k] * y  # Eq. (12)
                ξ_i = np.argmax(ξ)  # Get the observation index
                Θ[k].θ = (1 - λ) * Θ[k].θ - kronecker(ξ_i, N) @ np.log(x)  # Eq. (13)
        return self.π, self.ρ, Θ
