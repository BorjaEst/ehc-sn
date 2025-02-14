"""Sequential Navigation (SN) module for the Entorhinal–Hippocampal circuit (EHC)"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from ehc_sn.equations import (
    CognitiveMap,
    Item,
    Mixing,
    Observation,
    Sequence,
    PrioritizedMap,
)
from ehc_sn.utils import CognitiveMap, kron_delta


@dataclass
class HGModelParams:
    """The param set class example for model configuration."""

    δ: float = 0.7  # Discount factor for sequence
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


def get_observation(x: Item, i: int) -> Observation:  # Eq. (0)
    """Return the observation code for item."""
    return (x * np.eye(x.size))[i]


def get_item(Ξ: List[Observation]) -> Item:  # Eq. (1)
    """Return the hidden code for item."""
    return np.array(Ξ).sum(axis=0)


def get_sequence(items: List[Item], δ: float = 0.7) -> Sequence:  # Eq. (2)
    """Return the hidden code for sequence."""
    T = len(items)  # Number of items
    discounted = [x * δ ** (T - t) for t, x in enumerate(items, 1)]
    return np.array(discounted).sum(axis=0)


def prob_sequence(y: Sequence, Θ: List[CognitiveMap], z: Mixing) -> float:  # Eq. (3)
    """Return the probability of a sequence."""
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
        y: Sequence,  # y(t-1)
        Θ: List[CognitiveMap],
        z: Optional[Mixing] = None,
    ) -> Tuple[Item, Sequence, Mixing, np.int64]:
        """Inference function, returns predicted next item and sequence."""
        # Update mixing probabilities or use the provided ones
        z = self.π if z is None else self._estimate_mixing(z, y, Θ)
        k = np.argmax(z)  # Get the best map index
        x = self._estimate_item(ξ, y, Θ[k])  # Predict item code
        # TODO: Check: ξ = pred_observation(x)  # Predict the next observation
        y = self._estimate_sequence(ξ, y, Θ[k])  # Update the sequence
        return x, y, z, k

    def _estimate_mixing(  # Eq. (6) and Eq. (7)
        self, z: Mixing, y: Sequence, Θ: List[CognitiveMap]
    ) -> Mixing:
        """Estimate posterior mixing probabilities."""
        τ = self.parameters.τ  # Extract exponential decay for mixing
        return np.array([z_i**τ * θ(y) ** (1 - τ) for θ, z_i in zip(Θ, z)])

    def _estimate_item(  # Eq. (8) and Eq. (9)
        self, ξ: Observation, y: Sequence, θ: CognitiveMap
    ) -> Item:
        """Estimate the posterior hidden item code."""
        c = self.parameters.c  # Extract velocity rate for item code
        v = ξ - self._ξ  # velocity v(t-1) = ξ(t-1) - ξ(t-2)
        μ, Σ = ξ + c * v, v  # Using ξ(t-1) and v(t-1)
        self._ξ = ξ  # Update the observation code
        return θ * np.random.normal(μ, abs(Σ)) - y  # Random movement probability

    def _estimate_sequence(  # Eq. (10)
        self, ξ: Observation, y: Sequence, θ: CognitiveMap
    ) -> Sequence:
        """Estimate the posterior sequence code."""
        δ = self.parameters.δ  # Extract parameters
        return δ * y + θ * (1 - δ) + ξ

    def learning(  # pylint: disable=too-many-arguments
        self,
        episode: List[List[Observation]],
        γ: float = 0.1,
        λ: float = 0.1,
    ) -> List[PrioritizedMap]:
        """Learning function for the model."""
        _, N = self.shape  # Get the number of items
        z = None  # Initialize mixing probabilities
        for sequence in episode:
            Θ = self.maps_inference()  # Priority maps
            x, y = np.zeros(N), np.zeros(N)  # First item and sequence
            for ξ in sequence:
                x, y, z, k = self.inference(ξ, x, y, Θ, z)
                self.π[k] = (1 - γ) * self.π[k] + z[k]  # Eq. (11)
                self.ρ[k] = self.ρ[k] + z[k] * y  # Eq. (12)
                ξ_i = np.argmax(ξ)  # Get the observation index
                Θ[k].θ = (1 - λ) * Θ[k].θ - kron_delta(ξ_i, np.log(x))  # Eq. (13)
        return Θ  # Return the updated prioritized maps

    def maps_inference(self):
        """Generates a list of CognitiveMap instances."""
        return [CognitiveMap(θ) for θ in self.ρ]  # Priority maps
