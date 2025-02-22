"""Sequential Navigation (SN) module for the Entorhinal–Hippocampal circuit (EHC)"""

from typing import List, Optional, Tuple

import numpy as np
from ehc_sn import equations as eq
from ehc_sn.config import HGModelSettings
from ehc_sn.equations import Item, Mixing, Observation, PrioritizedMap, Sequence
from ehc_sn.utils import CognitiveMap, kron_delta

# pylint: disable=non-ascii-name


class HierarchicalGenerativeModel:
    """Hierarchical generative model for sequential navigation."""

    def __init__(self, α: List[float], N: int, settings: HGModelSettings):
        # Store and validate the parameters
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer.")
        self.settings = settings
        # Initialize mixing probabilities and structural parameters
        self.π = np.random.dirichlet(α)  # Belief degree for each map
        self.ρ = [np.random.dirichlet(np.ones(N)) for _ in range(len(α))]
        # Initialize private and auxiliary variables
        self._ξ: Observation = np.zeros(N, dtype=np.float32)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the model (k, N)."""
        return len(self.π), len(self.ρ[0])

    def __call__(  # pylint: disable=too-many-arguments
        self,
        ξ: Observation,  # ξ(t-1)
        x: Item,  # x(t-1)  TODO: This variable seems to not be used
        y: Sequence,  # y(t-1)
        Θ: List[CognitiveMap],
        z: Optional[Mixing] = None,
    ) -> Tuple[Item, Sequence, Mixing, np.int64]:
        """Inference function, returns predicted next item and sequence."""
        # Update mixing probabilities or use the provided ones
        z = self.π if z is None else eq.z(θ, )
        k = np.argmax(z)  # Get the best map index
        x = self._estimate_item(ξ, y, Θ[k])  # Predict item code
        # TODO: Check: ξ = pred_observation(x)  # Predict the next observation
        y = self._estimate_sequence(ξ, y, Θ[k])  # Update the sequence
        return x, y, z, k


def train(model: HierarchicalGenerativeModel, episode: List[List[Observation]]):

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
