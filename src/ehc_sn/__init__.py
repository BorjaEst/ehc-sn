"""Sequential Navigation (SN) module for the Entorhinal–Hippocampal circuit (EHC)"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from ehc_sn import equations as eq
from ehc_sn.equations import Item, Mixing, Observation, Sequence
from ehc_sn.utils import CognitiveMap
from numpy.typing import NDArray

# pylint: disable=non-ascii-name
# pylint: disable=invalid-name
# pylint: disable=too-few-public-methods


Trajectory = List[Observation]  # List of observations
Episode = List[Trajectory]  # List of trajectories


@dataclass
class HGModelSettings:
    """The parameters settings class for model configuration."""

    δ: float = 0.7  # Discount factor for sequence
    τ: float = 0.9  # Exponential decay for mixing categorical distribution
    c: float = 0.4  # Velocity rate for item code update

    def validate(self):
        """Validate the setting parameters."""
        if not isinstance(self.δ, float) or not 0 <= self.δ <= 1:
            raise ValueError(f"δ: {self.δ}. Must be a float between 0 and 1.")
        if not isinstance(self.τ, float) or not 0 <= self.τ <= 1:
            raise ValueError(f"τ: {self.τ}. Must be a float between 0 and 1.")
        if not isinstance(self.c, float) or not 0 <= self.c <= 1:
            raise ValueError(f"c: {self.c}. Must be a float between 0 and 1.")


class HierarchicalGenerativeModel:
    """Hierarchical generative model for sequential navigation."""

    def __init__(self, α: List[float], N: int, settings: HGModelSettings):
        # Store and validate the parameters
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer.")
        # Store the parameters of the model configuration
        self.settings = settings
        # Initialize mixing probabilities and structural parameters
        self.π = np.random.dirichlet(α)  # Mixing hyperparameter
        self.ρ = [np.random.dirichlet(np.ones(N)) for _ in range(len(α))]

    @property
    def settings(self) -> HGModelSettings:
        """Return the parameters of the model."""
        return self.__settings

    @settings.setter
    def settings(self, settings: HGModelSettings):
        """Set the settings of the model."""
        if not isinstance(settings, HGModelSettings):
            raise ValueError("parameters must be instance of HGModelParams.")
        settings.validate()  # Validate the parameters
        self.__settings = settings

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the model (k, N)."""
        return len(self.π), len(self.ρ[0])

    def inference(  # pylint: disable=too-many-arguments
        self,
        ξ: Observation,
        y: Sequence,
        Θ: List[CognitiveMap],
        z: Optional[Mixing] = None,
    ) -> Tuple[Item, Sequence, NDArray[np.float64], np.int64]:
        """Inference function, returns predicted next item and sequence."""
        τ = self.settings.τ  # Extract exponential decay for mixing
        z = self.π if z is None else eq.mixing(y, Θ, z, τ)
        k = z.argmax()  # Convert to list and get best
        x = eq.item(ξ, y, Θ[k])  # Predict item code
        y = eq.sequence(ξ, y, Θ[k])  # Update the sequence
        return x, y, z / z.sum(), k  # z ~ Cat(π)

    def learning(  # pylint: disable=too-many-arguments
        self,
        episode: Episode,
        γ: float = 0.1,
        λ: float = 0.1,
    ) -> None:
        """Learning function for the model."""
        _, N = self.shape  # Get the number of items
        z = None  # Initialize mixing probabilities
        for trajectory in episode:
            Θ = self.maps_inference()  # Priority maps
            x, y = np.zeros(N), np.zeros(N)  # First item and sequence
            for ξ in trajectory:
                x, y, z, k = self.inference(ξ, y, Θ)
                self.π[k] = (1 - γ) * self.π[k] + z[k]  # Eq. (11)
                self.ρ[k] = self.ρ[k] + z[k] * y  # Eq. (12)
                ξ_i = np.argmax(ξ)  # Get the observation index
                Θ[k].θ = (1 - λ) * Θ[k].θ - kron_delta(ξ_i, np.log(x))  # Eq. (13)
        return Θ  # Return the updated prioritized maps

    def maps_inference(self):
        """Generates a list of CognitiveMap instances."""
        return [CognitiveMap(ρ) for ρ in self.ρ]  # Priority maps
