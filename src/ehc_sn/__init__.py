"""Sequential Navigation (SN) module for the Entorhinal–Hippocampal circuit (EHC)"""

from typing import Any, List, Optional, Tuple

import numpy as np
from ehc_sn import equations as eq
from ehc_sn.config import HGMSettings
from ehc_sn.equations import Item, Mixing, Observation, Sequence
from ehc_sn.equations import get_item as item  # noqa: F401
from ehc_sn.equations import get_sequence as sequence  # noqa: F401
from ehc_sn.utils import CognitiveMap
from numpy.typing import NDArray

# pylint: disable=non-ascii-name
# pylint: disable=invalid-name
# pylint: disable=too-few-public-methods


Trajectory = List[Observation]  # List of observations
Episode = List[Trajectory]  # List of trajectories


class HierarchicalGenerativeModel:
    """Hierarchical generative model for sequential navigation."""

    def __init__(
        self,
        α: List[float],  # Mixing hyperparameters
        N: int,  # Number of items in the maps
        settings: Optional[HGMSettings] = None,
    ):
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer.")
        if settings and not isinstance(settings, HGMSettings):
            raise ValueError(f"Settings must be {HGMSettings}.")
        self.settings = settings or HGMSettings()  # type: ignore
        ρh, size = np.ones(N), len(α)
        self.π = np.random.dirichlet(α)  # Mixing hyperparameter
        self.ρ = [np.random.dirichlet(ρh) for _ in range(size)]

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the model (k, N)."""
        return len(self.π), len(self.ρ[0])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the model for inference."""
        return inference(self, *args, **kwargs)


def inference(  # pylint: disable=too-many-arguments
    model: HierarchicalGenerativeModel,
    ξ: Observation,
    y: Sequence,
    Θ: List[CognitiveMap],
    z: Optional[Mixing] = None,
) -> Tuple[Item, Sequence, NDArray[np.float64], np.int64]:
    """Inference function, returns predicted next item and sequence."""
    τ = model.settings.τ  # Extract exponential decay for mixing
    z = model.π if z is None else eq.mixing(y, Θ, z, τ)
    k = z.argmax()  # Convert to list and get best
    x = eq.item(ξ, y, Θ[k])  # Predict item code
    y = eq.sequence(ξ, y, Θ[k])  # Update the sequence
    return x, y, z / z.sum(), k  # z ~ Cat(π)


def learning(  # pylint: disable=too-many-arguments
    model: HierarchicalGenerativeModel,
    episode: Episode,
    γ: float = 0.1,
    λ: float = 0.1,
) -> None:
    """Learning function for the model."""
    _, N = model.shape  # Get the number of items
    z = None  # Initialize mixing probabilities
    for trajectory in episode:
        Θ = model.maps_inference()  # Priority maps
        x, y = np.zeros(N), np.zeros(N)  # First item and sequence
        for ξ in trajectory:
            x, y, z, k = model.inference(ξ, y, Θ)
            model.π[k] = (1 - γ) * self.π[k] + z[k]  # Eq. (11)
            model.ρ[k] = model.ρ[k] + z[k] * y  # Eq. (12)
            ξ_i = np.argmax(ξ)  # Get the observation index
            Θ[k].θ = (1 - λ) * Θ[k].θ - kron_delta(ξ_i, np.log(x))  # Eq. (13)
    return Θ  # Return the updated prioritized maps
