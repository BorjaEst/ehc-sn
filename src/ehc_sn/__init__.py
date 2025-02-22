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

    def sample_maps(self) -> List[CognitiveMap]:
        """Return a list of generated cognitive maps."""
        return [CognitiveMap(ρ) for ρ in self.ρ]


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
    return x, y, z, k  # z ~ Cat(π)
    # return x, y, z / z.sum(), k  # z ~ Cat(π)


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
        Θ = model.sample_maps()  # Get the cognitive maps
        x, y = np.zeros(N), np.zeros(N)  # First item and sequence
        for ξ in trajectory:
            x, y, z, k = inference(model, ξ, y, Θ)
            model.π[k] = eq.π_update(model.π[k], z[k], γ)
            model.ρ[k] = eq.ρ_update(model.ρ[k], z[k], y)
    # TODO: Implement priority map
