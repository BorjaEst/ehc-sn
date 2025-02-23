"""Sequential Navigation (SN) for the Entorhinal–Hippocampal circuit (EHC)"""

from typing import Any, Optional, Tuple

import numpy as np
from ehc_sn import equations as eq
from ehc_sn.config import HGMSettings
from ehc_sn.equations import Item, Map, Mixing, Observation, Sequence
from ehc_sn.equations import get_item as item  # noqa: F401
from ehc_sn.equations import get_sequence as sequence  # noqa: F401
from numpy.typing import NDArray

# pylint: disable=non-ascii-name
# pylint: disable=invalid-name
# pylint: disable=too-few-public-methods


Trajectory = list[Observation]  # List of observations
Episode = list[Trajectory]  # List of trajectories


class HierarchicalGenerativeModel:
    """Hierarchical generative model for sequential navigation."""

    def __init__(
        self,
        α: list[float],  # Mixing hyperparameters
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
        self.v = np.zeros(N)  # Velocity for the model

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the model (k, N)."""
        return len(self.π), len(self.ρ[0])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the model for inference."""
        return inference(self, *args, **kwargs)

    def sample_maps(self) -> list[Map]:
        """Return a list of generated cognitive maps."""
        return [ρ / ρ.sum() for ρ in self.ρ]  # Cat(ρ)


def inference(  # pylint: disable=too-many-arguments
    model: HierarchicalGenerativeModel,
    ξ: Observation,
    y: Sequence,
    Θ: list[Map],
    z: Optional[Mixing] = None,
) -> Tuple[Item, Sequence, NDArray[np.float64], np.int64]:
    """Inference function, returns predicted next item and sequence."""
    δ, τ, c = model.settings.δ, model.settings.τ, model.settings.c
    z = model.π if z is None else eq.mixing(y, Θ, z, τ)
    k = z.argmax()  # Convert to list and get best
    x = eq.item(ξ, y, Θ[k], model.v, c)  # Predict item code
    model.v = eq.observation(x) - ξ  # Update the velocity
    y = eq.sequence(ξ, y, Θ[k], δ)  # Update the sequence
    return x, y, z / z.sum(), k  # z ~ Cat(π)


def learning(  # pylint: disable=too-many-arguments
    model: HierarchicalGenerativeModel,
    episode: Episode,
    γ: float = 0.1,
    λ: float = 0.1,
) -> list[Map]:
    """Learning function for the model."""
    n_clusters, N = model.shape  # Get the number of items
    p_maps = [np.zeros(N) for _ in range(n_clusters)]
    z = None  # Initialize mixing probabilities
    for trajectory in episode:
        Θ = model.sample_maps()  # Get the cognitive maps
        x, y = np.zeros(N), np.zeros(N)  # First item and sequence
        model.v = np.zeros(N)  # Reset the velocity
        for ξ in trajectory:
            x, y, z, k = inference(model, ξ, y, Θ)
            model.π[k] = eq.π_update(model.π[k], z[k], γ)
            model.ρ[k] = eq.ρ_update(model.ρ[k], z[k], y)
            p_maps[k] = eq.pmap_update(p_maps[k], ξ, x, λ)
    return p_maps  # Return the updated priority maps
