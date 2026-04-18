"""Core definitions for TEM backbones and related modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.modules.projection import ProjectionSettings


# =============================================================================
@dataclass(frozen=True)
class GridCodes:
    """Named container for the two TEM grid-pathway codes.

    Attributes:
        prior: Prior grid code derived from path integration of the previous step's grid code and
            executed action.
        post: Posterior grid code derived from the current step's sensory-grounded place code and
            the previous step's grid code.

    Shape conventions:
        Each tensor has shape ``(batch, grid_dim)`` where ``grid_dim`` is the sum of MEC grid code
        sizes across all frequency scales.

    """

    post: Tensor
    prior: Tensor


# =============================================================================
@dataclass(frozen=True)
class PlaceCodes:
    """Named container for the three TEM place-pathway codes.

    Attributes:
        inference: Posterior place code grounded by the current sensory observation (HPC inference).
        ancestral: Structural prior place code derived from the grid prior path-integration (HPC generative).
        retrieved: Corrected-grid generative place code (HPC generative from post-corrected grid). ``None``
            when sensory recall is disabled.
        sensory: Sensory-cued place retrieval from the previous memory state, used as the
            ``PLACE_SENSORY_RELATION`` target. ``None`` when ``enable_sensory_recall=False``.

    Shape conventions:
        Each non-None tensor has shape ``(batch, place_dim)`` where ``place_dim`` is the sum
        of HPC attractor pattern sizes across all frequency scales.
    """

    inference: Tensor
    ancestral: Tensor
    retrieved: Optional[Tensor] = None
    sensory: Optional[Tensor] = None


# =============================================================================
class TEMProjectionSettings(BaseModel, extra="forbid", strict=False):
    """Inter-region projection settings for TEM backbones."""

    lec_to_hpc: ProjectionSettings = Field(
        default_factory=lambda: ProjectionSettings(mode="tiling", learnable=False),
        description="Projection settings mapping LEC features into hippocampal query space.",
    )
    mec_to_hpc: ProjectionSettings = Field(
        default_factory=lambda: ProjectionSettings(mode="low_rank", learnable=False, rank=[10, 10, 8, 6, 6]),
        description="Projection settings mapping MEC codes into hippocampal query space.",
    )


# =============================================================================
__all__ = ["TEMProjectionSettings", "GridCodes", "PlaceCodes"]
