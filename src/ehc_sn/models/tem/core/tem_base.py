"""Core definitions for TEM backbones and related modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.modules.hpc import HPCState, HPCTransition, SensoryReadResult
from ehc_sn.modules.hpc.query_policy import CueRead, MemoryRead
from ehc_sn.modules.projection import ProjectionSettings
from ehc_sn.types import AbstractLocation, GroundedLocation


# =============================================================================
@dataclass(frozen=True)
class GridCodes:
    """Named container for the two TEM grid-pathway codes.

    Attributes:
        posterior: Posterior grid code derived from the current step's sensory-grounded place code and
            the previous step's grid code.
        prior: Prior grid code derived from path integration of the previous step's grid code and
            executed action.

    Shape conventions:
        Each code is a multi-scale bundle with length ``n_freq``. Every tensor in the bundle has
        shape ``(batch, grid_dim_f)`` for its frequency-specific MEC grid width.

    """

    posterior: AbstractLocation
    prior: AbstractLocation


# =============================================================================
@dataclass(frozen=True)
class PlaceCodes:
    """Named container for the three TEM place-pathway codes.

    Attributes:
        posterior: Posterior place code grounded by the current sensory observation (HPC inference).
        prior: Structural prior place code derived from the grid prior path-integration (HPC generative).
        retrieved: Corrected-grid generative place code (HPC generative from post-corrected grid). ``None``
            when sensory recall is disabled.
        sensory: Sensory-cued place retrieval from the previous memory state, used as the
            ``PLACE_SENSORY_RELATION`` target. ``None`` when ``enable_sensory_recall=False``.

    Shape conventions:
        Each non-None code is a multi-scale bundle with length ``n_freq``. Every tensor in the
        bundle has shape ``(batch, place_dim_f)`` for its frequency-specific hippocampal width.
    """

    posterior: GroundedLocation
    prior: GroundedLocation
    retrieved: Optional[GroundedLocation] = None
    sensory: Optional[GroundedLocation] = None


# =============================================================================
@dataclass(frozen=True)
class PredCodes:
    """ """  # TODO: Docstring for PredCodes.

    inference: GroundedLocation
    ancestral: GroundedLocation
    retrieved: Optional[GroundedLocation] = None


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
__all__ = ["TEMProjectionSettings", "GridCodes", "PlaceCodes", "PredCodes"]
