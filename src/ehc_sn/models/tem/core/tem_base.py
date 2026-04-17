"""Core definitions for TEM backbones and related modules."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.modules.projection import ProjectionSettings

# =============================================================================
# Community-standard map-style batch: plain dict returned by MazeDataset / DataLoader.
ObsLogits = tuple[Tensor, Tensor, Tensor]  # (inference, retrieved, ancestral)
GridCodes = tuple[Tensor, Tensor]  # (posterior, prior)
PlaceCodes = tuple[Tensor, Tensor, Optional[Tensor]]  # (posterior, prior, sensory-cued retrieval)


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
__all__ = ["TEMProjectionSettings", "ObsLogits", "GridCodes", "PlaceCodes"]
