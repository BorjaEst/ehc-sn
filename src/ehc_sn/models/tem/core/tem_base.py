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


@dataclass(frozen=True)
class TEMTransitionPlan:
    """Named MEC-to-HPC handoff for one TEM transition.

    TEM computes the observation-cued sensory phase first, then MEC resolves
    the prior and posterior grid codes, and only then can HPC complete its
    retrieval/generative/write phase. This object keeps that handoff explicit
    and named at the model layer.
    """

    sensory: SensoryReadResult
    grid_prior: list[Tensor]
    grid_query_prior: list[Tensor]
    grid_post: list[Tensor]
    grid_query_posterior: list[Tensor]
    sensory_family: str = "x"
    generative_family: str = "g"
    generative_read: MemoryRead = field(default_factory=lambda: CueRead(kind="cue", cue="g"))
    named_writes: dict[str, list[Tensor]] = field(default_factory=dict)

    def to_hpc_transition(self, state: HPCState) -> HPCTransition:
        """Convert the resolved MEC/HPC handoff into phase-2 HPC inputs."""
        return HPCTransition(
            state=state,
            sensory=self.sensory,
            prior_read_cues=self.sensory.read_cues.with_family(self.generative_family, self.grid_query_prior),
            prior_read=self.generative_read,
            posterior_read_cues=self.sensory.read_cues.with_family(self.generative_family, self.grid_query_posterior),
            posterior_read=self.generative_read,
            inference_sensory_query=self.sensory.read_cues.require(self.sensory_family),
            inference_structural_query=self.grid_query_posterior,
            named_writes=self.named_writes,
        )


# =============================================================================
__all__ = ["TEMProjectionSettings", "TEMTransitionPlan", "GridCodes", "PlaceCodes", "PredCodes"]
