from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor

from ehc_sn.modules.hpc import HPCState, HPCTransition, SensoryReadResult
from ehc_sn.modules.hpc.query_policy import CueRead, MemoryRead


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


__all__ = ["TEMTransitionPlan"]


# =================================================================================================
@dataclass(frozen=True)
class PredCodes:
    """Model-native prediction codes for the three TEM observation pathways."""

    inference: list[Tensor]
    retrieved: Optional[list[Tensor]]
    ancestral: list[Tensor]


@dataclass(frozen=True)
class GridCodes:
    """Model-native grid codes from MEC (posterior and prior)."""

    posterior: list[Tensor]
    prior: list[Tensor]


@dataclass(frozen=True)
class PlaceCodes:
    """Model-native place codes from HPC (posterior, prior, retrieved, sensory)."""

    posterior: list[Tensor]
    prior: list[Tensor]
    retrieved: Optional[list[Tensor]]
    sensory: Optional[list[Tensor]]


__all__ = ["TEMTransitionPlan", "PredCodes", "GridCodes", "PlaceCodes"]
