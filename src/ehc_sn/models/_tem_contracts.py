from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ehc_sn.modules.hpc import HPCSensoryStepOutput, HPCState, HPCStepInput


@dataclass(frozen=True)
class TEMMemoryTransition:
    """Named MEC-to-HPC handoff for one TEM transition.

    TEM computes the observation-cued sensory phase first, then MEC resolves
    the prior and posterior grid codes, and only then can HPC complete its
    retrieval/generative/write phase. This object keeps that handoff explicit
    and named at the model layer.
    """

    sensory: HPCSensoryStepOutput
    grid_prior: list[Tensor]
    grid_query_prior: list[Tensor]
    grid_post: list[Tensor]
    grid_query_posterior: list[Tensor]
    sensory_family: str = "x"
    generative_family: str = "g"

    def to_hpc_step_input(self, state: HPCState) -> HPCStepInput:
        """Convert the resolved MEC/HPC handoff into phase-2 HPC inputs."""
        return HPCStepInput(
            state=state,
            sensory=self.sensory,
            prior_cues=self.sensory.cues.with_family(self.generative_family, self.grid_query_prior),
            prior_anchor_family=self.generative_family,
            posterior_cues=self.sensory.cues.with_family(self.generative_family, self.grid_query_posterior),
            posterior_anchor_family=self.generative_family,
            inference_sensory_query=self.sensory.cues.require(self.sensory_family),
            inference_structural_query=self.grid_query_posterior,
        )


__all__ = ["TEMMemoryTransition"]
