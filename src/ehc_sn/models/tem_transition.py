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

    def to_hpc_step_input(self, state: HPCState) -> HPCStepInput:
        """Convert the resolved MEC/HPC handoff into phase-2 HPC inputs."""
        return HPCStepInput(
            state=state,
            sensory=self.sensory,
            grid_query_prior=self.grid_query_prior,
            grid_query_posterior=self.grid_query_posterior,
        )


__all__ = ["TEMMemoryTransition"]
