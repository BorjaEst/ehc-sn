from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from torch import Tensor

from ehc_sn.modules.hpc import HPCState, HPCTransition, SensoryReadResult
from ehc_sn.modules.hpc.query_policy import CueRead, MemoryRead

if TYPE_CHECKING:
    from ehc_sn.models.ehc.ehc_v1 import EHCState


@dataclass(frozen=True)
class EHCMemoryStep:
    """Memory-side diagnostics returned by one EHC forward step.

    The v1 contract keeps the TEM stage ordering intact and treats cortical
    context as an additional target-bank pathway. Bank ``"c"`` stores the
    reinstated cue ``c_mem`` only; the routed control cue ``c_use`` remains a
    transient control object and is not written back into memory.
    """

    c_prop: list[Tensor]
    c_mem: list[Tensor]
    c_use: list[Tensor]
    c_gate: Tensor
    grid_prior: list[Tensor]
    grid_post: list[Tensor]
    place_query_from_obs: list[Tensor]
    place_query_from_grid_prior: list[Tensor]
    place_query_from_grid_post: list[Tensor]
    place_sensory: Optional[list[Tensor]]
    place_prior: list[Tensor]
    place_retrieved: list[Tensor]
    place_post: list[Tensor]
    named_writes: dict[str, list[Tensor]]


@dataclass(frozen=True)
class EHCControlStep:
    """Control-side outputs returned by one EHC forward step."""

    z_H: Tensor
    theta_summary: Tensor
    internal_control_logits: Tensor
    motor_logits: Tensor
    reward_logits: Tensor


@dataclass(frozen=True)
class EHCOutput:
    """Structured one-step EHC model output.

    The model stops at producing memory diagnostics and control outputs. Any
    environment stepping or internal-cycle orchestration belongs to a later
    adapter/controller layer.
    """

    state: EHCState
    obs_logits: tuple[Tensor, Tensor, Tensor]
    memory: EHCMemoryStep
    control: EHCControlStep

    @property
    def grid_codes(self) -> tuple[list[Tensor], list[Tensor]]:
        """Return posterior/prior structural codes in TEM-compatible order."""
        return self.memory.grid_post, self.memory.grid_prior

    @property
    def place_codes(self) -> tuple[list[Tensor], list[Tensor], Optional[list[Tensor]]]:
        """Return posterior/prior/sensory place codes in TEM-compatible order."""
        return self.memory.place_post, self.memory.place_prior, self.memory.place_sensory


@dataclass(frozen=True)
class EHCTransitionPlan:
    """Named MEC-to-HPC handoff for one EHC transition.

    EHC v1 preserves the TEM phase ordering exactly. Cortical context only
    enters through typed read cues and named writes: ``"c"`` is a grounded
    target bank, not a symmetric source family, and ``named_writes["c"]`` must
    carry reinstated evidence rather than routed cortical proposals.
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


__all__ = ["EHCControlStep", "EHCMemoryStep", "EHCOutput", "EHCTransitionPlan"]
