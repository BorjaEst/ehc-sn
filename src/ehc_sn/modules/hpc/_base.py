from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.hpc.location import GroundLocation, GroundLocSettings
from ehc_sn.types import Device, Dtype, LocationBelief, MemoryEntry, MemoryState, OperationMode
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class HPCCommonSettings(BaseModel, extra="forbid"):
    """Settings shared by all hippocampal memory implementations.

    These fields are consumed either by ``HPCBase`` directly or by both
    concrete backend implementations.
    """

    shape: list[int] = Field(
        ...,
        min_length=1,
        description="Feature dimensionality per frequency module.",
    )
    common_memory: bool = Field(
        default=False,
        description="Whether the g-cued and x-cued memories share the same underlying store.",
    )
    do_sample: bool = Field(
        default=False,
        description="Whether to sample from grounded-location beliefs or use their means.",
    )
    location: GroundLocSettings = Field(
        default_factory=GroundLocSettings,
        description="Grounded-location inference module config.",
    )


# =================================================================================================
@dataclass
class HPCState(DetachMixin):
    """Container for HPC state.

    Attributes:
        grounded_belief: A ``LocationBelief`` over grounded location codes.
        memory: Backend-specific retrieval state with named g-cued and x-cued entries.
    """

    grounded_belief: LocationBelief
    _memory: MemoryState

    def new(  # -----------------------------------------------------------------------------------
        self, cells: list[Tensor], uncertainty: Optional[list[Tensor]], *,
        memory: Optional[MemoryState] = None,
    ) -> "HPCState":  # fmt: skip
        """Return a copy with updated grounded belief and optional memory."""
        return replace(
            self,
            grounded_belief=LocationBelief(mean=cells, uncertainty=uncertainty),
            _memory=self._memory if memory is None else memory,
        )

    @property
    def cells(self) -> list[Tensor]:
        """Return grounded location features."""
        return self.grounded_belief.mean

    @property
    def uncertainty(self) -> Optional[list[Tensor]]:
        """Return grounded location uncertainty."""
        return self.grounded_belief.uncertainty

    @property
    def memory(self) -> MemoryState:
        """Return backend-specific memory state for the two cue-indexed retrieval entries."""
        return self._memory

    def replace_rows(  # --------------------------------------------------------------------------
        self, flag: Tensor, fresh: "HPCState",
        *,
        merge_memory_rows: Callable[[Tensor, MemoryEntry, MemoryEntry], MemoryEntry],
        common_memory: bool = False,
    ) -> "HPCState":  # fmt: skip
        """Return a state where flagged rows are replaced from ``fresh``."""
        uncertainty = None
        if self.uncertainty is not None and fresh.uncertainty is not None:
            uncertainty = utils.merge_multiscale_rows(flag, self.uncertainty, fresh.uncertainty)

        merged_g_cued = merge_memory_rows(flag, self.memory.g_cued, fresh.memory.g_cued)
        merged_x_cued = (
            merged_g_cued
            if common_memory
            else merge_memory_rows(flag, self.memory.x_cued, fresh.memory.x_cued)
        )

        return self.new(
            cells=utils.merge_multiscale_rows(flag, self.cells, fresh.cells),
            uncertainty=uncertainty,
            memory=MemoryState(g_cued=merged_g_cued, x_cued=merged_x_cued),
        )


@dataclass(frozen=True)
class HPCSensoryStepInput:
    """Phase-1 HPC inputs resolved before MEC posterior inference."""

    state: HPCState
    sensory_query: list[Tensor]
    use_x_cued_recall: bool = True


@dataclass
class HPCSensoryStepOutput:
    """Phase-1 HPC outputs passed from TEM into MEC posterior inference."""

    sensory_query: list[Tensor]
    sensory_recall: Optional[list[Tensor]]


@dataclass(frozen=True)
class HPCStepInput:
    """Phase-2 HPC inputs after MEC has resolved the posterior grid query."""

    state: HPCState
    sensory: HPCSensoryStepOutput
    grid_query_prior: list[Tensor]
    grid_query_posterior: list[Tensor]


@dataclass
class HPCStepOutput:
    """Structured outputs for the second HPC transition phase."""

    sensory: HPCSensoryStepOutput
    grid_prior_recall: list[Tensor]
    grid_posterior_recall: list[Tensor]
    place_prior: list[Tensor]
    place_retrieved: list[Tensor]
    place_post: list[Tensor]
    state: HPCState


# =================================================================================================
class HPCBase(nn.Module, ABC):
    """Shared orchestration for hippocampal memory modules.

    Concrete classes own memory representation, recall, update, runtime, and
    row-merge semantics. The base class owns grounded-location inference,
    generative sampling, and the TEM-compatible two-phase step choreography.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: HPCCommonSettings,
        *, device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config
        self._shape = list(config.shape)
        self._n_freq = len(config.shape)
        self.grounded_location = GroundLocation(self._shape, config.location, device=device, dtype=dtype)

    @property
    def config(self) -> HPCCommonSettings:
        """Return hippocampal module config."""
        return self._config

    @property
    def shape(self) -> list[int]:
        """Return the grounded-location shape exposed to projections."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Return the number of HPC frequency modules."""
        return self._n_freq

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int, *,
        device: Optional[Device] = None, memory: Optional[MemoryState] = None,
    ) -> HPCState:  # fmt: skip
        """Create an initial ``HPCState``."""
        p_init = [torch.zeros((batch_size, n), device=device) for n in self.shape]
        grounded_belief = LocationBelief(mean=p_init, uncertainty=None)
        memory = memory or self.init_memory(batch_size=batch_size, device=device)
        return HPCState(grounded_belief=grounded_belief, _memory=memory)

    def prepare_sensory_step(self, step_input: HPCSensoryStepInput) -> HPCSensoryStepOutput:
        """Resolve the phase-1 observation-cued recall used by MEC inference."""
        sensory_recall = None
        if step_input.use_x_cued_recall:
            sensory_recall = self.recall(step_input.sensory_query, step_input.state, operation="inference")
        return HPCSensoryStepOutput(
            sensory_query=step_input.sensory_query,
            sensory_recall=sensory_recall,
        )

    def step(self, step_input: HPCStepInput) -> HPCStepOutput:
        """Run phase 2 of the TEM-compatible HPC transition."""
        state = step_input.state
        grid_prior_recall = self.recall(step_input.grid_query_prior, state, operation="generative")
        grid_posterior_recall = self.recall(step_input.grid_query_posterior, state, operation="generative")

        place_retrieved, state = self.generative(grid_posterior_recall, state)
        place_prior, state = self.generative(grid_prior_recall, state)
        place_post, state = self.inference(
            step_input.sensory.sensory_query, step_input.grid_query_posterior, state
        )
        state = self.update(place_post, place_retrieved, step_input.sensory.sensory_recall, state)

        return HPCStepOutput(
            sensory=step_input.sensory,
            grid_prior_recall=grid_prior_recall,
            grid_posterior_recall=grid_posterior_recall,
            place_prior=place_prior,
            place_retrieved=place_retrieved,
            place_post=place_post,
            state=state,
        )

    def generative(  # ----------------------------------------------------------------------------
        self, p_g: list[Tensor], state: HPCState,
    ) -> tuple[list[Tensor], HPCState]:  # fmt: skip
        """Return a grounded-location sample or mean from a provided distribution."""
        transition = LocationBelief(mean=p_g, uncertainty=state.uncertainty)
        p_gen = utils.sample_diag_gaussian(transition) if self.config.do_sample else transition.mean
        return p_gen, state.new(p_gen, state.uncertainty)

    def inference(  # -----------------------------------------------------------------------------
        self, x_: list[Tensor], g_: list[Tensor], state: HPCState,
    ) -> tuple[list[Tensor], HPCState]:  # fmt: skip
        """Infer grounded location from projected sensory and abstract features."""
        transition = self.grounded_location(x_, g_)
        p_inf = utils.sample_diag_gaussian(transition) if self.config.do_sample else transition.mean
        return p_inf, state.new(p_inf, transition.uncertainty)

    @abstractmethod
    def init_memory(
        self,
        batch_size: int,
        *,
        device: Optional[Device] = None,
    ) -> MemoryState:
        """Initialize backend-specific memory state."""

    @abstractmethod
    def set_runtime(self, *, eta: float, hebbian_decay: float) -> None:
        """Apply runtime parameters required by the concrete memory system."""

    @abstractmethod
    def recall(
        self,
        p_query: list[Tensor],
        state: HPCState,
        *,
        operation: OperationMode,
    ) -> list[Tensor]:
        """Retrieve grounded-location code from the backend-specific memory."""

    @abstractmethod
    def update(
        self,
        p_inf: list[Tensor],
        p_gen_gi: list[Tensor],
        p_xi: Optional[list[Tensor]],
        state: HPCState,
    ) -> HPCState:
        """Write one TEM step into the backend-specific memory state."""

    @abstractmethod
    def merge_memory_rows(self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry) -> MemoryEntry:
        """Merge backend-specific memory rows during partial reset."""


# =================================================================================================
__all__ = [
    "HPCCommonSettings",
    "HPCState",
    "HPCSensoryStepInput",
    "HPCSensoryStepOutput",
    "HPCStepInput",
    "HPCStepOutput",
    "HPCBase",
]
