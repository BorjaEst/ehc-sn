"""Shared orchestration and state contracts for hippocampal memory modules."""

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
from ehc_sn.modules.hpc.query_policy import (
    QueryPolicySettings,
    RetrievalEvidence,
    RetrievalTarget,
    RoleQueryPolicySettings,
    build_query_policy,
)
from ehc_sn.types import Device, Dtype, LocationBelief, MemoryEntry, MemoryState, RetrievalRole
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class HPCCommonSettings(BaseModel, extra="forbid"):
    """Settings shared by all hippocampal memory implementations.

    These fields are consumed either by ``HPCBase`` directly or by both
    concrete memory implementations.
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
    query_policy: QueryPolicySettings = Field(
        default_factory=RoleQueryPolicySettings,
        description="Policy used to resolve a retrieval query from the available x and g cues.",
    )


# =================================================================================================
@dataclass
class HPCState(DetachMixin):
    """Container for HPC state.

    Attributes:
        grounded_belief: A ``LocationBelief`` over grounded location codes.
        memory: Memory state with named g-cued and x-cued retrieval entries.
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
        """Return the two cue-indexed memory entries for this state."""
        return self._memory

    def replace_rows(  # --------------------------------------------------------------------------
        self, flag: Tensor, fresh: "HPCState", *,
        merge_memory_rows: Callable[[Tensor, MemoryEntry, MemoryEntry], MemoryEntry],
        common_memory: bool = False,
    ) -> "HPCState":  # fmt: skip
        """Return a state where flagged rows are replaced from ``fresh``."""
        uncertainty = None
        if self.uncertainty is not None and fresh.uncertainty is not None:
            uncertainty = utils.merge_multiscale_rows(flag, self.uncertainty, fresh.uncertainty)

        merged_g_cued = merge_memory_rows(flag, self.memory.g_cued, fresh.memory.g_cued)
        if common_memory:
            merged_x_cued = merged_g_cued
        else:
            merged_x_cued = merge_memory_rows(flag, self.memory.x_cued, fresh.memory.x_cued)

        return self.new(
            cells=utils.merge_multiscale_rows(flag, self.cells, fresh.cells),
            uncertainty=uncertainty,
            memory=MemoryState(g_cued=merged_g_cued, x_cued=merged_x_cued),
        )


# =================================================================================================
@dataclass(frozen=True)
class HPCSensoryStepInput:
    """Phase-1 HPC inputs resolved before MEC posterior inference."""

    state: HPCState
    x_query: list[Tensor]
    g_query: Optional[list[Tensor]] = None
    use_x_cued_recall: bool = True


# =================================================================================================
@dataclass
class HPCSensoryStepOutput:
    """Phase-1 HPC outputs passed from TEM into MEC posterior inference."""

    x_query: list[Tensor]
    g_query: Optional[list[Tensor]]
    sensory_recall: Optional[list[Tensor]]


# =================================================================================================
@dataclass(frozen=True)
class HPCStepInput:
    """Phase-2 HPC inputs after MEC has resolved the posterior grid query."""

    state: HPCState
    sensory: HPCSensoryStepOutput
    grid_query_prior: list[Tensor]
    grid_query_posterior: list[Tensor]


# =================================================================================================
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

    Concrete classes implement memory representation, runtime, recall, update,
    and row-merge semantics through protected hooks. The base class owns
    grounded-location inference, semantic multi-scale IO, generative sampling,
    and the TEM-compatible two-phase step choreography.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: HPCCommonSettings, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Initialize HPC base."""
        super().__init__()
        self._config = config
        self._shape = list(config.shape)
        self._n_freq = len(config.shape)
        self.grounded_location = GroundLocation(self._shape, config.location, device=device, dtype=dtype)
        self.query_policy = build_query_policy(self._shape, config.query_policy, device=device, dtype=dtype)

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

    def _flatten_memory_code(  # ------------------------------------------------------------------
        self, code: list[Tensor],
    ) -> Tensor:  # fmt: skip
        """Return a flattened `(B, S)` view of a multi-scale memory-space code."""
        if len(code) != self.n_freq:
            raise ValueError(f"Expected {self.n_freq} frequency tensors, got {len(code)}.")

        batch_size: int | None = None
        for index, (tensor, width) in enumerate(zip(code, self.shape, strict=True)):
            if tensor.ndim != 2:
                raise ValueError(f"code[{index}] must be rank-2 `(B, {width})`, got shape {tuple(tensor.shape)}.")
            if int(tensor.shape[1]) != width:
                raise ValueError(f"code[{index}] must have width {width}, got {int(tensor.shape[1])}.")
            if batch_size is None:
                batch_size = int(tensor.shape[0])
            elif int(tensor.shape[0]) != batch_size:
                raise ValueError("All frequency tensors must have the same batch size.")

        return torch.cat(code, dim=1)

    def _unflatten_memory_code(  # ----------------------------------------------------------------
        self, flat_code: Tensor,
    ) -> list[Tensor]:  # fmt: skip
        """Return the multi-scale view of a flattened memory-space tensor `(B, S)`."""
        if flat_code.ndim != 2:
            raise ValueError(f"flat_code must be rank-2 `(B, S)`, got shape {tuple(flat_code.shape)}.")

        feature_dim = sum(self.shape)
        if int(flat_code.shape[1]) != feature_dim:
            raise ValueError(f"flat_code must have width {feature_dim}, got {int(flat_code.shape[1])}.")

        return list(torch.split(flat_code, split_size_or_sections=self.shape, dim=1))

    @abstractmethod
    def _init_memory_impl(  # ---------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> MemoryState:  # fmt: skip
        """Initialize the concrete memory representation for a batch."""

    @abstractmethod
    def _set_runtime_impl(  # ---------------------------------------------------------------------
        self, *, eta: float, hebbian_decay: float,
    ) -> None:  # fmt: skip
        """Apply runtime write parameters to the concrete memory system."""

    @abstractmethod
    def _recall_flat_impl(  # ---------------------------------------------------------------------
        self, query: Tensor, memory: MemoryEntry, *, role: RetrievalRole,
    ) -> Tensor:  # fmt: skip
        """Return a flattened recalled code with shape ``(B, S)``."""

    @abstractmethod
    def _update_memory_impl(  # -------------------------------------------------------------------
        self, memory: MemoryState, key: Tensor,
        g_value: Tensor, x_value: Optional[Tensor],
    ) -> MemoryState:  # fmt: skip
        """Write one TEM step into the concrete memory state.

        All tensors are flattened memory-space tensors with shape ``(B, S)``.
        """

    @abstractmethod
    def _merge_memory_rows_impl(self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry) -> MemoryEntry:
        """Merge representation-specific memory rows during partial reset."""

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int, *, memory: Optional[MemoryState] = None,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> HPCState:  # fmt: skip
        """Create an initial ``HPCState``."""
        p_init = [torch.zeros((batch_size, n), device=device) for n in self.shape]
        grounded_belief = LocationBelief(mean=p_init, uncertainty=None)
        memory = memory or self.init_memory(batch_size=batch_size, device=device)
        return HPCState(grounded_belief=grounded_belief, _memory=memory)

    def init_memory(  # ---------------------------------------------------------------------------
        self, batch_size: int, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> MemoryState:  # fmt: skip
        """Initialize the concrete memory state."""
        del dtype
        return self._init_memory_impl(batch_size=batch_size, device=device)

    def set_runtime(  # ---------------------------------------------------------------------------
        self, *, eta: float, hebbian_decay: float,
    ) -> None:  # fmt: skip
        """Apply runtime parameters required by the concrete memory system."""
        self._set_runtime_impl(eta=eta, hebbian_decay=hebbian_decay)

    def recall(  # --------------------------------------------------------------------------------
        self, *, x_query: Optional[list[Tensor]], g_query: Optional[list[Tensor]],
        state: HPCState, role: RetrievalRole,
    ) -> list[Tensor]:  # fmt: skip
        """Retrieve grounded-location code from the concrete memory representation."""
        memory = state.memory.for_role(role)
        evidence = self.compose_retrieval_evidence(
            x_query=x_query,
            g_query=g_query,
            memory=memory,
            role=role,
            target="grounded",
        )
        if evidence.mode != "anchor_query":
            raise TypeError(
                f"Base recall does not support retrieval evidence mode '{evidence.mode}'. Override recall in the concrete module."
            )
        query = evidence.anchor_query if evidence.anchor_query is not None else evidence.fallback_query
        if query is None:
            raise ValueError("Retrieval evidence must provide an anchor or fallback query for base recall.")
        recalled = self._recall_flat_impl(
            query,
            memory,
            role=role,
        )
        return self._unflatten_memory_code(recalled)

    def compose_retrieval_evidence(  # -----------------------------------------------------------
        self, *,
        x_query: Optional[list[Tensor]], g_query: Optional[list[Tensor]], memory: MemoryEntry,
        role: RetrievalRole, target: RetrievalTarget,
    ) -> RetrievalEvidence:  # fmt: skip
        """Compose structured retrieval evidence before backend-specific memory read."""
        return self.query_policy.compose_evidence(
            x_query=x_query,
            g_query=g_query,
            role=role,
            target=target,
            memory=memory,
        )

    def update(  # --------------------------------------------------------------------------------
        self, p_inf: list[Tensor], p_gen_gi: list[Tensor], p_xi: Optional[list[Tensor]],
        state: HPCState,
    ) -> HPCState:  # fmt: skip
        """Write one TEM step into the concrete memory state."""
        memory = self._update_memory_impl(
            state.memory,
            self._flatten_memory_code(p_inf),
            self._flatten_memory_code(p_gen_gi),
            None if p_xi is None else self._flatten_memory_code(p_xi),
        )
        return HPCState(state.grounded_belief, _memory=memory)

    def merge_memory_rows(  # ---------------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> MemoryEntry:  # fmt: skip
        """Merge concrete memory rows during partial reset."""
        return self._merge_memory_rows_impl(flag, current, fresh)

    def prepare_sensory_step(  # ------------------------------------------------------------------
        self, step_input: HPCSensoryStepInput,
    ) -> HPCSensoryStepOutput:  # fmt: skip
        """Resolve the phase-1 observation-cued recall used by MEC inference."""
        sensory_recall = None
        if step_input.use_x_cued_recall:
            sensory_recall = self.recall(
                x_query=step_input.x_query,
                g_query=step_input.g_query,
                state=step_input.state,
                role="inference",
            )
        return HPCSensoryStepOutput(
            x_query=step_input.x_query,
            g_query=step_input.g_query,
            sensory_recall=sensory_recall,
        )

    def step(  # ----------------------------------------------------------------------------------
        self, step_input: HPCStepInput,
    ) -> HPCStepOutput:  # fmt: skip
        """Run phase 2 of the TEM-compatible HPC transition."""
        state = step_input.state
        grid_prior_recall = self.recall(
            x_query=step_input.sensory.x_query,
            g_query=step_input.grid_query_prior,
            state=state,
            role="generative",
        )
        grid_posterior_recall = self.recall(
            x_query=step_input.sensory.x_query,
            g_query=step_input.grid_query_posterior,
            state=state,
            role="generative",
        )

        place_retrieved, state = self.generative(grid_posterior_recall, state)
        place_prior, state = self.generative(grid_prior_recall, state)
        place_post, state = self.inference(step_input.sensory.x_query, step_input.grid_query_posterior, state)
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


# =================================================================================================
__all__ = [
    "HPCCommonSettings", "HPCSensoryStepInput", "HPCSensoryStepOutput", "HPCBase",
    "HPCState", "HPCStepInput", "HPCStepOutput",
]  # fmt: skip
