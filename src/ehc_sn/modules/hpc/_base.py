"""Shared state, operators, and TEM transition choreography for HPC memory.

This module defines the stable contracts shared by all hippocampal memory
implementations in the repo: configuration, recurrent state, phase-1 sensory
reads, phase-2 transition payloads, and the common orchestration implemented by
``HPCBase``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.hpc.location import PlaceInference, PlaceInferenceSettings
from ehc_sn.modules.hpc.query_policy import MemoryRead, PreparedCueRead, PreparedRead, ReadCues, build_read_composer
from ehc_sn.types import Device, Dtype, LocationBelief, MemoryEntry, MemoryState, RetrievalRole
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class HPCCommonSettings(BaseModel, extra="forbid"):
    """Settings shared by all hippocampal memory implementations.

    These fields define the grounded-location feature layout exposed to the
    rest of TEM, whether cue-specific memories are shared, the sampling policy
    applied to grounded-location beliefs, and the nested configuration for the
    grounded-location inference module.
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
    location: PlaceInferenceSettings = Field(
        default_factory=PlaceInferenceSettings,
        description="Grounded-location inference module config.",
    )


# =================================================================================================
@dataclass
class HPCState(DetachMixin):
    """Recurrent HPC state carried between TEM steps.

    Attributes:
        grounded_belief: Current grounded-location belief over place-like codes.
        memory: Backend-specific memory state with cue-indexed retrieval
            entries.
    """

    grounded_belief: LocationBelief
    _memory: MemoryState

    def new(  # -----------------------------------------------------------------------------------
        self, cells: list[Tensor], uncertainty: Optional[list[Tensor]], *,
        memory: Optional[MemoryState] = None,
    ) -> "HPCState":  # fmt: skip
        """Return a copy with updated grounded belief and optional memory.

        Args:
            cells: Grounded-location mean per frequency, each tensor shaped
                ``(B, N_f)``.
            uncertainty: Optional grounded-location uncertainty per frequency.
            memory: Optional replacement memory state. When omitted, the current
                memory is preserved.

        Returns:
            A new ``HPCState`` with the requested fields updated.
        """
        return replace(
            self,
            grounded_belief=LocationBelief(mean=cells, uncertainty=uncertainty),
            _memory=self._memory if memory is None else memory,
        )

    @property
    def cells(self) -> list[Tensor]:
        """Return grounded-location means per frequency."""
        return self.grounded_belief.mean

    @property
    def uncertainty(self) -> Optional[list[Tensor]]:
        """Return grounded-location uncertainty per frequency."""
        return self.grounded_belief.uncertainty

    @property
    def memory(self) -> MemoryState:
        """Return the cue-indexed memory entries carried by this state."""
        return self._memory

    def replace_rows(  # --------------------------------------------------------------------------
        self, flag: Tensor, fresh: "HPCState", *,
        merge_memory_rows: Callable[[Tensor, MemoryEntry, MemoryEntry], MemoryEntry],
        common_memory: bool = False,
    ) -> "HPCState":  # fmt: skip
        """Return a state where flagged batch rows are replaced from ``fresh``.

        This owner-internal helper supports module reset logic by row-wise
        merging grounded-location beliefs and backend-specific memory entries.
        """
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
class SensoryRead:
    """Operator inputs for the phase-1 sensory-cued memory read.

    Phase 1 resolves observation-driven recall before MEC performs its
    correction step.
    """

    state: HPCState
    read_cues: ReadCues
    read: MemoryRead
    enable_sensory_recall: bool = True


# =================================================================================================
@dataclass
class SensoryReadResult:
    """Results produced by the phase-1 sensory-cued memory read.

    ``recall`` is optional so TEM can disable sensory recall while preserving a
    uniform handoff contract.
    """

    read_cues: ReadCues
    recall: Optional[list[Tensor]]


# =================================================================================================
@dataclass(frozen=True)
class WritePayload:
    """Projected values written to hippocampal memory for one TEM step.

    ``generative`` is the recalled place code used for the generative memory
    entry, ``inference`` is the optional sensory-driven place code used for the
    inference entry, and ``named_writes`` carries any additional backend-
    specific bank writes.
    """

    generative: list[Tensor]
    inference: Optional[list[Tensor]]
    named_writes: dict[str, list[Tensor]] = field(default_factory=dict)


# =================================================================================================
@dataclass(frozen=True)
class HPCTransition:
    """Inputs for the full phase-2 hippocampal transition.

    The model layer assembles this payload after phase-1 sensory recall and MEC
    prior/posterior updates are available.
    """

    state: HPCState
    sensory: SensoryReadResult
    prior_read_cues: ReadCues
    prior_read: MemoryRead
    posterior_read_cues: ReadCues
    posterior_read: MemoryRead
    inference_sensory_query: list[Tensor]
    inference_structural_query: list[Tensor]
    named_writes: dict[str, list[Tensor]] = field(default_factory=dict)


# =================================================================================================
@dataclass
class HPCTransitionResult:
    """Outputs produced by the full phase-2 hippocampal transition.

    The result exposes both the intermediate recalls used for TEM diagnostics
    and the updated ``HPCState`` carried into the next timestep.
    """

    sensory: SensoryReadResult
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
        """Initialize the shared HPC orchestration layer.

        Args:
            config: Shared HPC configuration.
            device: Optional device used when allocating submodules.
            dtype: Optional dtype used when allocating submodules.
        """
        super().__init__()
        self._config = config
        self._shape = list(config.shape)
        self._n_freq = len(config.shape)
        self.place_inference = PlaceInference(self._shape, config.location, device=device, dtype=dtype)
        self.read_composer = build_read_composer(self._shape, device=device, dtype=dtype)

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
        self, batch_size: int, *, memory: Optional[MemoryState] = None,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> HPCState:  # fmt: skip
        """Create an initial ``HPCState`` for one batch.

        Args:
            batch_size: Number of batch rows carried by the returned state.
            memory: Optional preallocated memory state to reuse.
            device: Optional device for newly allocated tensors.
            dtype: Unused placeholder kept for API parity with other modules.

        Returns:
            A fresh state whose grounded-location mean is zero-initialized.
        """
        p_init = [torch.zeros((batch_size, n), device=device) for n in self.shape]
        grounded_belief = LocationBelief(mean=p_init, uncertainty=None)
        memory = memory or self.init_memory(batch_size=batch_size, device=device)
        return HPCState(grounded_belief=grounded_belief, _memory=memory)

    def reset_state(  # ---------------------------------------------------------------------------
        self, state: HPCState, reset_flag: Tensor,
    ) -> HPCState:  # fmt: skip
        """Reset flagged HPC rows to a fresh episode state.

        Args:
            state: Current HPC state.
            reset_flag: Boolean / 0-1 tensor of shape ``(B,)`` indicating
                which rows should be reset.

        Returns:
            New state with flagged rows replaced by fresh initialization.
        """
        device = state.cells[0].device
        reset_flag = reset_flag.to(device=device, dtype=torch.bool).view(-1)
        if not torch.any(reset_flag):
            return state

        fresh = self.init_state(int(reset_flag.shape[0]), device=device, memory=None)
        return state.replace_rows(
            reset_flag,
            fresh,
            merge_memory_rows=self.merge_memory_rows,
            common_memory=self.config.common_memory,
        )

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

    def init_memory(  # ---------------------------------------------------------------------------
        self, batch_size: int, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> MemoryState:  # fmt: skip
        """Initialize the concrete memory state for one batch."""
        del dtype
        return self._init_memory_impl(batch_size=batch_size, device=device)

    @abstractmethod
    def _init_memory_impl(  # ---------------------------------------------------------------------
        self, batch_size: int, *, device: Optional[Device] = None,
    ) -> MemoryState:  # fmt: skip
        """Initialize the concrete memory representation for a batch."""

    def set_runtime(  # ---------------------------------------------------------------------------
        self, *, eta: float, hebbian_decay: float,
    ) -> None:  # fmt: skip
        """Apply runtime parameters required by the concrete memory system.

        Some implementations consume both values directly, while others ignore
        them to preserve a uniform model-level contract.
        """
        self._set_runtime_impl(eta=eta, hebbian_decay=hebbian_decay)

    @abstractmethod
    def _set_runtime_impl(  # ---------------------------------------------------------------------
        self, *, eta: float, hebbian_decay: float,
    ) -> None:  # fmt: skip
        """Apply runtime write parameters to the concrete memory system."""

    def recall(  # --------------------------------------------------------------------------------
        self, *,
        read_cues: ReadCues, state: HPCState, role: RetrievalRole, read: MemoryRead,
    ) -> list[Tensor]:  # fmt: skip
        """Retrieve a grounded-location code from the concrete memory system.

        The base implementation supports cue-resolved reads only. Implementations
        that support richer evidence payloads may override this method.
        """
        memory = state.memory.for_role(role)
        prepared_read = self.prepare_read(read_cues=read_cues, read=read)
        if not isinstance(prepared_read, PreparedCueRead):
            raise TypeError(f"{type(self).__name__} only supports resolved read requests.")
        recalled = self._recall_flat_impl(prepared_read.query, memory, role=role)
        return self._unflatten_memory_code(recalled)

    @abstractmethod
    def _recall_flat_impl(  # ---------------------------------------------------------------------
        self, query: Tensor, memory: MemoryEntry, *, role: RetrievalRole,
    ) -> Tensor:  # fmt: skip
        """Return a flattened recalled code with shape ``(B, S)``."""

    def update(  # --------------------------------------------------------------------------------
        self, key: list[Tensor], write: WritePayload,
        state: HPCState,
    ) -> HPCState:  # fmt: skip
        """Write one TEM step into the concrete memory state.

        Args:
            key: Multi-frequency grounded-location key used to index memory.
            write: Flattenable values to write into the backend-specific memory.
            state: Current HPC state whose memory will be updated.

        Returns:
            A new ``HPCState`` with unchanged grounded belief and updated memory.
        """
        named_writes = {name: self._flatten_memory_code(value) for name, value in write.named_writes.items()}
        memory = self._update_memory_impl(
            state.memory,
            self._flatten_memory_code(key),
            self._flatten_memory_code(write.generative),
            None if write.inference is None else self._flatten_memory_code(write.inference),
            named_writes,
        )
        return HPCState(state.grounded_belief, _memory=memory)

    @abstractmethod
    def _update_memory_impl(  # -------------------------------------------------------------------
        self, memory: MemoryState, key: Tensor,
        g_value: Tensor, x_value: Optional[Tensor], named_writes: dict[str, Tensor],
    ) -> MemoryState:  # fmt: skip
        """Write one TEM step into the concrete memory state.

        All tensors are flattened memory-space tensors with shape ``(B, S)``.
        """

    def merge_memory_rows(  # ---------------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> MemoryEntry:  # fmt: skip
        """Merge concrete memory rows during partial reset."""
        return self._merge_memory_rows_impl(flag, current, fresh)

    @abstractmethod
    def _merge_memory_rows_impl(  # ---------------------------------------------------------------
        self, flag: Tensor, current: MemoryEntry, fresh: MemoryEntry,
    ) -> MemoryEntry:  # fmt: skip
        """Merge representation-specific memory rows during partial reset."""

    def prepare_read(  # -------------------------------------------------------------------------
        self, *,
        read_cues: ReadCues, read: MemoryRead,
    ) -> PreparedRead:  # fmt: skip
        """Compose structured retrieval evidence before backend-specific recall."""
        return self.read_composer.compose(read_cues=read_cues, read=read)

    def read_sensory(  # -------------------------------------------------------------------------
        self, sensory_read: SensoryRead,
    ) -> SensoryReadResult:  # fmt: skip
        """Resolve the phase-1 observation-cued recall used by MEC inference."""
        sensory_recall = None
        if sensory_read.enable_sensory_recall:
            sensory_recall = self.recall(
                read_cues=sensory_read.read_cues,
                state=sensory_read.state,
                role="inference",
                read=sensory_read.read,
            )
        return SensoryReadResult(read_cues=sensory_read.read_cues, recall=sensory_recall)

    def transition(  # ---------------------------------------------------------------------------
        self, transition: HPCTransition,
    ) -> HPCTransitionResult:  # fmt: skip
        """Run phase 2 of the TEM-compatible HPC transition.

        Phase 2 recalls place codes from prior and posterior structural cues,
        produces generative and inference grounded-location beliefs, and writes
        the resulting episode into memory.
        """
        state = transition.state
        grid_prior_recall = self.recall(
            read_cues=transition.prior_read_cues,
            state=state,
            role="generative",
            read=transition.prior_read,
        )
        grid_posterior_recall = self.recall(
            read_cues=transition.posterior_read_cues,
            state=state,
            role="generative",
            read=transition.posterior_read,
        )

        place_retrieved, state = self.generative(grid_posterior_recall, state)
        place_prior, state = self.generative(grid_prior_recall, state)
        place_post, state = self.inference(transition.inference_sensory_query, transition.inference_structural_query, state)
        payload = WritePayload(generative=place_retrieved, inference=transition.sensory.recall, named_writes=transition.named_writes)
        state = self.update(place_post, payload, state)

        return HPCTransitionResult(
            sensory=transition.sensory,
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
        """Return a generative grounded-location sample or mean.

        The provided ``p_g`` is interpreted as the mean of a grounded-location
        belief whose uncertainty is inherited from the current state.
        """
        transition = LocationBelief(mean=p_g, uncertainty=state.uncertainty)
        p_gen = utils.sample_diag_gaussian(transition) if self.config.do_sample else transition.mean
        return p_gen, state.new(p_gen, state.uncertainty)

    def inference(  # -----------------------------------------------------------------------------
        self, x_: list[Tensor], g_: list[Tensor], state: HPCState,
    ) -> tuple[list[Tensor], HPCState]:  # fmt: skip
        """Infer grounded location from projected sensory and structural cues."""
        transition = self.place_inference(x_, g_)
        p_inf = utils.sample_diag_gaussian(transition) if self.config.do_sample else transition.mean
        return p_inf, state.new(p_inf, transition.uncertainty)


# =================================================================================================
__all__ = [
    "HPCCommonSettings", "SensoryRead", "SensoryReadResult", "HPCBase",
    "HPCState", "HPCTransition", "HPCTransitionResult", "WritePayload",
]  # fmt: skip
