from __future__ import annotations

"""HPC (hippocampus) memory and retrieval.

This package implements the hippocampal component of TEM responsible for:

- Retrieving grounded location codes (place-cell-like) via attractor dynamics
    over a Hebbian memory matrix.
- Inferring grounded location distributions from projected sensory features and
    projected abstract location.
- Writing to Hebbian memory via a masked outer-product update.

Naming conventions:

- Variables suffixed with an underscore (e.g. `x_`, `g_`) refer to values that
    have been projected into the *memory* feature space.
- Grounded location codes `p` and abstract location codes `g` are represented
    as multi-scale codes: `List[Tensor]` (one tensor per frequency module).

Shape conventions:

- `B`: batch size
- `shape[f]`: feature dimensionality of module `f`
- `S = sum(shape)`: flattened feature dimensionality
- A multi-scale code is a list of tensors shaped `(B, shape[f])`.
- A memory matrix is shaped `(B, S, S)`.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.hpc.attractor import AttractorNetwork, AttractorSettings
from ehc_sn.modules.hpc.location import GroundLocation, GroundLocSettings
from ehc_sn.modules.hpc.memory import HebbianUpdate, HebbianUpdateSettings
from ehc_sn.types import GroundedLocation, LocationBelief, Matrix, MemoryState, MultiScaleCode

__all__ = ["HPCModel", "HPCState"]


# =================================================================================================
class HPCSettings(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """Settings for HPC modules."""

    common_memory: bool = Field(  # Probably to move to hebbian which will rename memory
        default=False,
        description="Use common memory for generative and inference network",
    )
    do_sample: bool = Field(
        default=False,
        description="Whether to sample from location distributions or use means.",
    )
    attractor: AttractorSettings = Field(
        default_factory=AttractorSettings,
        description="Attractor dynamics module settings.",
    )
    location: GroundLocSettings = Field(
        default_factory=GroundLocSettings,
        description="Location distribution module settings.",
    )
    memory: HebbianUpdateSettings = Field(
        default_factory=HebbianUpdateSettings,
        description="Hebbian update module settings.",
    )


# =================================================================================================
@dataclass
class HPCState:
    """Container for HPC state.

    Attributes:
        transition: A `LocationBelief` over grounded location codes. `transition.mean`
            is a multi-scale code (one tensor per frequency module). The
            uncertainty may be `None` when not modeled/used.
        memory: Hebbian memory matrices.

            The current implementation maintains two matrices:

            - `memory[0]`: hierarchical retrieval memory
            - `memory[1]`: full retrieval memory

            When `HPCSettings.common_memory=True`, both entries may refer to
            the same underlying tensor.
    """

    location: LocationBelief  # State and uncertainty over grounded locations
    _memory: List[Matrix]  # Memory matrices

    def new(
        self, cells: Optional[GroundedLocation] = None, uncertainty: Optional[MultiScaleCode] = None
    ) -> "HPCState":
        """Return a new state with an updated transition.

        This is a convenience helper used throughout TEM to keep state updates
        explicit (no in-place mutation).

        Args:
            cells: New grounded location mean (multi-scale code).
            uncertainty: New grounded location uncertainty (multi-scale code).
                Note: the parameter name preserves a legacy spelling.

        Returns:
            A new `HPCState` with updated `transition` and preserved `memory`.

        Notes:
            This method performs a shallow copy of the state fields. Use
            `detach()` when you need to carry state across iterations without
            keeping autograd history.
        """
        batch_size, device = self.cells[0].shape[0], self.cells[0].device
        copy, shape = self.__dict__.copy(), [v.shape[1] for v in self.cells]
        location = LocationBelief(
            mean=cells or [torch.zeros((batch_size, n), device=device) for n in shape],
            uncertainty=uncertainty,
        )
        copy.update(location=location)  # Memory is preserved
        return HPCState(**copy)

    def detach(self) -> "HPCState":
        """Return a detached copy.

        Use this when carrying state across iterations without backpropagating
        through history.

        Returns:
            A detached `HPCState` where all tensors in `transition` and `memory`
            have been detached.
        """
        mean = [v.detach() for v in self.cells]
        uncertainty = [v.detach() for v in self.uncertainty] if self.uncertainty else None
        memory = [m.detach() for m in self.memory] if self.memory is not None else None
        return HPCState(LocationBelief(mean, uncertainty), memory)

    @property
    def cells(self) -> List[Tensor]:
        """Return grounded location features."""
        return self.location.mean

    @property
    def uncertainty(self) -> Optional[List[Tensor]]:
        """Return grounded location uncertainty."""
        return self.location.uncertainty

    @property
    def memory(self) -> Optional[List[Matrix]]:
        """Return Hebbian memory matrices."""
        return self._memory


# =================================================================================================
class HPCModel(nn.Module):
    """HPC façade that composes retrieval, inference, and memory write.

    The HPC module provides TEM-compatible methods:

    - `init_state` / `init_memory` to create initial state
    - `recall` to retrieve grounded locations via an attractor network
    - `inference` to infer grounded locations from sensory + abstract features
    - `generative` to sample/use grounded locations from a provided distribution
    - `update` to apply a Hebbian write to one or two memory matrices

    Internally it delegates to three single-responsibility submodules:

    - `AttractorNetwork` (pattern completion): p_query + M -> p_recalled
    - `GroundLocation` (distribution): x_, g_ -> LocationBelief(p_mean, p_sigma)
    - `HebbianUpdate` (write): M, p_inf, p_gen -> M'
    """

    def __init__(
        self,
        n_stages: int,  # Number of attractor update stages
        shape: List[int],  # Grounded-location feature sizes per frequency module
        f_init: List[float],  # Frequency values per module
        *,
        settings: Optional[HPCSettings] = None,  # HPC settings
    ):
        super().__init__()
        self._shape, self._n_freq = list(shape), len(shape)
        self._n_stages = n_stages
        self._settings = settings

        # Stage masks are buffers so `.to(device)` moves them automatically.
        # Each is shaped (n_stages, S) where S = sum(shape).
        masks = utils.update_to_masks(shape, update=utils.make_update_hierarchical(n_stages, self.n_freq))
        self.register_buffer("masks_hierarchical", masks, persistent=False)
        masks = utils.update_to_masks(shape, update=utils.make_update_full(n_stages, self.n_freq))
        self.register_buffer("masks_full", masks, persistent=False)

        # Hebbian write mask gates synapses in the flattened (S, S) matrix.
        mask = utils.make_hebbian_write_mask(n_stages, shape, f_init)
        self.register_buffer("update_mask", mask, persistent=False)

        # Instantiate submodules
        self.attractor = AttractorNetwork(shape, settings.attractor)
        self.location = GroundLocation(shape, settings.location)
        self.memory_system = HebbianUpdate(settings.memory)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        memory: Optional[MemoryState] = None,
    ) -> HPCState:
        """Create an initial `HPCState`.

        Args:
            batch_size: Batch size for all state tensors.
            device: Optional device.

        Returns:
            An initialized `HPCState`.

            - `transition.mean`: list of zeros shaped `(B, shape[f])`
            - `transition.uncertainty`: `None`
            - `memory`: output of `init_memory` (two matrices, shape `(B, S, S)`)
        """
        p_init = [torch.zeros((batch_size, n), device=device) for n in self.shape]
        transition = LocationBelief(mean=p_init, uncertainty=None)
        memory = memory or self.init_memory(batch_size=batch_size, device=device)
        return HPCState(transition, memory)

    def init_memory(self, *, batch_size: int, device: torch.device) -> List[Tensor]:
        """Initialize Hebbian memory matrices.

        Args:
            batch_size: Batch size.
            device: Device for the returned tensors.

        Returns:
            A list `[M_hier, M_full]` where each matrix is shaped `(B, S, S)`.

            If `settings.common_memory=True`, `M_full` is the same tensor object
            as `M_hier` (shared memory).
        """
        m0 = torch.zeros((batch_size, sum(self.shape), sum(self.shape)), dtype=torch.float, device=device)
        memory = [m0]
        memory.append(m0 if self.settings.common_memory else m0.clone())
        return memory

    def set_runtime(self, *, eta: float, hebbian_decay: float) -> None:
        """Set runtime hyperparameters.

        These values are commonly controlled by the training loop and are not
        part of the static settings tree.

        Args:
            eta: Hebbian learning rate.
            hebbian_decay: Hebbian decay factor.
        """
        self.memory_system.runtime.eta = float(eta)
        self.memory_system.runtime.hebbian_decay = float(hebbian_decay)

    @property
    def settings(self) -> HPCSettings:
        """HPC module settings."""
        return self._settings

    @property
    def shape(self) -> List[int]:
        """Dimensionality of features per frequency module."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Number of frequency modules."""
        return self._n_freq

    @property
    def n_stages(self) -> int:
        """Number of attractor stages."""
        return self._n_stages

    def forward(self, *, state: HPCState) -> Tuple[List[Tensor], HPCState]:
        raise NotImplementedError("HPC forward not implemented. Use generative() or inference().")

    def generative(self, p_g: List[Tensor], state: HPCState) -> Tuple[List[Tensor], HPCState]:
        """Return a grounded location sample/mean from a provided distribution.

        This is used by TEM when generating grounded location `p` from retrieved
        place-cell-like patterns.

        Args:
            p_g: Grounded location mean per frequency module.
            state: Current `HPCState`.

        Returns:
            `(p_gen, new_state)` where `p_gen` is either a sample from the
            diagonal Gaussian (if `settings.do_sample=True`) or the provided
            mean, and `new_state` updates `transition` accordingly.
        """
        transition = LocationBelief(mean=p_g, uncertainty=state.uncertainty)
        p_gen = utils.sample_diag_gaussian(transition) if self.settings.do_sample else transition.mean
        return p_gen, state.new(p_gen, state.uncertainty)

    def inference(self, x_: List[Tensor], g_: List[Tensor], state: HPCState) -> Tuple[List[Tensor], HPCState]:
        """Infer grounded location from projected sensory and abstract features.

        Args:
            x_: Projected sensory features per frequency module.
            g_: Projected abstract location per frequency module.
            state: Current `HPCState`.

        Returns:
            `(p_inf, new_state)` where `p_inf` is either a sample from the
            inferred diagonal Gaussian (if `settings.do_sample=True`) or the
            mean, and `new_state` updates both mean and uncertainty.
        """
        transition = self.location(x_, g_)
        p_inf = utils.sample_diag_gaussian(transition) if self.settings.do_sample else transition.mean
        return p_inf, state.new(p_inf, transition.uncertainty)

    def recall(
        self, p_query: List[Tensor], state: HPCState, *, mode: Literal["full", "hierarchical"]
    ) -> List[Tensor]:
        """Retrieve grounded location via attractor dynamics.

        Args:
            p_query: Query code (multi-scale) in memory feature space.
            state: Current `HPCState` containing Hebbian memory matrices.
            mode: Retrieval mode.

                - `"hierarchical"`: use `memory[0]` and staged hierarchical masks
                - `"full"`: use `memory[1]` and full-update masks

        Returns:
            Retrieved grounded location code (multi-scale).

        Raises:
            ValueError: If `mode` is not one of `"full"` or `"hierarchical"`.
        """
        if mode == "hierarchical":
            return self.attractor(p_query, state.memory[0], masks=self.masks_hierarchical)
        elif mode == "full":
            return self.attractor(p_query, state.memory[1], masks=self.masks_full)
        raise ValueError(f"Invalid mode '{mode}'. Expected 'full' or 'hierarchical'.")

    def update(
        self, p_inf: List[Tensor], p_gen_gi: List[Tensor], p_xi: Optional[List[Tensor]], state: HPCState
    ) -> HPCState:
        """Apply a Hebbian write to the memory matrices.

        The update is applied to the hierarchical memory, and optionally to the
        full memory depending on `settings.common_memory`.

        Args:
            p_inf: Inferred grounded location per frequency module.
            p_gen_gi: Grounded location generated/retrieved from inferred
                abstract location.
            p_xi: Grounded location retrieved from sensory features (x-cued
                recall). This is expected to be present when full memory writes
                are enabled. If `None`, full memory is not updated.
            state: Current `HPCState`.

        Returns:
            A new `HPCState` with updated `memory` and unchanged `transition`.
        """
        m_hier, m_full = state.memory
        m_hier = self.memory_system(m_hier, p_inf, p_gen_gi, mask=self.update_mask)
        m_full = (
            self.memory_system(m_full, p_inf, p_xi) if not self.settings.common_memory and p_xi else m_hier
        )
        return HPCState(state.location, _memory=[m_hier, m_full])
