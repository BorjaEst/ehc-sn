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
    as multi-scale codes: `list[Tensor]` (one tensor per frequency module).

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
from ehc_sn.types import Device, Dtype, LocationBelief, Matrix, MemoryState
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class HPCSettings(BaseModel, extra="forbid"):
    """Settings for HPC modules."""

    shape: list[int] = Field(
        ...,
        description="Feature dimensionality per frequency module.",
    )

    common_memory: bool = Field(  # Probably to move to hebbian which will rename memory
        default=False,
        description="Use common memory for generative and inference network",
    )
    do_sample: bool = Field(
        default=False,
        description="Whether to sample from location distributions or use means.",
    )

    clamp_min: float = Field(
        default=-1.0,
        description="Minimum activation clamp for OVC cells.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum activation clamp for OVC cells.",
    )

    attractor: AttractorSettings = Field(
        default_factory=AttractorSettings,
        description="Attractor dynamics module config.",
    )
    location: GroundLocSettings = Field(
        default_factory=GroundLocSettings,
        description="Location distribution module config.",
    )
    memory: HebbianUpdateSettings = Field(
        default_factory=HebbianUpdateSettings,
        description="Hebbian update module config.",
    )


# =================================================================================================
@dataclass
class HPCState(DetachMixin):
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
    _memory: list[Matrix]  # Memory matrices

    @property
    def cells(self) -> list[Tensor]:
        """Return grounded location features."""
        return self.location.mean

    @property
    def uncertainty(self) -> Optional[list[Tensor]]:
        """Return grounded location uncertainty."""
        return self.location.uncertainty

    @property
    def memory(self) -> Optional[list[Matrix]]:
        """Return Hebbian memory matrices."""
        return self._memory


# =================================================================================================
class HPCModel(nn.Module):
    """HPC facade that composes retrieval, inference, and memory write.

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

    def __init__(  # ------------------------------------------------------------------------------
        self, n_stages: int, f_init: list[float], config: HPCSettings, 
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self._config = config
        shape, n_stages, f_init = config.shape, config.n_stages, config.f_init
        n_freq = len(shape)

        # Stage masks are buffers so `.to(device)` moves them automatically.
        # Each is shaped (n_stages, S) where S = sum(shape).
        masks = utils.update_to_masks(shape, update=utils.make_update_hierarchical(n_stages, n_freq))
        self.register_buffer("masks_hierarchical", masks, persistent=False)
        masks = utils.update_to_masks(shape, update=utils.make_update_full(n_stages, n_freq))
        self.register_buffer("masks_full", masks, persistent=False)

        # Hebbian write mask gates synapses in the flattened (S, S) matrix.
        mask = utils.make_hebbian_write_mask(n_stages, shape, f_init)
        self.register_buffer("update_mask", mask, persistent=False)

        # Instantiate submodules
        self.attractor = AttractorNetwork(shape, config.attractor)
        self.location = GroundLocation(shape, config.location)
        self.memory_system = HebbianUpdate(config.memory)

        self.reset_parameters()  # Initialize parameters and buffers

    @property
    def config(self) -> HPCSettings:
        """HPC module config."""
        return self._config

    def reset_parameters(  # ----------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Initialize parameters and buffers."""
        pass

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int, *, 
        device: Optional[Device] = None, memory: Optional[MemoryState] = None,
    ) -> HPCState:  # fmt: skip
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
        self.location = LocationBelief(mean=p_init, uncertainty=None)
        memory = memory or self.init_memory(batch_size=batch_size, device=device)
        return HPCState(location=self.location, _memory=memory)

    def reset_state(  # ---------------------------------------------------------------------------
        self, state: HPCState,  # TODO: define based in other modules reset_state
    ) -> HPCState:  # fmt: skip
        """ """
        raise NotImplementedError(
            "HPC reset_state not implemented. Use init_state or implement reset logic here."
        )

    def init_memory(  # ---------------------------------------------------------------------------
        self, batch_size: int, *, 
        device: Optional[Device] = None,
    ) -> list[Tensor]:  # fmt: skip
        """Initialize Hebbian memory matrices.

        Args:
            batch_size: Batch size.
            device: Device for the returned tensors.

        Returns:
            A list `[M_hier, M_full]` where each matrix is shaped `(B, S, S)`.

            If `config.common_memory=True`, `M_full` is the same tensor object
            as `M_hier` (shared memory).
        """
        m0 = torch.zeros((batch_size, sum(self.shape), sum(self.shape)), dtype=torch.float, device=device)
        memory = [m0]
        memory.append(m0 if self.config.common_memory else m0.clone())
        return memory

    def set_runtime(  # ---------------------------------------------------------------------------
        self, *, eta: float, hebbian_decay: float,
    ) -> None:  # fmt: skip
        """Set runtime hyperparameters.

        These values are commonly controlled by the training loop and are not
        part of the static config tree.

        Args:
            eta: Hebbian learning rate.
            hebbian_decay: Hebbian decay factor.
        """
        self.memory_system.runtime.eta = float(eta)
        self.memory_system.runtime.hebbian_decay = float(hebbian_decay)

    def forward(  # -------------------------------------------------------------------------------
        self, *, state: HPCState,
    ) -> tuple[list[Tensor], HPCState]:  # fmt: skip
        """ """
        raise NotImplementedError("HPC forward not implemented. Use generative() or inference().")

    def generative(  # ----------------------------------------------------------------------------
        self, p_g: list[Tensor], state: HPCState,
    ) -> tuple[list[Tensor], HPCState]:  # fmt: skip
        """Return a grounded location sample/mean from a provided distribution.

        This is used by TEM when generating grounded location `p` from retrieved
        place-cell-like patterns.

        Args:
            p_g: Grounded location mean per frequency module.
            state: Current `HPCState`.

        Returns:
            `(p_gen, new_state)` where `p_gen` is either a sample from the
            diagonal Gaussian (if `config.do_sample=True`) or the provided
            mean, and `new_state` updates `transition` accordingly.
        """
        transition = LocationBelief(mean=p_g, uncertainty=state.uncertainty)
        p_gen = utils.sample_diag_gaussian(transition) if self.config.do_sample else transition.mean
        return p_gen, state.new(p_gen, state.uncertainty)

    def inference(  # -----------------------------------------------------------------------------
        self, x_: list[Tensor], g_: list[Tensor], state: HPCState,
    ) -> tuple[list[Tensor], HPCState]:  # fmt: skip
        """Infer grounded location from projected sensory and abstract features.

        Args:
            x_: Projected sensory features per frequency module.
            g_: Projected abstract location per frequency module.
            state: Current `HPCState`.

        Returns:
            `(p_inf, new_state)` where `p_inf` is either a sample from the
            inferred diagonal Gaussian (if `config.do_sample=True`) or the
            mean, and `new_state` updates both mean and uncertainty.
        """
        transition = self.location(x_, g_)
        p_inf = utils.sample_diag_gaussian(transition) if self.config.do_sample else transition.mean
        return p_inf, state.new(p_inf, transition.uncertainty)

    def recall(  # --------------------------------------------------------------------------------
        self, p_query: list[Tensor], state: HPCState, *, mode: Literal["full", "hierarchical"],
    ) -> list[Tensor]:  # fmt: skip
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

    def update(  # --------------------------------------------------------------------------------
        self, p_inf: list[Tensor], p_gen_gi: list[Tensor], p_xi: Optional[list[Tensor]], state: HPCState
    ) -> HPCState:  # fmt: skip
        """Apply a Hebbian write to the memory matrices.

        The update is applied to the hierarchical memory, and optionally to the
        full memory depending on `config.common_memory`.

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
        m_full = self.memory_system(m_full, p_inf, p_xi) if not self.config.common_memory and p_xi else m_hier
        return HPCState(state.location, _memory=[m_hier, m_full])


# =================================================================================================
__all__ = ["HPCModel", "HPCState", "HPCSettings"]
