"""MEC (medial entorhinal cortex) dynamics.

This package implements grid-cell path integration with optional corrections
from hippocampal memory (p→g) and shiny landmark cues (OVC).

The public entry point is `MECModel`, which exposes a TEM-compatible API via
`init_state`, `generative`, and `inference`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from scipy.stats import truncnorm
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.controllers.tem import NO_PREVIOUS_ACTION
from ehc_sn.modules.mec.ovc import OVCCorrection, OVCSettings
from ehc_sn.modules.mec.p2g import P2GMemory, P2GMemSettings
from ehc_sn.modules.mec.path import PathIntegrator, PathSettings
from ehc_sn.types import AbstractLocation, Device, Dtype, GroundedLocation, LocationBelief
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class MECSettings(BaseModel, extra="forbid"):
    """Settings for MEC modules."""

    grid_shape: list[int] = Field(
        ...,
        min_length=1,
        description=(
            "Sizes of grid-cell frequency modules. "
            "The number of frequencies is inferred from the length of this list."
        ),
    )

    do_sample: bool = Field(
        default=False,
        description="Whether to sample from distributions (stochastic) or use means (deterministic). Sampling policy is centralized in MECModel.",
    )
    sigma_init: float = Field(
        default=0.5,
        frozen=True,
        description="Standard deviation for initializing grid cell activations.",
    )

    clamp_min: float = Field(
        default=-1.0,
        description="Minimum activation clamp for OVC cells.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum activation clamp for OVC cells.",
    )

    path: PathSettings = Field(
        default_factory=PathSettings,
        description="Path integration module config.",
    )
    p2g: P2GMemSettings = Field(
        default_factory=P2GMemSettings,
        description="Place-to-grid memory inference module config.",
    )
    ovc: OVCSettings = Field(
        default_factory=OVCSettings,
        description="OVC module config.",
    )


# =================================================================================================
@dataclass()
class MECState(DetachMixin):
    """Container for MEC state.

    The state represents a belief over abstract location encoded as grid (and
    optionally OVC) activations with per-frequency uncertainty.

    Attributes:
        abstract_belief: Belief over the abstract/grid code.
        cells: List of per-frequency activations. If OVC modules are enabled,
            their activations are appended after the grid modules.
        uncertainty: Optional list of per-frequency uncertainties aligned with
            `cells`.
    """

    abstract_belief: LocationBelief  # State and uncertainty over abstract locations
    _n_ovc_modules: Optional[int] = None  # Cached number of appended OVC modules

    @property
    def cells(self) -> list[Tensor]:
        """Return grid + OVC activations."""
        return self.abstract_belief.mean

    @property
    def uncertainty(self) -> Optional[list[Tensor]]:
        """Return grid + OVC uncertainties."""
        return self.abstract_belief.uncertainty

    def new(  # -----------------------------------------------------------------------------------
        self, cells: list[Tensor], uncertainty: Optional[list[Tensor]],
    ) -> MECState:  # fmt: skip
        """Return a copy with an updated abstract-location belief."""
        return replace(self, abstract_belief=LocationBelief(mean=cells, uncertainty=uncertainty))

    def replace_rows(  # --------------------------------------------------------------------------
        self, flag: Tensor, fresh: "MECState",
    ) -> "MECState":  # fmt: skip
        """Return a state where flagged rows are replaced from ``fresh``."""
        uncertainty = None
        if self.uncertainty is not None and fresh.uncertainty is not None:
            uncertainty = utils.merge_multiscale_rows(flag, self.uncertainty, fresh.uncertainty)
        return self.new(
            cells=utils.merge_multiscale_rows(flag, self.cells, fresh.cells),
            uncertainty=uncertainty,
        )

    @property
    def grid_cells(self) -> list[Tensor]:
        """Return only the grid-cell activations."""
        if self._n_ovc_modules is None:
            return self.cells
        return self.cells[: len(self.cells) - self._n_ovc_modules]

    @property
    def ovc_cells(self) -> Optional[list[Tensor]]:
        """Return only the OVC activations, or `None` if not present."""
        if self._n_ovc_modules is None:
            return None
        return self.cells[len(self.cells) - self._n_ovc_modules :]


# =================================================================================================
class MECModel(nn.Module):
    """Compose MEC submodules into a TEM-compatible interface.

    The model exposes a stateful interface using explicit `MECState` objects.
    Internally it composes:

    - `PathIntegrator` for action-driven transitions
    - `P2GMemory` for memory-based correction (p→g)
    - `OVCCorrection` for shiny landmark cue fusion
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, action_count: int, n_hippocampal: list[int], f_initial: list[float], config: MECSettings,
        *, device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self._config = config
        self._action_count = action_count
        self._shape = self._resolve_shape(config)
        self._n_freq = len(self._shape)
        self._n_ovc_modules = len(config.ovc.shape or []) if config.ovc.mode == "separate" else None

        # Prior: learned "default phase" of the grid code at reset
        init_fn = lambda size: truncnorm.rvs(-2, 2, size=size, loc=0, scale=config.sigma_init)
        self.cells_init = nn.ParameterList([nn.Parameter(torch.tensor(init_fn(n), dtype=torch.float32)) for n in self._shape])  # fmt: skip
        self.uncertainty_init = nn.ParameterList([nn.Parameter(torch.tensor(init_fn(n), dtype=torch.float32)) for n in self._shape])  # fmt: skip

        # Instantiate submodules
        self.path_integration = PathIntegrator(action_count, self._shape, f_initial, config=config.path)
        self.p2g_correction = P2GMemory(n_hippocampal, self._shape, config=config.p2g)
        self.ovc_correction = OVCCorrection(self._shape, config=config.ovc)

    @property
    def config(self) -> MECSettings:
        """Return the MEC config."""
        return self._config

    @property
    def shape(self) -> list[int]:
        """Return the resolved full MEC shape."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Return the total number of MEC frequency modules."""
        return self._n_freq

    @property
    def n_ovc_modules(self) -> int:
        """Return the number of appended OVC modules."""
        return 0 if self._n_ovc_modules is None else self._n_ovc_modules

    @staticmethod
    def _resolve_shape(  # ------------------------------------------------------------------------
        config: MECSettings,
    ) -> list[int]:  # fmt: skip
        """Resolve the full MEC shape from the configured OVC mode."""
        if config.ovc.mode == "separate":
            return list(config.grid_shape) + list(config.ovc.shape or [])
        return list(config.grid_shape)

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int, device: Optional[Device] = None,
    ) -> MECState:  # fmt: skip
        """Create an initial MEC state from learned priors.

        Args:
            batch_size: Batch size for the returned state tensors.
            device: Optional device to place the returned tensors on.

        Returns:
            An initialized `MECState`.
        """
        g0 = [g.unsqueeze(0).expand(batch_size, -1).to(device) for g in self.cells_init]
        sigma_0 = [std.unsqueeze(0).expand(batch_size, -1).to(device) for std in self.uncertainty_init]
        transition = LocationBelief(mean=g0, uncertainty=sigma_0)
        return MECState(transition, _n_ovc_modules=self._n_ovc_modules)

    def set_runtime(  # ---------------------------------------------------------------------------
        self, *, p2g_uncertainty_offset: float,
    ) -> None:  # fmt: skip
        """Set runtime hyperparameters.

        Args:
            p2g_uncertainty_offset: Additive uncertainty offset for P2G inference.
        """
        self.p2g_correction.runtime.uncertainty_offset = p2g_uncertainty_offset

    def forward(  # -------------------------------------------------------------------------------
        self, *, _,
    ) -> tuple[list[Tensor], MECState]:  # fmt: skip
        """Not implemented.

        Raises:
            NotImplementedError: Always. Use `generative` or `inference`.
        """
        raise NotImplementedError("MEC forward not implemented. Use generative() or inference().")

    def generative(  # ----------------------------------------------------------------------------
        self, action: Tensor, landmark_id: Tensor | None, state: MECState,
    ) -> tuple[AbstractLocation, MECState]:  # fmt: skip
        """Run the generative (path integration) update.

        Args:
            action: Action ids of shape `(batch, 1)` from the environment.
            landmark_id: Optional current-cell landmark ids of shape `(batch, 1)`.
            state: Current MEC state.

        Returns:
            A tuple `(g_gen, new_state)` where `g_gen` is the generative grid
            code and `new_state` is the updated MEC state.
        """
        action_encoded, no_previous_action = self._encode_action_ids(action)
        no_direc_mask = None
        if landmark_id is not None:
            no_direc_mask = landmark_id.squeeze(-1).to(torch.int64) != 0
            if not torch.any(no_direc_mask):
                no_direc_mask = None

        # 1) Action-driven transition for the state (legacy g_path)
        transition = self.path_integration(action_encoded, state.cells, no_direc_mask=None)
        if self.config.do_sample:
            cells_next = utils.sample_diag_gaussian(transition)
        else:
            cells_next = transition.mean

        # 2) g_gen: reuse mu when possible, only compute no_direc when needed
        if no_direc_mask is not None:
            g_gen = self._clamp(self.path_integration.mean(action_encoded, state.cells, no_direc_mask))
        elif self.config.do_sample:
            g_gen = cells_next  # legacy: g_gen == sampled g when no shiny
        else:
            g_gen = self._clamp(transition.mean)

        if torch.any(no_previous_action):
            g_gen, next_state = self._preserve_reset_rows(
                no_previous_action, g_gen, state, state.new(cells_next, transition.uncertainty)
            )
            return g_gen, next_state

        return g_gen, state.new(cells_next, transition.uncertainty)

    def inference(  # -----------------------------------------------------------------------------
        self, p_x: Optional[GroundedLocation], landmark_id: Tensor | None, state: MECState,
    ) -> tuple[AbstractLocation, MECState]:  # fmt: skip
        """Run inference by fusing memory and OVC cues into the state.

        Args:
            p_x: Retrieved place-cell activations per frequency (from HPC).
            landmark_id: Optional current-cell landmark ids of shape `(batch, 1)`.
            state: Current MEC state (typically after path integration).

        Returns:
            A tuple `(g_inf, new_state)` where `g_inf` is the inferred grid code
            and `new_state` is the updated MEC state.
        """
        transition: LocationBelief = state.abstract_belief
        # Step 1: Correct path integration with memory-based inference
        transition = self.p2g_correction(p_x, transition) if p_x is not None else transition
        # Step 2: Apply OVC correction from shiny landmarks.
        transition = self.ovc_correction(landmark_id, transition) if self.config.ovc.mode != "off" else transition # fmt: skip

        # Apply central sampling policy (legacy parity: g_inf is sampled when do_sample=True)
        if self.config.do_sample:
            cells_next = utils.sample_diag_gaussian(transition)
        else:
            cells_next = transition.mean
        g_inf = self._clamp(cells_next)

        return g_inf, state.new(cells_next, transition.uncertainty)

    def _clamp(  # --------------------------------------------------------------------------------
        self, g: AbstractLocation,
    ) -> AbstractLocation:  # fmt: skip
        """Clamp activations for numerical stability.

        Args:
            g: Per-frequency activations.

        Returns:
            Clamped activations.
        """
        return [torch.clamp(g_f, min=self._config.clamp_min, max=self._config.clamp_max) for g_f in g]

    @staticmethod
    def _preserve_reset_rows(  # ------------------------------------------------------------------
        reset_mask: Tensor, g_gen: AbstractLocation, state_before: MECState, state_after: MECState,
    ) -> tuple[AbstractLocation, MECState]:  # fmt: skip
        """Restore pre-step MEC priors for rows with no previous action."""
        if not torch.any(reset_mask):
            return g_gen, state_after
        return (
            utils.merge_multiscale_rows(reset_mask, g_gen, state_before.cells),
            state_after.replace_rows(reset_mask, state_before),
        )

    def _encode_action_ids(  # --------------------------------------------------------------------
        self, action: Tensor,
    ) -> tuple[Tensor, Tensor]:  # fmt: skip
        """Encode environment action ids for path integration.

        Returns the one-hot action basis plus a boolean mask selecting rows that
        represent the controller-internal "no previous action" boundary.
        """
        action_ids = action.squeeze(-1).to(torch.int64)
        no_previous_action = action_ids == NO_PREVIOUS_ACTION
        invalid = ((action_ids < 0) & ~no_previous_action) | (action_ids >= self._action_count)
        if torch.any(invalid):
            bad_ids = action_ids[invalid].unique(sorted=True)
            raise ValueError(f"Action ids must be in [0, {self._action_count}), got {bad_ids.tolist()}.")
        encoded_ids = action_ids.masked_fill(no_previous_action, 0)
        encoded = torch.nn.functional.one_hot(encoded_ids, num_classes=self._action_count).to(torch.float32)
        return encoded, no_previous_action


# =================================================================================================
__all__ = ["MECModel", "MECState", "MECSettings"]
