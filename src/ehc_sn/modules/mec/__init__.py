"""MEC (medial entorhinal cortex) dynamics.

This package implements grid-cell path integration with optional corrections
from hippocampal memory (p→g) and shiny landmark cues (OVC).

The public entry point is `MECModel`, which exposes a TEM-compatible API via
`init_state`, `generative`, and `inference`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from scipy.stats import truncnorm
from torch import Tensor, nn

from ehc_sn import utils
from ehc_sn.modules.mec.ovc import OVCCorrection, OVCSettings
from ehc_sn.modules.mec.p2g import P2GMemory, P2GMemSettings
from ehc_sn.modules.mec.path import PathIntegrator, PathSettings
from ehc_sn.types import AbstractLocation, Device, Dtype, GroundedLocation, LocationBelief, LocationLabel
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class MECSettings(BaseModel, extra="forbid"):
    """Settings for MEC modules."""

    shape: list[int] = Field(
        ...,
        description="",
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
    ovc: Optional[OVCSettings] = Field(
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
    _shape_ovc_modules: Optional[int] = None  # Cached number of OVC modules

    @property
    def cells(self) -> list[Tensor]:
        """Return grid + OVC activations."""
        return self.abstract_belief.mean

    @property
    def uncertainty(self) -> Optional[list[Tensor]]:
        """Return grid + OVC uncertainties."""
        return self.abstract_belief.uncertainty

    def new(self, cells: list[Tensor], uncertainty: Optional[list[Tensor]]) -> MECState:
        """Return a copy with an updated abstract-location belief."""
        return replace(self, abstract_belief=LocationBelief(mean=cells, uncertainty=uncertainty))

    @property
    def grid_cells(self) -> list[Tensor]:
        """Return only the grid-cell activations."""
        if self._shape_ovc_modules is None:
            return self.cells
        return self.cells[: len(self.cells) - self._shape_ovc_modules]

    @property
    def ovc_cells(self) -> Optional[list[Tensor]]:
        """Return only the OVC activations, or `None` if not present."""
        if self._shape_ovc_modules is None:
            return None
        return self.cells[len(self.cells) - self._shape_ovc_modules :]


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
        self, n_actions: int, n_hippocampal: list[int], f_init: list[float], config: MECSettings,
        device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self._config = config

        # Prior: learned "default phase" of the grid code at reset
        init_fn = lambda size: truncnorm.rvs(-2, 2, size=size, loc=0, scale=config.sigma_init)
        self.cells_init = nn.ParameterList([nn.Parameter(torch.tensor(init_fn(n), dtype=torch.float32)) for n in self.shape])  # fmt: skip
        self.uncertainty_init = nn.ParameterList([nn.Parameter(torch.tensor(init_fn(n), dtype=torch.float32)) for n in self.shape])  # fmt: skip

        # Instantiate submodules
        self.path_integration = PathIntegrator(n_actions, self.shape, f_init, config=config.path)
        self.p2g_correction = P2GMemory(n_hippocampal, self.shape, config=config.p2g)
        self.ovc_correction = OVCCorrection(config.ovc.shape, self.shape, config=config.ovc)

    @property
    def config(self) -> MECSettings:
        """Return the MEC config."""
        return self._config

    def init_state(self, batch_size: int, device: Optional[Device] = None) -> MECState:
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
        return MECState(transition, _shape_ovc_modules=self._shape_ovc_modules)

    def set_runtime(self, *, p2g_uncertainty_offset: float) -> None:
        """Set runtime hyperparameters.

        Args:
            p2g_uncertainty_offset: Additive uncertainty offset for P2G inference.
        """
        self.p2g_correction.runtime.uncertainty_offset = p2g_uncertainty_offset

    def forward(self, *, _) -> tuple[list[Tensor], MECState]:
        """Not implemented.

        Raises:
            NotImplementedError: Always. Use `generative` or `inference`.
        """
        raise NotImplementedError("MEC forward not implemented. Use generative() or inference().")

    def generative(
        self, action: Tensor, locations: list[LocationLabel], state: MECState
    ) -> tuple[AbstractLocation, MECState]:
        """Run the generative (path integration) update.

        Args:
            a: One-hot action tensor of shape `(batch, n_actions)`.
            locations: Per-environment metadata. A non-`None` `"shiny"` value
                indicates a landmark cue is present.
            state: Current MEC state.

        Returns:
            A tuple `(g_gen, new_state)` where `g_gen` is the generative grid
            code and `new_state` is the updated MEC state.
        """
        # Build no-direction mask for shiny environments
        shiny_envs = [loc.get("shiny") is not None for loc in locations]
        any_shiny = any(shiny_envs)
        no_direc_mask = (
            torch.tensor(shiny_envs, device=action.device, dtype=torch.bool) if any_shiny else None
        )

        # 1) Action-driven transition for the state (legacy g_path)
        transition = self.path_integration(action, state.cells, no_direc_mask=None)
        if self.config.do_sample:
            cells_next = utils.sample_diag_gaussian(transition)
        else:
            cells_next = transition.mean

        # 2) g_gen: reuse mu when possible, only compute no_direc when needed
        if any_shiny:
            g_gen = self._clamp(self.path_integration.mean(action, state.cells, no_direc_mask))
        elif self.config.do_sample:
            g_gen = cells_next  # legacy: g_gen == sampled g when no shiny
        else:
            g_gen = self._clamp(transition.mean)

        return g_gen, state.new(cells_next, transition.uncertainty)

    def inference(
        self, p_x: Optional[GroundedLocation], locations: list[LocationLabel], state: MECState
    ) -> tuple[AbstractLocation, MECState]:
        """Run inference by fusing memory and OVC cues into the state.

        Args:
            p_x: Retrieved place-cell activations per frequency (from HPC).
            locations: Per-environment metadata used for OVC correction.
            state: Current MEC state (typically after path integration).

        Returns:
            A tuple `(g_inf, new_state)` where `g_inf` is the inferred grid code
            and `new_state` is the updated MEC state.
        """
        transition: LocationBelief = state.abstract_belief
        # Step 1: Correct path integration with memory-based inference
        transition = self.p2g_correction(p_x, transition) if p_x is not None else transition
        # Step 2: Apply OVC correction from shiny landmarks.
        transition = (
            self.ovc_correction(locations, transition) if self._shape_ovc_modules is not 0 else transition
        )

        # Apply central sampling policy (legacy parity: g_inf is sampled when do_sample=True)
        if self.config.do_sample:
            cells_next = utils.sample_diag_gaussian(transition)
        else:
            cells_next = transition.mean
        g_inf = self._clamp(cells_next)

        return g_inf, state.new(cells_next, transition.uncertainty)

    def _clamp(self, g: AbstractLocation) -> AbstractLocation:
        """Clamp activations for numerical stability.

        Args:
            g: Per-frequency activations.

        Returns:
            Clamped activations.
        """
        return [torch.clamp(g_f, min=self._config.clamp_min, max=self._config.clamp_max) for g_f in g]


# =================================================================================================
__all__ = ["MECModel", "MECState"]
