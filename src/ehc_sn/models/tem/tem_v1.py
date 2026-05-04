"""TEM v1 backbone with canonical HPC-MEC-LEC architecture.

This file defines the TEMModelV1 class, which implements a standard
architecture for the Temporal Episodic Memory (TEM) model. The TEMModelV1
class integrates three core components: the Hippocampus (HPC), the Medial
Entorhinal Cortex (MEC), and the Lateral Entorhinal Cortex (LEC). Each
component is implemented as a separate module with its own settings and
state management.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, cast

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.models.tem.core.tem_base import GridCodes, PlaceCodes, PredCodes, TEMProjectionSettings
from ehc_sn.modules.hpc import HPCAttractor, HPCAttractorSettings, HPCState, WritePayload
from ehc_sn.modules.hpc.query_policy import CueRead, ReadCues
from ehc_sn.modules.lec import LECModel, LECSettings, LECState
from ehc_sn.modules.mec import MECModel, MECSettings, MECState
from ehc_sn.modules.projection import ProjectionBundle, ProjectionModule
from ehc_sn.types import MemoryState, MultiScaleCode
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class ModelSettingsV1(BaseModel, extra="forbid", strict=False):
    """Canonical TEM v1 model settings.

    This compact schema keeps the environment-facing observation/action
    contract, shared multiscale frequencies, and component-local settings in a
    single model config without reintroducing legacy aliases.
    """

    @classmethod
    def from_config(cls, path: Path) -> "ModelSettingsV1":
        """Load model settings from a TOML configuration file."""
        config_map = tomllib.load(Path(path).open("rb"))
        return cls.model_validate(config_map)

    transition_action_count: int = Field(
        ...,
        ge=1,
        description="Discrete actions to support for path integration in the MEC module.",
    )
    f_initial: list[float] = Field(
        default_factory=lambda: [0.99, 0.3, 0.09, 0.5, 0.4],
        min_length=1,
        description="Initial feature frequencies resolved across MEC and HPC modules.",
    )

    hpc: HPCAttractorSettings = Field(..., description="Settings for the HPC attractor module.")
    lec: LECSettings = Field(..., description="Settings for the LEC module.")
    mec: MECSettings = Field(..., description="Settings for the MEC module.")

    projections: TEMProjectionSettings = Field(
        ...,
        description="Settings for the inter-region projection modules connecting MEC and LEC to HPC.",
    )
    enable_sensory_recall: bool = Field(
        default=True,
        description="Whether the HPC should perform sensory-cued recall during the phase-1 TEM step.",
    )


# =============================================================================
@dataclass
class TEMInputV1(DetachMixin):
    """Task-agnostic input payload for TEM v1 forward steps."""

    sensory_codes: MultiScaleCode
    previous_action: Tensor
    episode_start: Optional[Tensor] = None
    landmark_id: Optional[Tensor] = None


# =============================================================================
@dataclass
class TEMOutputV1(DetachMixin):
    """Task-agnostic output payload for TEM v1 forward steps."""

    grid_codes: GridCodes
    place_codes: PlaceCodes
    pred_codes: PredCodes


# =============================================================================
@dataclass
class TEMStateV1(DetachMixin):
    """Container for the full recurrent TEM state."""

    lec: LECState  # State of the Lateral Entorhinal Cortex
    mec: MECState  # State of the Medial Entorhinal Cortex
    hpc: HPCState  # State of the Hippocampus


# =============================================================================
class TEMModelV1(nn.Module):
    """ """

    def __init__(  # ----------------------------------------------------------
        self,
        config: ModelSettingsV1,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        """Construct the TEM backbone from the resolved TEM v1 model settings."""
        super().__init__()
        self._config = config
        n_freq = len(config.hpc.shape)

        # Entorhinal Hippocampal Circuit components
        self.hpc = HPCAttractor(n_freq, config.f_initial, config.hpc, device=device, dtype=dtype)
        self.mec = MECModel(config.transition_action_count, config.hpc.shape, config.f_initial, config.mec, device=device, dtype=dtype)
        self.lec = LECModel(config.f_initial, config.lec, device=device, dtype=dtype)

        # Projection modules
        self.projections = ProjectionBundle.from_modules(
            mec_to_hpc=(self.mec, self.hpc, config.projections.mec_to_hpc),
            lec_to_hpc=(self.lec, self.hpc, config.projections.lec_to_hpc),
        )

        self.reset_parameters()

    @property
    def config(self) -> ModelSettingsV1:
        """Return the parsed TEM v1 model settings."""
        return self._config

    @property
    def mec_to_hpc(self) -> ProjectionModule:
        """Return the MEC-to-HPC projection edge."""
        return cast(ProjectionModule, self.projections["mec_to_hpc"])

    @property
    def lec_to_hpc(self) -> ProjectionModule:
        """Return the LEC-to-HPC projection edge."""
        return cast(ProjectionModule, self.projections["lec_to_hpc"])

    def reset_parameters(  # --------------------------------------------------
        self,
    ) -> None:
        """Reset all model parameters."""
        # self.hpc.reset_parameters(); already reset by the module constructor
        # self.mec.reset_parameters(); already reset by the module constructor
        # self.lec.reset_parameters(); already reset by the module constructor
        self.projections.reset_parameters()

    def init_state(  # --------------------------------------------------------
        self,
        batch_size: int,
        *,
        memory: Optional[MemoryState] = None,
        device: Optional[Device] = None,
    ) -> TEMStateV1:
        """Create an initial recurrent TEM state."""
        memory = memory if memory is not None else self.hpc.init_memory(batch_size, device=device)
        return TEMStateV1(
            lec=self.lec.init_state(batch_size, device=device),
            mec=self.mec.init_state(batch_size, device=device),
            hpc=self.hpc.init_state(batch_size, device=device, memory=memory),
        )

    def reset_state(  # -------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: TEMStateV1,
    ) -> TEMStateV1:
        """Reset flagged rows to a fresh episode state while preserving active rows."""
        reset_flag = reset_flag.to(torch.bool).view(-1)
        if not torch.any(reset_flag):
            return state

        return TEMStateV1(
            lec=self.lec.reset_state(state.lec, reset_flag),
            mec=self.mec.reset_state(state.mec, reset_flag),
            hpc=self.hpc.reset_state(state.hpc, reset_flag),
        )

    def set_runtime(  # -------------------------------------------------------
        self,
        eta: float,
        hebbian_decay: float,
        p2g_uncertainty_offset: float,
    ) -> None:
        """ """
        self.mec.set_runtime(p2g_uncertainty_offset=p2g_uncertainty_offset)
        self.hpc.set_runtime(eta=eta, hebbian_decay=hebbian_decay)

    def forward(  # -----------------------------------------------------------
        self,
        inputs: TEMInputV1,
        state: Optional[TEMStateV1] = None,
    ) -> tuple[TEMOutputV1, TEMStateV1]:
        """Run one TEM step from the current payload and recurrent state.

        ``state`` is assumed to have already been reset for any fresh episode
        rows by the caller. When ``state`` is ``None``, a fresh full-batch state
        is allocated and the normal single-step TEM transition is executed.
        """
        sensory_codes = inputs.sensory_codes
        previous_action = inputs.previous_action
        episode_start = inputs.episode_start
        landmark_id = inputs.landmark_id
        batch_size = int(sensory_codes[0].shape[0])
        device = sensory_codes[0].device

        # 0. Prepare the state, ensuring batch-alignment and extracting memories.
        if state is None:  # Init state if not provided
            state = self.init_state(batch_size, memory=None, device=device)
        else:  # Preserve the outer-state without truncating autograd
            state = replace(state)

        # 1. Compute the grid prior by path integration:
        g_prior, state.mec = self.mec.generative(previous_action, episode_start, landmark_id, state=state.mec)
        g_query_prior = self.mec_to_hpc(g_prior)

        # 2. Read sensory-cued place from the previous memory state.
        x_, state.lec = self.lec.inference(sensory_codes, state.lec)
        x_query = self.lec_to_hpc(x_)
        p_sensory_read = None
        if self.config.enable_sensory_recall:
            p_sensory_read = self.hpc.recall(
                read_cues=ReadCues(families={"x": x_query}),
                state=state.hpc,
                role="inference",
                read=CueRead(kind="cue", cue="x"),
            )

        # 3. Read ancestral place from the grid prior:
        p_grid_prior_read = self.hpc.recall(
            read_cues=ReadCues(families={"g": g_query_prior}),
            state=state.hpc,
            role="generative",
            read=CueRead(kind="cue", cue="g"),
        )

        # 4. Correct the grid prior using sensory recall:
        g_post, state.mec = self.mec.inference(p_sensory_read, landmark_id, state=state.mec)
        g_query_post = self.mec_to_hpc(g_post)

        # 5. Read retrieved place from the corrected grid:
        p_grid_post_read = self.hpc.recall(
            read_cues=ReadCues(families={"g": g_query_post}),
            state=state.hpc,
            role="generative",
            read=CueRead(kind="cue", cue="g"),
        )

        # 6. Form ancestral and retrieved place beliefs from the two structural recalls.
        p_prior, state.hpc = self.hpc.generative(p_grid_prior_read, state=state.hpc)
        p_retrieved, state.hpc = self.hpc.generative(p_grid_post_read, state=state.hpc)

        # 7. Infer the final grounded posterior from sensory and corrected grid cues:
        p_post, state.hpc = self.hpc.inference(x_query, g_query_post, state=state.hpc)

        # 8. Update memory only after all current-step reads are complete:
        payload = WritePayload(generative=p_retrieved, inference=p_sensory_read)
        state.hpc = self.hpc.update(p_post, payload, state=state.hpc)

        # 9. Decode the place codes back to sensory space for output:
        x_inference = self.projections.lec_to_hpc.inverse(p_post)
        x_ancestral = self.projections.lec_to_hpc.inverse(p_prior)
        x_retrieved = self.projections.lec_to_hpc.inverse(p_retrieved)

        # Package controller-compatible latent outputs and return the new state.
        grid_codes = GridCodes(prior=g_prior, posterior=g_post)
        place_codes = PlaceCodes(prior=p_prior, posterior=p_post, retrieved=p_retrieved, sensory=p_sensory_read)
        pred_codes = PredCodes(ancestral=x_ancestral, inference=x_inference, retrieved=x_retrieved)

        return TEMOutputV1(grid_codes=grid_codes, place_codes=place_codes, pred_codes=pred_codes), state


# =============================================================================
__all__ = ["ModelSettingsV1", "TEMInputV1", "TEMOutputV1", "PlaceCodes", "TEMStateV1", "TEMModelV1"]
