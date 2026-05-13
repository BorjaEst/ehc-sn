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

from ehc_sn import utils
from ehc_sn.models.tem.core.tem_base import GridCodes, PlaceCodes, PredCodes, TEMProjectionSettings, TEMTransitionPlan
from ehc_sn.modules.hpc import HPCAttractor, HPCAttractorSettings, HPCState
from ehc_sn.modules.hpc import SensoryRead as HPCSensoryRead
from ehc_sn.modules.hpc import WritePayload
from ehc_sn.modules.hpc.query_policy import CueRead, ReadCues
from ehc_sn.modules.lec import LECModel, LECSettings, LECState
from ehc_sn.modules.mec import MECModel, MECSettings, MECState
from ehc_sn.modules.projection import ProjectionBundle, ProjectionModule
from ehc_sn.types import Batch, MemoryState, MultiScaleCode
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
        description=(
            "List of initial feature frequencies for the model's resolved modules. "
            "Its length must match the full MEC/HPC frequency count after OVC mode resolution."
        ),
    )

    hpc: HPCAttractorSettings = Field(..., description="Settings for the attractor-based hippocampal module.")
    lec: LECSettings = Field(..., description="Settings for the LEC module, including feature filtering parameters.")
    mec: MECSettings = Field(..., description="Settings for the MEC module, including path integration and correction parameters.")

    projections: TEMProjectionSettings = Field(
        ...,
        description="Settings for the inter-region projection modules connecting MEC and LEC to HPC.",
    )
    enable_sensory_recall: bool = Field(
        default=True,
        description="Whether the HPC should perform sensory-cued recall during the phase-1 TEM step.",
    )


# =============================================================================
@dataclass(frozen=True)
class TEMInputV1(DetachMixin):
    """Task-agnostic input payload for TEM v1 forward steps."""

    sensory_codes: MultiScaleCode
    previous_action: Tensor
    episode_start: Optional[Tensor] = None
    landmark_id: Optional[Tensor] = None


# =============================================================================
@dataclass(frozen=True)
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
        n_actions = config.transition_action_count
        f_initial = config.f_initial

        # Entorhinal Hippocampal Circuit components
        self.hpc = HPCAttractor(n_freq, f_initial, config.hpc, device=device, dtype=dtype)
        self.mec = MECModel(n_actions, config.hpc.shape, f_initial, config.mec, device=device, dtype=dtype)
        self.lec = LECModel(f_initial, config.lec, device=device, dtype=dtype)

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

    def _sensory_correction_error(
        self,
        sensory_features: MultiScaleCode,
        p_sensory_read: Optional[list[Tensor]],
    ) -> Optional[list[Tensor]]:
        """Assemble detached p→g confidence evidence in HPC-projected sensory space.

        Error is computed in HPC space (after forward projection) rather than by
        inverting back to LEC space.  The transpose-based inverse scales by the
        tiling factor k, so a perfect tiled match would yield (k-1)²‖x‖² instead
        of zero.  Projecting forward keeps both sides in the same space and gives
        zero error for an exact recall regardless of tiling factor.
        """
        if p_sensory_read is None:
            return None

        x_query = self.lec_to_hpc(sensory_features)
        return [band_error.detach() for band_error in utils.squared_error(x_query, p_sensory_read)]

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
        observation_embedding = inputs.sensory_codes
        previous_action = inputs.previous_action
        episode_start = inputs.episode_start
        landmark_id = inputs.landmark_id

        if state is None:
            state = self.init_state(int(observation_embedding[0].shape[0]), memory=None, device=observation_embedding.device)

        # Grid transition prior from action-driven path integration.
        grid_prior, state.mec = self.mec.generative(previous_action, episode_start, landmark_id, state.mec)
        place_query_from_grid_prior = self.mec_to_hpc(grid_prior)

        # Sensory inference: encode observations into LEC features and query place memory from them.
        lec_features_post, state.lec = self.lec.inference(observation_embedding, state.lec)
        place_query_from_obs = self.lec_to_hpc(lec_features_post)
        sensory = self.hpc.read_sensory(
            HPCSensoryRead(
                state=state.hpc,
                read_cues=ReadCues(families={"x": place_query_from_obs, "g": place_query_from_grid_prior}),
                read=CueRead(kind="cue", cue="x"),
                enable_sensory_recall=self.config.enable_sensory_recall,
            )
        )

        # Grid posterior after correcting the prior with recalled place evidence.
        grid_post, state.mec = self.mec.inference(sensory.recall, landmark_id=landmark_id, state=state.mec)  # fmt: skip
        place_query_from_grid_post = self.mec_to_hpc(grid_post)
        transition = TEMTransitionPlan(
            sensory=sensory,
            grid_prior=grid_prior,
            grid_query_prior=place_query_from_grid_prior,
            grid_post=grid_post,
            grid_query_posterior=place_query_from_grid_post,
            generative_read=CueRead(kind="cue", cue="g"),
        )

        step = self.hpc.transition(transition.to_hpc_transition(state.hpc))
        state.hpc = step.state

        # Decode observation logits for the three TEM pathways.
        lec_features_from_place_post = self.lec_to_hpc.inverse(step.place_post)
        lec_features_from_place_retrieved = self.lec_to_hpc.inverse(step.place_retrieved)
        lec_features_from_place_prior = self.lec_to_hpc.inverse(step.place_prior)

        # Build and return model-native typed output (output-first, state second).
        pred_codes = PredCodes(
            inference=lec_features_from_place_post,
            retrieved=lec_features_from_place_retrieved,
            ancestral=lec_features_from_place_prior,
        )
        grid_codes_out = GridCodes(
            posterior=grid_post,
            prior=grid_prior,
        )
        place_codes_out = PlaceCodes(
            posterior=step.place_post,
            prior=step.place_prior,
            retrieved=step.place_retrieved,
            sensory=sensory.recall,
        )
        model_output = TEMOutputV1(pred_codes=pred_codes, grid_codes=grid_codes_out, place_codes=place_codes_out)
        return model_output, state


# =============================================================================
__all__ = ["ModelSettingsV1", "TEMInputV1", "TEMOutputV1", "PlaceCodes", "TEMStateV1", "TEMModelV1"]
