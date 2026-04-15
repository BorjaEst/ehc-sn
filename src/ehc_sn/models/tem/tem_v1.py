"""TEM v1 backbone."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeAlias, cast

import torch
from pydantic import BaseModel, Field, computed_field, model_validator
from torch import Tensor, nn

from ehc_sn.models.tem.core.tem_base import TEMTransitionPlan
from ehc_sn.modules.autoencoder import Autoencoder, AutoencoderSettings
from ehc_sn.modules.hpc import HPCAttractor, HPCAttractorSettings, HPCState
from ehc_sn.modules.hpc import SensoryRead as HPCSensoryRead
from ehc_sn.modules.hpc.query_policy import CueRead, ReadCues
from ehc_sn.modules.lec import LECModel, LECSettings, LECState
from ehc_sn.modules.mec import MECModel, MECSettings, MECState
from ehc_sn.modules.projection import ProjectionBundle, ProjectionModule, ProjectionSettings
from ehc_sn.types import Device, Dtype, MemoryState
from ehc_sn.utils.detach import DetachMixin

# Community-standard map-style batch: plain dict returned by MazeDataset / DataLoader.
Batch: TypeAlias = dict[str, Tensor]
ObsLogits = tuple[Tensor, Tensor, Tensor]  # (inference, retrieved, ancestral)
GridCodes = tuple[Tensor, Tensor]  # (posterior, prior)
PlaceCodes = tuple[Tensor, Tensor, Optional[Tensor]]  # (posterior, prior, sensory-cued retrieval)


# =================================================================================================
class TEMProjectionSettings(BaseModel, extra="forbid", strict=False):
    """Inter-region projection settings for TEM backbones."""

    lec_to_hpc_x: ProjectionSettings = Field(
        default_factory=lambda: ProjectionSettings(mode="tiling", learnable=False),
        description="Projection settings mapping LEC features into hippocampal query space.",
    )
    mec_to_hpc_g: ProjectionSettings = Field(
        default_factory=lambda: ProjectionSettings(mode="low_rank", learnable=False, rank=[10, 10, 8, 6, 6]),
        description="Projection settings mapping MEC codes into hippocampal query space.",
    )


# =================================================================================================
class ModelSettings_V1(BaseModel, extra="forbid", strict=False):
    """Canonical TEM v1 model settings.

    This compact schema keeps the environment-facing observation/action
    contract, shared multiscale frequencies, and component-local settings in a
    single model config without reintroducing legacy aliases.
    """

    observation_dim: int = Field(
        ...,
        ge=1,
        description="Dimensionality of raw observations from the environment.",
    )
    action_count: int = Field(
        ...,
        ge=1,
        description="Number of discrete actions in the environment.",
    )

    f_initial: list[float] = Field(
        default_factory=lambda: [0.99, 0.3, 0.09, 0.5, 0.4],
        min_length=1,
        description=(
            "List of initial feature frequencies for the model's resolved modules. "
            "Its length must match the full MEC/HPC frequency count after OVC mode resolution."
        ),
    )
    enable_sensory_recall: bool = Field(
        default=True,
        description="Whether the HPC should perform sensory-cued recall during the phase-1 TEM step.",
    )

    @model_validator(mode="after")
    def validate_model(self) -> "ModelSettings_V1":
        self._validate_f_initial()
        self._validate_stage_alignment()
        self._validate_frequency_alignment()
        return self

    def _validate_f_initial(self) -> None:
        if any(not (0.0 < value < 1.0) for value in self.f_initial):
            raise ValueError("f_initial values must be strictly between 0 and 1.")

    def _validate_stage_alignment(self) -> None:
        expected_freq = len(self.mec_shape)
        if expected_freq != self.n_stages:
            raise ValueError("The resolved MEC frequency count must equal len(f_initial).")

    def _validate_frequency_alignment(self) -> None:
        if len(self.hpc.shape) != self.n_total_freq:
            raise ValueError("len(hpc.shape) must equal the derived total MEC frequency count.")

    @computed_field
    @property
    def n_stages(self) -> int:
        """Return the resolved number of TEM frequency modules."""
        return len(self.f_initial)

    @computed_field
    @property
    def n_freq(self) -> int:
        """Return the total number of frequency modules."""
        return len(self.f_initial)

    hpc: HPCAttractorSettings = Field(
        ...,
        description="Settings for the attractor-based hippocampal module.",
    )
    lec: LECSettings = Field(
        ...,
        description="Settings for the LEC module, including feature filtering parameters.",
    )
    mec: MECSettings = Field(
        ...,
        description="Settings for the MEC module, including path integration and correction parameters.",
    )

    autoencoder: AutoencoderSettings = Field(
        ...,
        description="Settings for the autoencoder module used for observation compression.",
    )
    projections: TEMProjectionSettings = Field(
        default_factory=TEMProjectionSettings,
        description="Inter-region projection settings grouped by named edge.",
    )

    @computed_field
    @property
    def lec_shape(self) -> list[int]:
        """Return the resolved LEC feature shape across all frequencies."""
        return [self.lec.feature_dim] * self.n_total_freq

    @computed_field
    @property
    def mec_shape(self) -> list[int]:
        """Return the full MEC shape including optional OVC modules."""
        return self.mec.mec_shape

    @computed_field
    @property
    def mec_ovc_shape(self) -> list[int]:
        """Return the appended OVC shape implied by the configured OVC mode."""
        return self.mec.mec_ovc_shape

    @computed_field
    @property
    def n_total_freq(self) -> int:
        """Return the total number of MEC/HPC frequencies after OVC expansion."""
        return self.mec.n_total_freq

    @classmethod
    def from_config(cls, path: Path) -> "ModelSettings_V1":
        """Load model settings from a TOML configuration file."""
        config_map = tomllib.load(Path(path).open("rb"))
        return cls.model_validate(config_map)


# =================================================================================================
@dataclass
class TEMState(DetachMixin):
    """Container for the full recurrent TEM state."""

    lec: LECState
    mec: MECState
    hpc: HPCState


class TEMModelV1(nn.Module):
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelSettings_V1, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Construct the TEM backbone from the resolved TEM v1 model settings."""
        super().__init__()
        self._config = config
        n_freq, n_actions = config.n_total_freq, config.action_count
        f_initial = config.f_initial

        # Autoencoder module for observation compression/decoding
        self.autoencoder = Autoencoder(config.observation_dim, config.lec.feature_dim, config.autoencoder)

        # Entorhinal Hippocampal Circuit components
        self.hpc = HPCAttractor(n_freq, f_initial, config.hpc, device=device, dtype=dtype)
        self.mec = MECModel(n_actions, config.hpc.shape, f_initial, config.mec, device=device, dtype=dtype)
        self.lec = LECModel(f_initial, config.lec, device=device, dtype=dtype)

        # Projection modules
        self.projections = ProjectionBundle.from_modules(
            mec_to_hpc_g=(self.mec, self.hpc, config.projections.mec_to_hpc_g),
            lec_to_hpc_x=(self.lec, self.hpc, config.projections.lec_to_hpc_x),
        )

    @property
    def config(self) -> ModelSettings_V1:
        """Return the parsed TEM v1 model settings."""
        return self._config

    @property
    def mec_to_hpc_g(self) -> ProjectionModule:
        """Return the MEC-to-HPC projection edge."""
        return cast(ProjectionModule, self.projections["mec_to_hpc_g"])

    @property
    def lec_to_hpc_x(self) -> ProjectionModule:
        """Return the LEC-to-HPC projection edge."""
        return cast(ProjectionModule, self.projections["lec_to_hpc_x"])

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int, *, memory: Optional[MemoryState] = None,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> TEMState:  # fmt: skip
        """Create an initial recurrent TEM state."""
        memory = memory if memory is not None else self.hpc.init_memory(batch_size=batch_size, device=device)
        lec_state = self.lec.init_state(batch_size, device=device)
        state_mec = self.mec.init_state(batch_size, device=device)
        hpc_state = self.hpc.init_state(batch_size, device=device, memory=memory)
        return TEMState(lec_state, state_mec, hpc_state)

    def reset_state(  # ---------------------------------------------------------------------------
        self, reset_flag: Tensor, state: TEMState,
    ) -> TEMState:  # fmt: skip
        """Reset flagged rows to a fresh episode state while preserving active rows."""
        reset_flag = reset_flag.to(torch.bool).view(-1)
        if not torch.any(reset_flag):
            return state

        return TEMState(
            lec=self.lec.reset_state(state.lec, reset_flag),
            mec=self.mec.reset_state(state.mec, reset_flag),
            hpc=self.hpc.reset_state(state.hpc, reset_flag),
        )

    def set_runtime(  # ---------------------------------------------------------------------------
        self, eta: float, hebbian_decay: float, p2g_uncertainty_offset: float,
    ) -> None:  # fmt: skip
        """ """
        self.mec.set_runtime(p2g_uncertainty_offset=p2g_uncertainty_offset)
        self.hpc.set_runtime(eta=eta, hebbian_decay=hebbian_decay)

    def forward(  # -------------------------------------------------------------------------------
        self, inputs: Batch, state: Optional[TEMState] = None,
    ) -> tuple[TEMState, ObsLogits, None, GridCodes, PlaceCodes]:  # fmt: skip
        """Run one TEM step from the current payload and recurrent state.

        ``state`` is assumed to have already been reset for any fresh episode
        rows by the caller. When ``state`` is ``None``, a fresh full-batch state
        is allocated and the normal single-step TEM transition is executed.
        """
        observation = inputs["observation"]
        previous_action = inputs["previous_action"]
        episode_start = inputs.get("episode_start")
        landmark_id = inputs.get("landmark_id")
        observation_embedding = self.autoencoder.encode(observation)

        if state is None:
            state = self.init_state(int(observation.shape[0]), memory=None, device=observation.device)

        # Grid transition prior from action-driven path integration.
        grid_prior, state.mec = self.mec.generative(previous_action, episode_start, landmark_id, state.mec)
        place_query_from_grid_prior = self.mec_to_hpc_g(grid_prior)

        # Sensory inference: encode observations into LEC features and query place memory from them.
        lec_features_post, state.lec = self.lec.inference(observation_embedding, state.lec)
        place_query_from_obs = self.lec_to_hpc_x(lec_features_post)
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
        place_query_from_grid_post = self.mec_to_hpc_g(grid_post)
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
        lec_features_from_place_post = self.lec_to_hpc_x.inverse(step.place_post)
        obs_features_inference = self.lec.generative(lec_features_from_place_post)
        logits_inference = self.autoencoder.decode(obs_features_inference)

        lec_features_from_place_retrieved = self.lec_to_hpc_x.inverse(step.place_retrieved)
        obs_features_retrieved = self.lec.generative(lec_features_from_place_retrieved)
        logits_retrieved = self.autoencoder.decode(obs_features_retrieved)

        lec_features_from_place_prior = self.lec_to_hpc_x.inverse(step.place_prior)
        obs_features_ancestral = self.lec.generative(lec_features_from_place_prior)
        logits_ancestral = self.autoencoder.decode(obs_features_ancestral)

        # Return controller-compatible rollout outputs for the TEM loss head.
        obs_logits = (logits_inference, logits_retrieved, logits_ancestral)
        grid = (transition.grid_post, transition.grid_prior)
        place = (step.place_post, step.place_prior, step.sensory.recall)
        return state, obs_logits, None, grid, place  # Action=None as TEM provides no direct action outputs


# =================================================================================================
__all__ = [
    "ModelSettings_V1", "TEMState", "TEMModelV1",
    "Batch", "ObsLogits", "GridCodes", "PlaceCodes",
]  # fmt: skip
