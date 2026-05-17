"""TEM v2 backbone model and associated dataclasses for settings, state, and
forward I/O.

This module defines the TEMModelV2 backbone, which implements the canonical TEM
architecture as a single PyTorch module with a TEM v2-compatible forward
contract. The forward contract is designed to be task-agnostic, with all
environment interaction abstracted into the input and all controller-compatible
latent outputs abstracted into the output.
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

from ehc_sn.models.tem.core.tem_base import GridCodes, PlaceCodes, PredCodes
from ehc_sn.modules.hpc import HPCAttention, HPCAttentionSettings, HPCState
from ehc_sn.modules.hpc import SensoryRead as HPCSensoryRead
from ehc_sn.modules.hpc.query_policy import ReadCues, TargetRead
from ehc_sn.modules.lec import LECModel, LECSettings, LECState
from ehc_sn.modules.mec import MECModel, MECSettings, MECState
from ehc_sn.modules.projection import ProjectionBundle, ProjectionModule
from ehc_sn.types import MemoryState, MultiScaleCode
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
@dataclass(frozen=True)
class TEMInputV2:
    """Model-native input payload for one TEM v2 step."""

    observation_embedding: list[Tensor]
    previous_action: Tensor
    episode_start: Optional[Tensor] = None
    landmark_id: Optional[Tensor] = None


@dataclass(frozen=True)
class TEMOutputV2:
    """Model-native output bundle for one TEM v2 step."""

    pred_codes: PredCodes
    grid_codes: GridCodes
    place_codes: PlaceCodes


# =============================================================================
class ModelSettingsV2(BaseModel, extra="forbid", strict=False):
    """Canonical TEM v2 model settings.

    This compact schema keeps the environment-facing observation/action
    contract, shared multiscale frequencies, and component-local settings in a
    single model config without reintroducing legacy aliases.
    """

    transition_action_count: int = Field(
        ...,
        ge=1,
        description="Number of discrete actions in the environment.",
    )

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

    hpc: HPCAttentionSettings = Field(..., description="Settings for the HPC attractor module.")
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

    @model_validator(mode="after")
    def validate_model(self) -> "ModelSettingsV2":
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

    hpc: HPCAttentionSettings = Field(..., description="Settings for the attention-based hippocampal module.")
    lec: LECSettings = Field(
        ...,
        description="Settings for the LEC module, including feature filtering parameters.",
    )
    mec: MECSettings = Field(
        ...,
        description="Settings for the MEC module, including path integration and correction parameters.",
    )


    projection_lec: ProjectionSettings = Field(
        default_factory=lambda: ProjectionSettings(mode="tiling", learnable=False),
        description=(
            "Settings for the projection module between LEC and HPC. "
            "This module projects LEC features into the format expected by HPC memory."
        ),
    )
    projection_mec: ProjectionSettings = Field(
        default_factory=lambda: ProjectionSettings(mode="low_rank", learnable=False, rank=[10, 10, 8, 6, 6]),
        # default_factory=lambda: ProjectionSettings(mode="low_rank", learnable=False),
        description=(
            "Settings for the projection module between MEC and HPC. "
            "This module projects MEC abstract location codes into the format expected by HPC memory."
        ),
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
    def from_config(cls, path: Path) -> "ModelSettingsV2":
        """Load model settings from a TOML configuration file."""
        config_map = tomllib.load(Path(path).open("rb"))
        return cls.model_validate(config_map)


# =============================================================================
@dataclass
class TEMStateV2(DetachMixin):
    """Container for the full recurrent TEM state."""

    lec: LECState  # State of the Lateral Entorhinal Cortex
    mec: MECState  # State of the Medial Entorhinal Cortex
    hpc: HPCState  # State of the Hippocampus


# =============================================================================
class TEMModelV2(nn.Module):
    """TEM v2 backbone with a TEM v2-compatible forward contract."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelSettingsV2, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Construct the TEM backbone from the resolved TEM v2 model settings."""
        super().__init__()
        self._config = config
        n_freq, n_actions = config.n_total_freq, config.transition_action_count
        f_initial = config.f_initial

        # Entorhinal Hippocampal Circuit components
        self.hpc = HPCAttention(n_freq, config.f_initial, config.hpc, device=device, dtype=dtype)
        self.mec = MECModel(config.transition_action_count, config.hpc.shape, config.f_initial, config.mec, device=device, dtype=dtype)
        self.lec = LECModel(config.f_initial, config.lec, device=device, dtype=dtype)

        # Projection modules
        self.projections = ProjectionBundle.from_modules(
            mec_to_hpc=(self.mec, self.hpc, config.projections.mec_to_hpc),
            lec_to_hpc=(self.lec, self.hpc, config.projections.lec_to_hpc),
        )

        self.reset_parameters()

    @property
    def config(self) -> ModelSettingsV2:
        """Return the parsed TEM v2 model settings."""
        return self._config

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int, *, memory: Optional[MemoryState] = None,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> TEMStateV2:  # fmt: skip
        """Create an initial recurrent TEM state."""
        memory = memory if memory is not None else self.hpc.init_memory(batch_size=batch_size, device=device)
        lec_state = self.lec.init_state(batch_size, device=device)
        state_mec = self.mec.init_state(batch_size, device=device)
        hpc_state = self.hpc.init_state(batch_size, device=device, memory=memory)
        return TEMStateV2(lec_state, state_mec, hpc_state)

    def reset_state(  # ---------------------------------------------------------------------------
        self, reset_flag: Tensor, state: TEMStateV2,
    ) -> TEMStateV2:  # fmt: skip
        """Reset flagged rows to a fresh episode state while preserving active rows."""
        reset_flag = reset_flag.to(torch.bool).view(-1)
        if not torch.any(reset_flag):
            return state

        return TEMStateV2(
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

    def forward(  # -------------------------------------------------------------------------------
        self, inputs: TEMInputV2, state: Optional[TEMStateV2] = None,
    ) -> tuple[TEMOutputV2, TEMStateV2]:  # fmt: skip
        """Run one TEM step from the current payload and recurrent state.

        ``state`` is assumed to have already been reset for any fresh episode
        rows by the caller. When ``state`` is ``None``, a fresh full-batch state
        is allocated and the normal single-step TEM transition is executed.
        """
        observation_embedding = inputs.observation_embedding[0]
        previous_action = inputs.previous_action
        episode_start = inputs.episode_start
        landmark_id = inputs.landmark_id

        if state is None:
            state = self.init_state(int(observation_embedding.shape[0]), memory=None, device=observation_embedding.device)

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

        # Decode observation logits for the three TEM pathways.
        lec_features_from_place_post = self.projection_lec.inverse(step.place_post)
        lec_features_from_place_retrieved = self.projection_lec.inverse(step.place_retrieved)
        lec_features_from_place_prior = self.projection_lec.inverse(step.place_prior)

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
        model_output = TEMOutputV2(pred_codes=pred_codes, grid_codes=grid_codes_out, place_codes=place_codes_out)
        return model_output, state


# =============================================================================
__all__ = [
    "ModelSettingsV2", "TEMStateV2", "TEMStateV2", "TEMModelV2",
    "TEMInputV2", "TEMOutputV2",
    "Batch", "ObsLogits",
]  # fmt: skip
