from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypeAlias

import torch
from pydantic import AliasChoices, BaseModel, Field, computed_field, model_validator
from torch import Tensor, nn

from ehc_sn.modules.autoencoder import Autoencoder, AutoencoderSettings
from ehc_sn.modules.hpc import HPCAttention, HPCAttentionSettings, HPCState
from ehc_sn.modules.hpc import SensoryRead as HPCSensoryRead
from ehc_sn.modules.hpc.query_policy import ReadCues, TargetRead
from ehc_sn.modules.lec import LECModel, LECSettings, LECState
from ehc_sn.modules.mec import MECModel, MECSettings, MECState
from ehc_sn.modules.pfc import PFCModel, PFCSettings, PFCState
from ehc_sn.modules.projection import ProjectionModule, ProjectionSettings
from ehc_sn.modules.str import STRModelLinear, STRSettings, STRState
from ehc_sn.modules.transformer import TransformerSequenceSummary, TransformerSequenceSummaryConfig
from ehc_sn.types import Device, Dtype, MemoryState, MultiScaleCode
from ehc_sn.utils import trunc_normal_init_
from ehc_sn.utils.detach import DetachMixin
from ehc_sn.utils.tensor_ops import as_batch_column, as_optional_batch_column, as_optional_feature_matrix

# Community-standard map-style batch: plain dict returned by MazeDataset / DataLoader.
Batch: TypeAlias = dict[str, Tensor]
GridCodes = tuple[MultiScaleCode, MultiScaleCode]  # (posterior, prior)
PlaceCodes = tuple[MultiScaleCode, MultiScaleCode, Optional[MultiScaleCode]]  # (posterior, prior, sensory-cued retrieval)


# =================================================================================================
class ModelSettings_V2(BaseModel, extra="forbid", strict=False):
    """Canonical EHC v2 model settings.

    EHC v2 preserves the staged TEM memory path while adding a fixed semantic
    cortical workspace, a dedicated cortical sequence summary path, and a
    target-bank cortical cue pathway.
    """


# =================================================================================================
# TODO: Move these to EHC controller or so
@dataclass
class EHCInputs(DetachMixin):
    """Container for the inputs to the EHC backbone, preprocessed and ready for use by the regions."""

    observation: Tensor
    previous_action: Tensor
    episode_start: Tensor
    landmark_id: Tensor
    external_context: Tensor
    goal_cue: Optional[Tensor] = None

    def __init__(self, batch: Batch) -> None:
        observation = batch["observation"].to(torch.float32)
        batch_size = int(observation.shape[0])
        device = observation.device

        self.previous_action = as_batch_column(
            batch["previous_action"],
            batch_size=batch_size,
            device=device,
            dtype=torch.int64,
            name="previous_action",
        )
        self.episode_start = as_optional_batch_column(
            batch.get("episode_start"),
            batch_size=batch_size,
            device=device,
            dtype=torch.bool,
            name="episode_start",
            fill_value=False,
        )
        self.landmark_id = as_optional_batch_column(
            batch.get("landmark_id"),
            batch_size=batch_size,
            device=device,
            dtype=torch.int64,
            name="landmark_id",
            fill_value=0,
        )
        self.external_context = as_optional_feature_matrix(
            batch.get("external_context"),
            batch_size=batch_size,
            width=self.config.external_context_dim,
            device=device,
            dtype=torch.float32,
            name="external_context",
        )
        self.goal_cue = as_optional_batch_column(
            batch.get("goal_cue"),
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
            name="goal_cue",
            fill_value=0.0,
        )

    @property
    def batch_size(self) -> int:
        """Return the batch size of the inputs."""
        return self.observation.shape[0]

    @property
    def device(self) -> torch.device:
        """Return the device of the inputs."""
        return self.observation.device


@dataclass
class ObsLogits:
    """Container for the observation prediction logits from the EHC backbone."""

    inference: Tensor
    retrieved: Tensor
    ancestral: Tensor


@dataclass
class RewardLogis:
    """Container for the reward prediction logits from the EHC backbone."""

    reward: Tensor


@dataclass
class MotorLogits:
    """Container for the motor action logits from the EHC backbone."""

    motor: Tensor


# =================================================================================================
@dataclass
class EHCState(DetachMixin):
    """Container for the recurrent state owned by EHC region modules only."""

    pfc: PFCState  # State of the Prefrontal Cortex
    str: STRState  # State of the Striatum
    lec: LECState  # State of the Lateral Entorhinal Cortex
    mec: MECState  # State of the Medial Entorhinal Cortex
    hpc: HPCState  # State of the Hippocampus


# =================================================================================================
class EHCModelV2(nn.Module):
    """EHC v1 backbone with a structured step output and no environment stepping."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ModelSettings_V2, *,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        """Construct the EHC backbone from the resolved EHC v1 model settings.

        The staged TEM memory path is preserved exactly. Cortical context is
        introduced through a slot-derived cue proposal, a reinstated target-bank
        cortical trace, and a transient routed control cue that biases replay
        without being written back into hippocampal memory.
        """
        super().__init__()
        self._config = config
        n_freq, n_actions = config.n_total_freq, config.action_count
        f_initial = config.f_initial

        # Sensory transducer plus region modules / dynamical cores.
        self.autoencoder = Autoencoder(config.observation_dim, config.lec.feature_dim, config.autoencoder)
        self.pfc = PFCModel(config.pfc, device=device, dtype=dtype)  # Reasoning module with embedded inputs
        self.str = STRModelLinear(config.str, device=device, dtype=dtype)  # Reward estimator
        self.hpc = HPCAttention(n_freq, f_initial, config.hpc, device=device, dtype=dtype)
        self.mec = MECModel(n_actions, config.hpc.shape, f_initial, config.mec, device=device, dtype=dtype)
        self.lec = LECModel(f_initial, config.lec, device=device, dtype=dtype)

        # Model-local grouped owners for projections, pathways, workspace packing, and heads.
        self.projections = ProjectionBundle(mec=self.mec, lec=self.lec, hpc=self.hpc, config=config)
        self.pathways = PathwayBundle(config, device=device, dtype=dtype)
        self.workspace = WorkspaceBundle(config, device=device, dtype=dtype)
        self.heads = HeadBundle(config, device=device, dtype=dtype)

        # Slot surfaces
        self.task_external_context = nn.Linear(config.external_context_dim, hidden_size, bias=False)
        self.task_episode_start = nn.Linear(1, hidden_size, bias=False)
        self.task_sequence = nn.Linear(hidden_size, hidden_size, bias=False)
        self.task_goal = nn.Linear(1, hidden_size, bias=False)  # optional; only if truly observable
        self.state_prev_action = nn.Embedding(config.action_count, hidden_size)
        self.state_slot_fuser = nn.Linear(config.lec.feature_dim + hpc_flat_dim + hidden_size, hidden_size)
        self.recall_slot_fuser = nn.Linear(hpc_flat_dim, hidden_size)
        self.workspace_slot_embedding = nn.Embedding(3, hidden_size)

        self.reset_parameters()

    @property
    def config(self) -> ModelSettings_V2:
        """Return the parsed EHC v1 model settings."""
        return self._config

    def reset_parameters(self) -> None:
        """Initialize the grouped learnable surfaces owned directly by EHC."""
        init_std = self.config.init_std
        self.pathways.reset_parameters(init_std=init_std)
        self.workspace.reset_parameters(init_std=init_std)
        self.heads.reset_parameters(init_std=init_std)

    def init_state(  # ----------------------------------------------------------------------------
        self, batch_size: int, *, memory: Optional[MemoryState] = None,
        device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> EHCState:  # fmt: skip
        """Create an initial recurrent EHC state."""
        memory = memory if memory is not None else self.hpc.init_memory(batch_size=batch_size, device=device)
        return EHCState(
            pfc=self.pfc.init_state(batch_size),
            str=self.str.init_state(batch_size),
            lec=self.lec.init_state(batch_size, device=device),
            mec=self.mec.init_state(batch_size, device=device),
            hpc=self.hpc.init_state(batch_size, device=device, memory=memory),
        )

    def reset_state(  # ---------------------------------------------------------------------------
        self, reset_flag: Tensor, state: EHCState,
    ) -> EHCState:  # fmt: skip
        """Reset flagged rows to a fresh episode state while preserving active rows."""
        device = state.lec.features[0].device
        reset_flag = reset_flag.to(device=device, dtype=torch.bool).view(-1)
        if not torch.any(reset_flag):
            return state

        return EHCState(
            pfc=self.pfc.reset_state(state.pfc, reset_flag),
            str=self.str.reset_state(state.str, reset_flag),
            lec=self.lec.reset_state(state.lec, reset_flag),
            mec=self.mec.reset_state(state.mec, reset_flag),
            hpc=self.hpc.reset_state(state.hpc, reset_flag),
        )

    def set_runtime(  # ---------------------------------------------------------------------------
        self, eta: float, hebbian_decay: float, p2g_uncertainty_offset: float,
    ) -> None:  # fmt: skip
        """Apply runtime parameters to MEC and HPC without touching cortical adapters."""
        self.mec.set_runtime(p2g_uncertainty_offset=p2g_uncertainty_offset)
        self.hpc.set_runtime(eta=eta, hebbian_decay=hebbian_decay)

    def forward(  # -------------------------------------------------------------------------------
        self,
        batch: Batch,
        state: Optional[EHCState] = None,
    ) -> tuple[ObsLogits, RewardLogis, MotorLogits, EHCState]:
        """Run one forward step of the EHC backbone, returning structured outputs."""

        # 0. Unpack the batch and prepare the state, ensuring batch-alignment and extracting memories.
        inputs: EHCInputs = EHCInputs(batch)
        if state is None:  # Init state if not provided
            state = self.init_state(memory=None, batch_size=inputs.batch_size, device=inputs.device)
        else:  # Clone the state to preserve immutability contract
            state = ...  # TODO: state.clone(device=inputs.device)

        # 1. Unpack working memory and episodic memory for ease of use in the step.
        wm, em = state.pfc.memory, state.hpc.memory
        theta_cls = state.pfc.memory.z_H[:, 0]  # [B, D] summary token from PFC memory for cue proposal
        c_query = [proj(theta_cls) for proj in self.projections.pfc_to_hpc_c]  # List of [B, f_i] queries

        # 2. Pure TEM sensory loop.
        obs_embedding = self.autoencoder.encode(inputs.observation)
        x_post, state.lec = self.lec.inference(obs_embedding, state.lec)
        p_query_from_obs = self.projections.lec_to_hpc_x(x_post)

        g_prior, state.mec = self.mec.generative(inputs.prev_action, inputs.episode_start, inputs.landmark_id, state.mec)
        p_query_from_g_prior = self.projections.mec_to_hpc_g(g_prior)

        p_sensory_read = self.hpc.recall(
            read_cues=ReadCues(families={"x": p_query_from_obs, "g": p_query_from_g_prior}),
            state=state.hpc,
            role="",
            read=TargetRead(kind="target", sources=("g",), target="x", target_init="x"),
        )
        g_post, state.mec = self.mec.inference(p_sensory_read, landmark_id=inputs.landmark_id, state=state.mec)
        p_query_from_g_post = self.projections.mec_to_hpc_g(g_post)

        # 3.
        p_prior_read = self.hpc.recall(
            read_cues=ReadCues(families={"g": p_query_from_g_prior}),
            state=state.hpc,
            role="generative",
            read=TargetRead(kind="target", sources=("g",), target="x"),
        )
        p_prior, state.hpc = self.hpc.generative(p_prior_read, state.hpc)

        # 4. c-biased replay branch, deterministic for PFC bias path
        p_replay_read = self.hpc.recall(
            read_cues=ReadCues(families={"g": p_query_from_g_post, "c": c_query}),
            state=state.hpc,
            role="generative",
            read=TargetRead(kind="target", sources=("g",), target="x", target_init="c"),
        )
        p_retrieved, state.hpc = self.hpc.generative(p_replay_read, state.hpc)

        # 5. Update the PFC state with the replay token as a transient bias
        workspace_tokens = torch.stack(
            [
                self.workspace.build_task_slot(inputs),
                self.workspace.build_state_slot(obs_embedding, p_post, inputs),
                self.workspace.build_recall_slot(p_replay_read),
            ],
            dim=1,
        )
        slots_ids = torch.arange(3, device=workspace_tokens.device)
        workspace_tokens = workspace_tokens + self.workspace_slot_embedding(slots_ids).unsqueeze(0)
        state.pfc, z_H, q_logits = self.pfc(workspace_tokens, state.pfc)

        # 6. Memory write to HPC
        p_post, state.hpc = self.hpc.inference(p_query_from_obs, p_query_from_g_post, state.hpc)
        payload = WritePayload(generative=p_replay_read, inference=p_sensory_read)
        state.hpc = self.hpc.update(p_post, payload, state.hpc)

        # 7. Control outputs from STR and heads
        theta_summary = z_H[:, 0]
        motor_logits = self.motor_action_head(theta_summary.to(torch.float32))
        state.str, r_logits = self.str(theta_summary.detach(), q_logits, state.str)

        # 8. Encapsulate outputs in structured containers for ease of use by the controller.
        obs_logits = ObsLogits(
            inference=self.lec.generative(p_post),
            retrieved=self.lec.generative(p_retrieved),
            ancestral=self.lec.generative(p_prior),
        )
        r_logits = RewardLogis(reward=r_logits)
        motor_logits = MotorLogits(motor=motor_logits)

        # Return
        return obs_logits, r_logits, motor_logits, state

    def build_task_slot(  # ---------------------------------------------------
        self,
        inputs: EHCInputs,
    ) -> Tensor:
        """Build the task context token from the current state and the inputs.
        task_token should contain only variables that matter for control:
         - external context
         - episode start / phase
         - explicit goal cue only if the environment provides one
        """
        slot = self.task_external_context(inputs.external_context)
        slot = slot + self.task_episode_start(inputs.episode_start)
        if inputs.goal_cue is not None:
            slot = slot + self.task_goal(inputs.goal_cue)
        return slot

    def build_state_slot(  # -------------------------------------------------
        self,
        obs_embedding: Tensor,
        p_post: MultiScaleCode,
        inputs: EHCInputs,
    ) -> Tensor:
        """Build the state context token from the current state and the inputs.
        state_token should contain the current integrated situation:
         - current observation embedding
         - pure TEM posterior, ideally p_post
         - previous action
        """
        state_features = torch.cat(
            [
                obs_embedding,
                utils.flatten_multiscale(p_post),
                self.state_prev_action(inputs.prev_action.squeeze(-1)),
            ],
            dim=1,
        )
        return self.state_slot_fuser(state_features)

    def build_recall_slot(  # -------------------------------------------------
        self,
        p_replay_read: MultiScaleCode,
    ) -> Tensor:
        """Build the recall context token from the current state and the inputs.
        recall_token should contain the ...:
         - deterministic replay result from HPC
        """
        recall_features = utils.flatten_multiscale(p_replay_read)
        return self.recall_slot_fuser(recall_features)


# =================================================================================================
__all__ = [
    "ModelSettings_V2", "EHCState", "EHCModelV2",
    "Batch", "ObsLogits", "GridCodes", "PlaceCodes",
]  # fmt: skip
