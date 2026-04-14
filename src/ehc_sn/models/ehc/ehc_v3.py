from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeAlias

import torch
from pydantic import BaseModel
from torch import Tensor, nn

from ehc_sn.models.ehc.core import EHCContentBankBuilder, EHCV3Codes, EHCV3Content, EHCV3Control, EHCV3Output, EHCWorkspaceWriter
from ehc_sn.models.ehc.core.ehc_base import *
from ehc_sn.models.ehc.core.ehc_base import GridCodes, PlaceCodes
from ehc_sn.modules.autoencoder import Autoencoder, AutoencoderSettings
from ehc_sn.modules.hpc import HPCAttention, HPCAttentionSettings, HPCState, WritePayload
from ehc_sn.modules.hpc.query_policy import ReadCues, TargetRead
from ehc_sn.modules.lec import LECModel, LECSettings, LECState
from ehc_sn.modules.mec import MECModel, MECSettings, MECState
from ehc_sn.modules.pfc import PFCModel, PFCSettings, PFCState
from ehc_sn.modules.projection import ProjectionModule, ProjectionSettings
from ehc_sn.modules.str import STRModelLinear, STRSettings, STRState
from ehc_sn.types import Device, Dtype, EHCStepInput, MemoryState, MultiScaleCode
from ehc_sn.utils import trunc_normal_init_
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class ModelSettings_V3(BaseModel, extra="forbid", strict=False):
    """Canonical EHC v3 model settings.

    EHC v3 preserves the staged TEM memory path while adding a fixed semantic
    cortical workspace, a dedicated cortical sequence summary path, and a
    target-bank cortical cue pathway.
    """


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
class EHCModelV3(nn.Module):
    """EHC v3 backbone with a structured step output and no environment stepping."""

    def __init__(  # ------------------------------------------------------------------------------
        self,
        config: ModelSettings_V3,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        """Construct the EHC backbone from the resolved EHC v3 model settings.

        The staged TEM memory path is preserved exactly. Cortical context is
        introduced through a slot-derived cue proposal, a reinstated target-bank
        cortical trace, and a transient routed control cue that biases replay
        without being written back into hippocampal memory.
        """
        super().__init__()
        self._config = config
        n_freq = len(config.hpc.shape)
        transition_action_count = getattr(config, "transition_action_count", getattr(config, "action_count"))
        f_initial = config.f_initial

        # Sensory transducer plus region modules / dynamical cores.
        self.autoencoder = Autoencoder(config.observation_dim, config.lec.feature_dim, config.autoencoder)
        self.pfc = PFCModel(config.pfc, device=device, dtype=dtype)  # Reasoning module with embedded inputs
        self.str = STRModelLinear(config.str, device=device, dtype=dtype)  # Reward estimator
        self.hpc = HPCAttention(n_freq, f_initial, config.hpc, device=device, dtype=dtype)
        self.mec = MECModel(transition_action_count, config.hpc.shape, f_initial, config.mec, device=device, dtype=dtype)
        self.lec = LECModel(f_initial, config.lec, device=device, dtype=dtype)

        # Model-local grouped owners for projections, pathways, workspace packing, and heads.
        self.projections = ProjectionBundle(mec=self.mec, lec=self.lec, hpc=self.hpc, config=config)
        self.workspace = WorkspaceBundle(config, device=device, dtype=dtype)
        self.content_bank = ContentBankBuilder(...)

        self.reset_parameters()

    @property
    def config(self) -> ModelSettings_V3:
        """Return the parsed EHC v3 model settings."""
        return self._config

    def reset_parameters(self) -> None:
        """Initialize the grouped learnable surfaces owned directly by EHC."""
        self.projections.reset_parameters()
        self.workspace.reset_parameters()
        self.content_bank.reset_parameters()

    def init_state(  # ----------------------------------------------------------------------------
        self,
        batch_size: int,
        *,
        memory: Optional[MemoryState] = None,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> EHCState:
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
        self,
        reset_flag: Tensor,
        state: EHCState,
    ) -> EHCState:
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
        self,
        eta: float,
        hebbian_decay: float,
        p2g_uncertainty_offset: float,
    ) -> None:
        """Apply runtime parameters to MEC and HPC without touching cortical adapters."""
        self.mec.set_runtime(p2g_uncertainty_offset=p2g_uncertainty_offset)
        self.hpc.set_runtime(eta=eta, hebbian_decay=hebbian_decay)

    def forward(  # -------------------------------------------------------------------------------
        self,
        inputs: EHCStepInput,
        state: Optional[EHCState] = None,
    ) -> tuple[EHCV3Output, EHCState]:
        """Run one forward step of the EHC backbone and return task-agnostic latents."""

        # 0. Prepare the state, ensuring batch-alignment and extracting memories.
        if state is None:  # Init state if not provided
            state = self.init_state(memory=None, batch_size=inputs.batch_size, device=inputs.device)
        else:  # Clone the state to preserve immutability contract
            state = ...  # TODO: state.clone(device=inputs.device)

        # 1. Unpack working memory and episodic memory for ease of use in the step.
        theta_cls = state.pfc.memory.z_H[:, 0]  # [B, D] summary token from PFC memory for cue proposal
        c_query = [proj(theta_cls) for proj in self.pfc_to_hpc_c]

        # 2. Pure TEM sensory loop.
        x_post, state.lec = self.lec.inference(inputs.obs_embedding, state.lec)
        p_query_from_obs = self.projections.lec_to_hpc_x(x_post)

        g_prior, state.mec = self.mec.generative(inputs.previous_action, inputs.episode_start, inputs.landmark_id, state.mec)
        p_query_from_g_prior = self.projections.mec_to_hpc_g(g_prior)

        p_sensory_read = self.hpc.recall(
            read_cues=ReadCues(families={"x": p_query_from_obs, "g": p_query_from_g_prior}),
            state=state.hpc,
            role="inference",
            read=TargetRead(kind="target", sources=("g",), target="x", target_init="x"),
        )
        g_post, state.mec = self.mec.inference(p_sensory_read, landmark_id=inputs.landmark_id, state=state.mec)
        p_query_from_g_post = self.projections.mec_to_hpc_g(g_post)

        # 3. Deterministic prior/replay reads plus the pure TEM posterior.
        p_prior_read = self.hpc.recall(
            read_cues=ReadCues(families={"g": p_query_from_g_prior}),
            state=state.hpc,
            role="generative",
            read=TargetRead(kind="target", sources=("g",), target="x"),
        )
        p_replay_read = self.hpc.recall(
            read_cues=ReadCues(families={"g": p_query_from_g_post, "c": c_query}),
            state=state.hpc,
            role="generative",
            read=TargetRead(kind="target", sources=("g", "c"), target="x"),
        )
        p_post, state.hpc = self.hpc.inference(p_query_from_obs, p_query_from_g_post, state.hpc)

        # 4. Build the explicit 3-slot workspace and run PFC reasoning.
        workspace_tokens = self.workspace(inputs, p_post, p_replay_read)
        state.pfc, z_H, control_logits = self.pfc(workspace_tokens, state.pfc)

        # 5. Commit the HPC write and build the content bank.
        payload = WritePayload(generative=p_replay_read, inference=p_sensory_read)
        state.hpc = self.hpc.update(p_post, payload, state.hpc)
        bank_tokens = self.content_bank(z_H=z_H, p_post=p_post, p_replay_read=p_replay_read)

        # 6. Return the strict task-agnostic output contract.
        theta_summary = z_H[:, 0]
        obs_logits = EHCV3ObsLogits(
            ancestral=self._decode_observation_logits_from_place(p_post),
            inference=self._decode_observation_logits_from_place(p_replay_read),
            recall=self._decode_observation_logits_from_place(p_prior_read),
        )
        g_codes = GridCodes(post=g_post, prior=g_prior)
        p_codes = PlaceCodes(post=p_post, prior=p_prior_read, sensory=p_sensory_read, replay=p_replay_read)

        # end. Return the full backbone output
        output = EHCV3Output(
            control=EHCV3Control(theta_summary=theta_summary, control_logits=control_logits),
            content=EHCV3Content(bank_tokens=bank_tokens, obs_logits=obs_logits),
            codes=EHCV3Codes(grid=g_codes, place=p_codes),
        )
        return output, state


# =================================================================================================
__all__ = ["ModelSettings_V3", "EHCState", "EHCModelV3", "EHCV3Output"]
