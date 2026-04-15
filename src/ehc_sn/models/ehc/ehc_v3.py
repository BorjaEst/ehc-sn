from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from pydantic import BaseModel
from torch import Tensor, nn

from ehc_sn.models.ehc.core import *
from ehc_sn.modules.hpc import HPCAttention, HPCAttentionSettings, HPCState, WritePayload
from ehc_sn.modules.hpc.query_policy import ReadCues, TargetRead
from ehc_sn.modules.lec import LECModel, LECSettings, LECState
from ehc_sn.modules.mec import MECModel, MECSettings, MECState
from ehc_sn.modules.pfc import PFCModel, PFCSettings, PFCState
from ehc_sn.modules.projection import ProjectionBundle, ProjectionSettings
from ehc_sn.modules.str import STRModelLinear, STRSettings, STRState
from ehc_sn.types import Device, Dtype, MemoryState
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class ModelSettings_V3(BaseModel, extra="forbid", strict=False):
    """Canonical EHC v3 model settings."""


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
    """Task-agnostic EHC v3 backbone over model-ready latent observations."""

    def __init__(  # ------------------------------------------------------------------------------
        self,
        config: ModelSettings_V3,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        """Construct the EHC backbone from the resolved EHC v3 model settings."""
        super().__init__()
        self._config = config
        n_freq = len(config.hpc.shape)
        transition_action_count = getattr(config, "transition_action_count", getattr(config, "action_count"))
        f_initial = config.f_initial

        # Initialize EHC region modules and inter-region projection modules in one place for clean state management
        self.pfc = PFCModel(config.pfc, device=device, dtype=dtype)
        self.str = STRModelLinear(config.str, device=device, dtype=dtype)
        self.hpc = HPCAttention(n_freq, f_initial, config.hpc, device=device, dtype=dtype)
        self.mec = MECModel(transition_action_count, config.hpc.shape, f_initial, config.mec, device=device, dtype=dtype)
        self.lec = LECModel(f_initial, config.lec, device=device, dtype=dtype)

        # Projections between regions
        self.projections = ProjectionBundle.from_modules(
            lec_to_hpc_x=(self.lec, self.hpc, self.config.projections.lec_to_hpc_x),
            mec_to_hpc_g=(self.mec, self.hpc, self.config.projections.mec_to_hpc_g),
            pfc_to_hpc_c=(self.pfc, self.hpc, self.config.projections.pfc_to_hpc_c),
        )

        #
        self.workspace = WorkspaceBundle(...)
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
    ) -> EHCState:
        """Create an initial recurrent EHC state."""
        memory = memory if memory is not None else self.hpc.init_memory(batch_size=batch_size, device=device)
        return EHCState(
            pfc=self.pfc.init_state(batch_size, device=device),
            str=self.str.init_state(batch_size, device=device),
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
        inputs: EHCV3Input,
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

        output = EHCV3Output(
            control=EHCV3Control(theta_summary=z_H[:, 0], control_logits=control_logits),
            content=EHCV3Content(bank_tokens=bank_tokens),
            g_codes=GridCodes(post=g_post, prior=g_prior),
            p_codes=PlaceCodes(post=p_post, prior=p_prior_read, sensory=p_sensory_read, replay=p_replay_read),
        )
        return output, state


# =================================================================================================
__all__ = ["ModelSettings_V3", "EHCState", "EHCModelV3", "EHCV3Input", "EHCV3Output"]
