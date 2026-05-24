"""EHC v2 backbone model, settings, state, and forward I/O.

EHC v2 (Entorhinal-Hippocampal Circuit, version 3) extends EHC v2 with a
Zheng-style current-step cortical cue path.  It is otherwise identical to v2.

Cue-timing contract (V2 — explicit difference from V2):
    V2 derives the cortical cue from *previous*-step ``state.pfc.summary``.
    V2 derives the cortical cue from the *current*-step PFC output via an
    explicit two-stage within-step schedule:

        Stage 1 — Cue generation pass:
            Run the full TEM-style bottom-up pipeline (g_prior, x_query,
            p_sensory_read, g_post, p_prior, p_retrieved, p_post) to ground
            the hippocampal state.  Build a provisional PFC body workspace
            using p_post as state, p_retrieved as a non-contextual replay
            baseline, and a projected zero vector as the cue.  Run one PFC
            step to obtain the current-step cortical summary.
            Set c_prop = pfc_to_hpc(first-pass summary), c_use = c_prop.

        Stage 2 — Cue-conditioned pass:
            Use c_use in the contextual replay read (TargetRead, sources g and c,
            target x) to obtain p_replay_read.  Build the final PFC body workspace
            with p_post as state, p_replay_read as replay, and c_use as cue.
            Run a second PFC step STARTING FROM the first-pass recurrent PFC state
            so the final cortical state reflects cue-conditioned replay.

    This file is intentionally self-contained.  The cue-timing difference is
    implemented locally and must not be hidden in shared helpers or in ehc_base.py.

Bank-c semantics (unchanged from V2):
    HPCAttention bank ``c`` is a place-keyed contextual auxiliary bank that
    shares the hippocampal write key with the default bank.

        - ``c_prop``: cortical cue proposal — current-step PFC summary (stage 1)
            projected into the hippocampal multi-frequency cue family via
            ``pfc_to_hpc``.  (Contrast V2: ``c_prop`` = previous-step summary.)
        - ``c_mem``:  reinstated contextual evidence from bank ``c`` (deferred in
            V2; set to ``None``).
        - ``c_use``:  the routed cue used for replay bias and bank-c writes.
            In V2: ``c_use = c_prop``.

PFC body workspace layout (size = ``pfc.seq_length``):
    Fixed slot 0 - ``state``   : projected current grounded-place (p_post).
    Fixed slot 1 - ``replay``  : projected contextual replay code.
    Fixed slot 2 - ``cue``     : projected cortical cue (c_use).
    Family ``content`` (size = pfc.seq_length - 3): previous-step recurrent
    cortical substrate from ``state.pfc.workspace.family("content")``.

For the reference config (pfc.seq_length = 36) the content family has 33 slots.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, cast

import torch
from pydantic import BaseModel, Field, model_validator
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.models.ehc.core.ehc_base import (
    FAMILY_CONTENT,
    SLOT_CUE,
    SLOT_REPLAY,
    SLOT_STATE,
    EHCProjectionSettings,
)
from ehc_sn.models.tem.core.tem_base import GridCodes, PlaceCodes
from ehc_sn.modules.hpc import (
    HPCAttention,
    HPCAttentionSettings,
    HPCState,
    WritePayload,
)
from ehc_sn.modules.hpc.query_policy import CueRead, ReadCues, TargetRead
from ehc_sn.modules.lec import LECModel, LECSettings, LECState
from ehc_sn.modules.mec import MECModel, MECSettings, MECState
from ehc_sn.modules.pfc import PFCModel, PFCSettings, PFCState
from ehc_sn.modules.pfc.workspace import (
    FixedSlot,
    SlotFamily,
    WorkspaceLayout,
    WorkspaceSchema,
)
from ehc_sn.modules.projection import (
    ProjectionBundle,
    ProjectionModule,
    flat_endpoint,
    workspace_endpoint,
)
from ehc_sn.modules.str import STRModelLinear, STRSettings, STRState
from ehc_sn.types import MemoryState, MultiScaleCode
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
@dataclass
class EHCControlV2:
    """Control-pathway outputs from the PFC/STR cortico-striatal loop.

    Attributes:
        theta_summary: PFC controller summary token (workspace position 0).
            Shape ``(B, D)`` where ``D = pfc.hidden_size``.
        control_logits: Q-value logits from the PFC auxiliary value head.
            Shape ``(B, n_actions)``.
        reward_prediction: Scalar STR reward prediction.  Shape ``(B,)``.
    """

    theta_summary: Tensor  # (B, D)
    control_logits: Tensor  # (B, n_actions)
    reward_prediction: Tensor  # (B,)

    @property
    def reward_logits(self) -> Tensor:
        """Alias for ``reward_prediction`` (notation-doc compatibility)."""
        return self.reward_prediction


# =============================================================================
@dataclass
class EHCContentV2:
    """Content-pathway outputs from the PFC body workspace.

    These tensors are taken from the post-reasoning PFC workspace (second pass)
    and reflect the working memory state after cue-conditioned replay.

    Attributes:
        state_slot:    PFC body ``state`` slot.  Shape ``(B, D)``.
        replay_slot:   PFC body ``replay`` slot.  Shape ``(B, D)``.
        cue_slot:      PFC body ``cue`` slot.  Shape ``(B, D)``.
        content_slots: PFC ``content`` family (recurrent substrate).
            Shape ``(B, content_size, D)``
            where ``content_size = pfc.seq_length - 3``.
    """

    state_slot: Tensor  # (B, D)
    replay_slot: Tensor  # (B, D)
    cue_slot: Tensor  # (B, D)
    content_slots: Tensor  # (B, content_size, D)


# =============================================================================
@dataclass
class EHCOutputV2(DetachMixin):
    """Task-agnostic output payload for one EHC v2 forward step.

    Attributes:
        control:     PFC/STR control-pathway outputs.
        content:     PFC body workspace content outputs.
        grid_codes:  Named MEC grid codes (prior and posterior).
        place_codes: Named HPC place codes (inference, ancestral, retrieved, sensory).
    """

    control: EHCControlV2
    content: EHCContentV2
    grid_codes: GridCodes
    place_codes: PlaceCodes


# =============================================================================
@dataclass
class EHCInputV2(DetachMixin):
    """Task-agnostic input payload for EHC v2 forward steps.

    Attributes:
        observation_embedding:   Multi-scale sensory observation codes.
            Length ``n_freq``; each tensor shape ``(B, lec_feature_dim)``.
        previous_action: Previous transition action indices.
            Shape ``(B,)`` or ``(B, 1)``.
        episode_start:   Optional binary episode-start flags.
            Shape ``(B,)`` or ``(B, 1)``.
        landmark_id:     Optional current-cell landmark identifiers.
            Shape ``(B,)`` or ``(B, 1)``.
    """

    observation_embedding: MultiScaleCode
    previous_action: Tensor
    episode_start: Optional[Tensor] = None
    landmark_id: Optional[Tensor] = None


# =============================================================================
@dataclass
class EHCStateV2(DetachMixin):
    """Container for the full recurrent state across all EHC region modules.

    Attributes:
        pfc: State of the Prefrontal Cortex reasoning module.
        str: State of the Striatum reward/value module.
        lec: State of the Lateral Entorhinal Cortex sensory pathway.
        mec: State of the Medial Entorhinal Cortex structural dynamics.
        hpc: State of the Hippocampus memory module.
    """

    pfc: PFCState
    str: STRState
    lec: LECState
    mec: MECState
    hpc: HPCState


# =============================================================================
class ModelSettingsV2(BaseModel, extra="forbid", strict=False):
    """Canonical EHC v2 model settings.

    All architectural dimensions are resolved from this config; no magic
    numbers appear in ``EHCModelV2``.
    """

    transition_action_count: int = Field(
        ...,
        ge=1,
        description="Number of discrete transition actions in the environment.",
    )
    internal_action_count: int = Field(
        ...,
        ge=1,
        description="Number of internal control actions scored by the PFC/STR control path.",
    )
    external_context_dim: int = Field(
        default=1,
        ge=1,
        description="Width of the optional external context payload.",
    )
    f_initial: list[float] = Field(
        default_factory=lambda: [0.99, 0.3, 0.09, 0.5, 0.4],
        min_length=1,
        description="Initial feature frequencies resolved across MEC and HPC modules.",
    )

    hpc: HPCAttentionSettings = Field(
        ..., description="Settings for the attention-based hippocampal memory."
    )
    lec: LECSettings = Field(
        ..., description="Settings for the LEC sensory pathway."
    )
    mec: MECSettings = Field(
        ..., description="Settings for the MEC structural dynamics."
    )
    pfc: PFCSettings = Field(
        ..., description="Settings for the PFC reasoning module."
    )
    str: STRSettings = Field(
        ..., description="Settings for the STR reward/value head."
    )

    projections: EHCProjectionSettings = Field(
        default_factory=EHCProjectionSettings,
        description="Inter-region multiscale projection settings (LEC->HPC, MEC->HPC).",
    )

    @model_validator(mode="after")
    def _validate_pfc_seq_length(self) -> "ModelSettingsV2":
        if self.pfc.seq_length < 4:
            raise ValueError(
                f"pfc.seq_length must be >= 4 (3 fixed body slots + at least 1 content slot), "
                f"got {self.pfc.seq_length}."
            )
        return self

    @property
    def hidden_size(self) -> int:
        """Hidden size shared by the PFC workspace and flat interface projectors."""
        return self.pfc.hidden_size

    @property
    def hpc_flat_dim(self) -> int:
        """Flattened hippocampal width across all frequency bands."""
        return sum(self.hpc.shape)

    @property
    def body_schema(self) -> WorkspaceSchema:
        """Body workspace schema for the PFC module (size = pfc.seq_length).

        Fixed slots (indices 0-2 in the body):
            ``state``  - projected current grounded-place code (p_post).
            ``replay`` - projected contextual replay code.
            ``cue``    - projected cortical cue (c_use).
        Family:
            ``content`` with size = pfc.seq_length - 3 (previous-step cortical
            recurrent substrate).

        Example: pfc.seq_length = 36  ->  content.size = 33.
        """
        content_size = self.pfc.seq_length - 3
        return WorkspaceSchema(
            fixed=(
                FixedSlot(SLOT_STATE),
                FixedSlot(SLOT_REPLAY),
                FixedSlot(SLOT_CUE),
            ),
            families=(SlotFamily(FAMILY_CONTENT, content_size),),
        )


# =============================================================================
class EHCModelV2(nn.Module):
    """EHC v2 backbone: TEM circuit with PFC reasoning and STR control.

    Identical to EHC v2 in region modules, projection edges, and output surface.
    The only architectural difference is the cue-timing contract (see module
    docstring).

    Region modules:
        - LEC  sensory-feature pathway (lateral entorhinal cortex).
        - MEC  grid path-integration pathway (medial entorhinal cortex).
        - HPC  attention-based episodic memory (hippocampus).
        - PFC  two-timescale recurrent workspace (prefrontal cortex).
        - STR  reward-prediction head (striatum).

    Projections registered as ``nn.Module`` children:
        Multiscale (ProjectionBundle):
            ``lec_to_hpc``, ``mec_to_hpc`` - aligned band-by-band projections.
        Broadcast flat->multiscale (ProjectionBundle):
            ``pfc_to_hpc``  PFC hidden -> HPC cue family ``c``.
        Workspace-aligned (ProjectionBundle):
            ``hpc_to_pfc``  HPC flat workspace (state/replay/cue) -> PFC hidden;
                            one independent parameter block per fixed role.
    """

    def __init__(  # ------------------------------------------------------------------
        self,
        config: ModelSettingsV2,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        """Construct EHC v2 from resolved model settings."""
        super().__init__()
        self._config = config
        n_freq = len(config.hpc.shape)
        hidden_size = config.hidden_size
        hpc_flat = config.hpc_flat_dim

        # ---- Core region modules -------------------------------------------
        self.pfc = PFCModel(config.pfc, device=device, dtype=dtype)
        self.str = STRModelLinear(config.str, device=device, dtype=dtype)
        self.hpc = HPCAttention(
            n_freq, config.f_initial, config.hpc, device=device, dtype=dtype
        )
        self.mec = MECModel(
            config.transition_action_count,
            config.hpc.shape,
            config.f_initial,
            config.mec,
            device=device,
            dtype=dtype,
        )
        self.lec = LECModel(
            config.f_initial, config.lec, device=device, dtype=dtype
        )

        # ---- Inter-region projections (LEC->HPC, MEC->HPC, PFC->HPC, HPC->PFC) ----
        # LEC/MEC expose aligned multiscale codes; PFC exposes a flat public summary;
        # the HPC->PFC reverse edge uses a workspace-aligned edge with one independent
        # parameter block per fixed role (state, replay, cue).
        _hpc_ws_from = workspace_endpoint(
            hpc_flat, fixed=[SLOT_STATE, SLOT_REPLAY, SLOT_CUE]
        )
        _hpc_ws_to = workspace_endpoint(
            hidden_size, fixed=[SLOT_STATE, SLOT_REPLAY, SLOT_CUE]
        )
        self.projections = ProjectionBundle.from_modules(
            lec_to_hpc=(self.lec, self.hpc, config.projections.lec_to_hpc),
            mec_to_hpc=(self.mec, self.hpc, config.projections.mec_to_hpc),
            pfc_to_hpc=(
                flat_endpoint(hidden_size),
                self.hpc,
                config.projections.pfc_to_hpc,
            ),
            hpc_to_pfc=(
                _hpc_ws_from,
                _hpc_ws_to,
                config.projections.hpc_to_pfc,
            ),
        )

        self.reset_parameters()

    @property
    def config(self) -> ModelSettingsV2:
        """Return the parsed EHC v2 model settings."""
        return self._config

    @property
    def mec_to_hpc(self) -> ProjectionModule:
        """MEC-to-HPC projection edge."""
        return cast(ProjectionModule, self.projections["mec_to_hpc"])

    @property
    def lec_to_hpc(self) -> ProjectionModule:
        """LEC-to-HPC projection edge."""
        return cast(ProjectionModule, self.projections["lec_to_hpc"])

    @property
    def pfc_to_hpc(self) -> nn.Module:
        """PFC-summary-to-HPC contextual cue projection edge."""
        return self.projections["pfc_to_hpc"]

    @property
    def hpc_to_pfc(self) -> nn.Module:
        """HPC workspace-to-PFC projection edge (one parameter block per fixed role)."""
        return self.projections["hpc_to_pfc"]

    def reset_parameters(self) -> None:
        """Reset all projection parameters owned directly by EHC."""
        self.projections.reset_parameters()

    def init_state(  # -----------------------------------------------------------
        self,
        batch_size: int,
        *,
        memory: Optional[MemoryState] = None,
        device: Optional[Device] = None,
    ) -> EHCStateV2:
        """Create an initial full-batch recurrent EHC v2 state.

        Args:
            batch_size: Number of parallel sequences.
            memory: Optional pre-built HPC memory state.  When ``None`` a fresh
                empty memory is allocated on ``device``.
            device: Target device for all fresh state tensors.

        Returns:
            Freshly initialized :class:`EHCStateV2`.
        """
        memory = (
            memory
            if memory is not None
            else self.hpc.init_memory(batch_size=batch_size, device=device)
        )
        return EHCStateV2(
            # NOTE: PFCModel.init_state does NOT accept a device kwarg.
            pfc=self.pfc.init_state(
                batch_size, body_schema=self.config.body_schema
            ),
            str=self.str.init_state(batch_size, device=device),
            lec=self.lec.init_state(batch_size, device=device),
            mec=self.mec.init_state(batch_size, device=device),
            hpc=self.hpc.init_state(batch_size, device=device, memory=memory),
        )

    def reset_state(  # ----------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: EHCStateV2,
    ) -> EHCStateV2:
        """Reset flagged batch rows to a fresh episode state.

        Args:
            reset_flag: Boolean tensor of shape ``(B,)``.
            state: Current recurrent state.

        Returns:
            Updated :class:`EHCStateV2` with flagged rows reset to initial values.
        """
        device = state.hpc.cells[0].device
        reset_flag = reset_flag.to(device=device, dtype=torch.bool).view(-1)
        if not torch.any(reset_flag):
            return state

        return EHCStateV2(
            pfc=self.pfc.reset_state(state.pfc, reset_flag),
            str=self.str.reset_state(state.str, reset_flag),
            lec=self.lec.reset_state(state.lec, reset_flag),
            mec=self.mec.reset_state(state.mec, reset_flag),
            hpc=self.hpc.reset_state(state.hpc, reset_flag),
        )

    def set_runtime(  # ----------------------------------------------------------
        self,
        eta: float,
        hebbian_decay: float,
        p2g_uncertainty_offset: float,
    ) -> None:
        """Apply runtime parameters to MEC and HPC (no cortical modules involved)."""
        self.mec.set_runtime(p2g_uncertainty_offset=p2g_uncertainty_offset)
        self.hpc.set_runtime(eta=eta, hebbian_decay=hebbian_decay)

    def forward(  # --------------------------------------------------------------
        self,
        inputs: EHCInputV2,
        state: Optional[EHCStateV2] = None,
    ) -> tuple[EHCOutputV2, EHCStateV2]:
        """Run one EHC v2 step and return architecture-native latents.

        Cue-timing (V2 contract — explicit delta from V2):
            V2 derives ``c_prop`` from *previous*-step ``state.pfc.summary``.
            V2 derives ``c_prop`` from a first PFC pass on the current step,
            implemented via an explicit two-stage within-step schedule (see
            module docstring).  ``state.pfc.summary`` is NOT used as the cue
            source in V2.

        All HPC recall calls in stage 1 are placed before the generative/inference
        updates so they read the prior memory state (same as TEM v2 / EHC v2).
        The contextual replay read in stage 2 reads from the post-update state
        because it requires c_use, which is only available after the first PFC pass.

        Args:
            inputs: Task-agnostic EHC v2 input payload.
            state:  Optional prior recurrent state.  ``None`` allocates a fresh
                state.  Episode resets must be applied by the caller via
                :meth:`reset_state` before this call.

        Returns:
            ``(output, next_state)`` — output first, state second (canonical
            backbone seam ordering from spec-model-interfaces.md).
        """
        observation_embedding = inputs.observation_embedding
        previous_action = inputs.previous_action
        episode_start = inputs.episode_start
        landmark_id = inputs.landmark_id

        # Derive batch size and device from the first sensory code tensor.
        batch_size = int(observation_embedding[0].shape[0])
        device = observation_embedding[0].device

        # 0. Prepare state ----------------------------------------------------
        if state is None:
            state = self.init_state(batch_size, device=device)
        else:  # Preserve the outer-state without truncating autograd
            state = replace(state)

        # --- V2 uses current-step PFC output as the cue source (not previous-step) ---
        # c_prop, c_mem, c_use are set after the first PFC pass (stage 1 below).
        # Do NOT use state.pfc.summary as the cue source here (contrast V2).

        # 1. Path integration: grid prior -------------------------------------
        g_prior, state.mec = self.mec.generative(
            previous_action, episode_start, landmark_id, state=state.mec
        )
        g_query_prior = self.mec_to_hpc(g_prior)

        # 2. LEC sensory encoding ---------------------------------------------
        x_, state.lec = self.lec.inference(observation_embedding, state.lec)
        x_query = self.lec_to_hpc(x_)

        # 3. Sensory-cued recall (read-only on state.hpc) ---------------------
        p_sensory_read = self.hpc.recall(
            read_cues=ReadCues(families={"x": x_query}),
            state=state.hpc,
            role="inference",
            read=CueRead(kind="cue", cue="x"),
        )

        # 4. MEC correction: grid posterior from sensory recall ---------------
        g_post, state.mec = self.mec.inference(
            p_sensory_read, landmark_id, state=state.mec
        )
        g_query_post = self.mec_to_hpc(g_post)

        # 5. Grid-cued ancestral recall (prior, read-only) --------------------
        p_grid_prior_read = self.hpc.recall(
            read_cues=ReadCues(families={"g": g_query_prior}),
            state=state.hpc,
            role="generative",
            read=CueRead(kind="cue", cue="g"),
        )

        # 6. Grid-cued retrieved recall (posterior, read-only) ----------------
        p_grid_post_read = self.hpc.recall(
            read_cues=ReadCues(families={"g": g_query_post}),
            state=state.hpc,
            role="generative",
            read=CueRead(kind="cue", cue="g"),
        )

        # 7. Form place beliefs and update HPC grounded-belief state ----------
        # (In V2 this is step 8; in V2 it precedes the provisional PFC pass so
        # p_post and p_retrieved are available for building the cue-generation
        # workspace.)
        p_prior, state.hpc = self.hpc.generative(
            p_grid_prior_read, state=state.hpc
        )
        p_retrieved, state.hpc = self.hpc.generative(
            p_grid_post_read, state=state.hpc
        )
        p_post, state.hpc = self.hpc.inference(
            x_query, g_query_post, state=state.hpc
        )

        # =====================================================================
        # Stage 1 — Cue-generation PFC pass (V2 explicit delta).
        # Build a provisional workspace with:
        #   state  = projected p_post
        #   replay = projected p_retrieved (non-contextual current-step baseline)
        #   cue    = projected zero (HPC flat, then through hpc_to_pfc)
        #   content = previous-step content family (unchanged external contract)
        # =====================================================================
        p_post_flat: Tensor = torch.cat(p_post, dim=-1)  # (B, hpc_flat)
        p_retrieved_flat: Tensor = torch.cat(
            p_retrieved, dim=-1
        )  # (B, hpc_flat)
        zero_c_flat: Tensor = torch.zeros(
            batch_size,
            self._config.hpc_flat_dim,
            device=device,
            dtype=p_post_flat.dtype,
        )

        # Stack into (B, 3, hpc_flat) and project through the workspace edge.
        hpc_slots_prov: Tensor = torch.stack(
            [p_post_flat, p_retrieved_flat, zero_c_flat], dim=1
        )
        hpc_slots_prov_out: Tensor = self.hpc_to_pfc(
            hpc_slots_prov
        )  # (B, 3, hidden_size)

        state_token_prov: Tensor = hpc_slots_prov_out[:, 0:1, :]  # (B, 1, D)
        replay_token_prov: Tensor = hpc_slots_prov_out[:, 1:2, :]  # (B, 1, D)
        cue_token_prov: Tensor = hpc_slots_prov_out[:, 2:3, :]  # (B, 1, D)

        # Previous-step content family provides the cortical substrate.
        prev_content: Tensor = state.pfc.workspace.family(
            FAMILY_CONTENT
        )  # (B, content_size, D)

        body_tokens_prov: Tensor = torch.cat(
            [state_token_prov, replay_token_prov, cue_token_prov, prev_content],
            dim=1,
        )
        body_layout = WorkspaceLayout.from_schema(self.config.body_schema)
        body_workspace_prov = body_layout.bind(body_tokens_prov)

        # First PFC pass: produces the current-step cortical summary for cue generation.
        pfc_out_cue, state_pfc_1 = self.pfc.step(
            body_workspace_prov, state=state.pfc
        )

        # --- V2 cue: current-step PFC summary (first pass) -------------------
        # c_prop: project current-step (first-pass) PFC summary into the
        #         hippocampal multi-frequency cue family via pfc_to_hpc.
        # c_mem:  reinstated contextual evidence from bank c — deferred in V2.
        # c_use:  routed cue for replay bias and bank-c writes; equals c_prop in V2.
        # NOTE: state.pfc.summary is NOT used here (contrast V2's cue source).
        c_prop: list[Tensor] = cast(
            list[Tensor], self.pfc_to_hpc(pfc_out_cue.summary)
        )
        c_mem: Optional[list[Tensor]] = (
            None  # deferred: bank-c reinstatement not implemented
        )
        c_use: list[Tensor] = (
            c_prop  # V2: use current-step cortical cue proposal
        )

        # =====================================================================
        # Stage 2 — Contextual replay and cue-conditioned PFC pass.
        # =====================================================================

        # 8. Contextual replay retrieval (using current-step c_use) -----------
        # Both grid (g) and cortical cue (c_use) source the replay read.
        # This read happens on the post-update HPC state because it requires
        # c_use, which was only available after the stage-1 PFC pass.
        p_replay_read = self.hpc.recall(
            read_cues=ReadCues(families={"g": g_query_post, "c": c_use}),
            state=state.hpc,
            role="generative",
            read=TargetRead(kind="target", sources=("g", "c"), target="x"),
        )

        # 9. Build final PFC body workspace tokens ----------------------------
        # Replay slot now carries the cue-conditioned contextual read.
        # Cue slot carries c_use (current-step cortical cue).
        # Content family is unchanged (previous-step substrate).
        p_replay_flat: Tensor = torch.cat(
            p_replay_read, dim=-1
        )  # (B, hpc_flat)
        c_use_flat: Tensor = torch.cat(c_use, dim=-1)  # (B, hpc_flat)

        hpc_slots_final: Tensor = torch.stack(
            [p_post_flat, p_replay_flat, c_use_flat], dim=1
        )
        hpc_slots_final_out: Tensor = self.hpc_to_pfc(
            hpc_slots_final
        )  # (B, 3, hidden_size)

        state_token_final: Tensor = hpc_slots_final_out[:, 0:1, :]
        replay_token_final: Tensor = hpc_slots_final_out[:, 1:2, :]
        cue_token_final: Tensor = hpc_slots_final_out[:, 2:3, :]

        body_tokens_final: Tensor = torch.cat(
            [
                state_token_final,
                replay_token_final,
                cue_token_final,
                prev_content,
            ],
            dim=1,
        )
        body_workspace_final = body_layout.bind(body_tokens_final)

        # 10. Second PFC pass — starting from the first-pass recurrent state --
        # This ensures the final cortical state reflects cue-conditioned replay.
        # The public control output, theta_summary, control_logits, and content
        # slots all come from this second (final) pass.
        pfc_out_final, state.pfc = self.pfc.step(
            body_workspace_final, state=state_pfc_1
        )
        control_logits: Tensor = pfc_out_final.q_values

        # 11. STR reward prediction -------------------------------------------
        theta_summary: Tensor = state.pfc.summary.detach()  # (B, D)
        state.str, reward_prediction = self.str(
            theta_summary, control_logits, state.str
        )

        # 12. HPC memory write ------------------------------------------------
        # Bank-c write contract: c_use (current-step cue) is written under key c.
        write_payload = WritePayload(
            generative=p_retrieved,
            inference=p_sensory_read,
            named_writes={"c": c_use},
        )
        state.hpc = self.hpc.update(p_post, write_payload, state=state.hpc)

        # 13. Package and return outputs --------------------------------------
        grid_codes = GridCodes(prior=g_prior, posterior=g_post)
        place_codes = PlaceCodes(
            posterior=p_post,
            prior=p_prior,
            retrieved=p_retrieved,
            sensory=p_sensory_read,
        )

        # EHCContentV2 exposes post-reasoning PFC workspace slots (second pass).
        content = EHCContentV2(
            state_slot=state.pfc.workspace.slot(SLOT_STATE),
            replay_slot=state.pfc.workspace.slot(SLOT_REPLAY),
            cue_slot=state.pfc.workspace.slot(SLOT_CUE),
            content_slots=state.pfc.workspace.family(FAMILY_CONTENT),
        )
        control = EHCControlV2(
            theta_summary=state.pfc.summary,  # (B, D) — gradient-attached summary
            control_logits=control_logits,
            reward_prediction=reward_prediction,
        )
        return (
            EHCOutputV2(
                control=control,
                content=content,
                grid_codes=grid_codes,
                place_codes=place_codes,
            ),
            state,
        )


# =============================================================================
__all__ = [
    "EHCControlV2",
    "EHCContentV2",
    "EHCInputV2",
    "EHCModelV2",
    "EHCOutputV2",
    "EHCStateV2",
    "ModelSettingsV2",
]
