"""EHP v1 backbone model, settings, state, and forward I/O.

EHP v1 (Entorhinal-Hippocampal Circuit, version 2) extends the base TEM
circuit with a Prefrontal Cortex (PFC) reasoning module and a Striatum (STR)
reward-prediction head.

Cue-timing contract (V1):
    The cortical cue that biases replay retrieval is derived from the
    *previous*-step public PFC summary (``state.pfc.summary``).  This is the
    defining V1 property; V3 will use the current-step PFC output instead.
    This distinction must not be hidden in shared helpers.

Bank-c semantics:
    HPCAttention bank ``c`` is a place-keyed contextual auxiliary bank that
    shares the hippocampal write key with the default bank.  Three variables
    make the routing explicit:

        - ``c_prop``: cortical cue proposal — previous-step PFC summary projected
            into the hippocampal multi-frequency cue family via ``pfc_to_hpc``.
    - ``c_mem``:  reinstated contextual evidence from bank ``c`` (deferred in
      V1; set to ``None``).
    - ``c_use``:  the routed cue used for replay bias and bank-c writes.
      In V1: ``c_use = c_prop``.

PFC body workspace layout (size = ``pfc.seq_length``):
    Fixed slot 0 - ``state``   : projected current grounded-place (p_post).
    Fixed slot 1 - ``replay``  : projected contextual replay code.
    Fixed slot 2 - ``cue``     : projected cortical cue (c_use).
    Family ``content`` (size = pfc.seq_length - 3): previous-step recurrent
    cortical substrate from ``state.pfc.workspace.family("content")``.

For the reference config (pfc.seq_length = 36) the content family has 33 slots.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, cast

import torch
from pydantic import BaseModel, Field, model_validator
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.models.ehp.core.ehp_base import (
    FAMILY_CONTENT,
    SLOT_CUE,
    SLOT_REPLAY,
    SLOT_STATE,
    EHCProjectionSettings,
)
from ehc_sn.models.tem.core.tem_base import GridCodes, PlaceCodes, PredCodes
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
class ModelSettingsV1(BaseModel, extra="forbid", strict=False):
    """Canonical EHP v1 model settings.

    All architectural dimensions are resolved from this config; no magic
    numbers appear in ``EHCModelV1``.
    """

    @classmethod
    def from_config(cls, path: str | Path) -> "ModelSettingsV1":
        config_map = tomllib.load(Path(path).open("rb"))
        return cls.model_validate(config_map)

    transition_action_count: int = Field(
        ...,
        ge=1,
        description="Number of discrete transition actions for the MEC path-integration surface.",
    )
    f_initial: list[float] = Field(
        default_factory=lambda: [0.99, 0.3, 0.09, 0.5, 0.4],
        min_length=1,
        description="Shared multiscale frequency ordering across LEC, MEC, and HPC.",
    )

    @property
    def internal_action_count(self) -> int:
        return int(self.pfc.value_head.n_actions)

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
    def _validate_pfc_seq_length(self) -> "ModelSettingsV1":
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
@dataclass
class EHCControlV1:
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
class EHCContentV1:
    """Content-pathway outputs from the PFC body workspace.

    These tensors are taken from the post-reasoning PFC workspace and reflect
    the working memory state after the current-step reasoning pass.

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
class EHCOutputV1(DetachMixin):
    """Task-agnostic output payload for one EHP v1 forward step.

    Attributes:
        control:     PFC/STR control-pathway outputs.
        content:     PFC body workspace content outputs.
        grid_codes:  Named MEC grid codes (prior and posterior).
        place_codes: Named HPC place codes (posterior, prior, retrieved, sensory).
        pred_codes:  LEC-space projections of HPC place codes for decoding.
    """

    control: EHCControlV1
    content: EHCContentV1
    grid_codes: GridCodes
    place_codes: PlaceCodes
    pred_codes: PredCodes


# =============================================================================
@dataclass
class EHCInputV1(DetachMixin):
    """Task-agnostic input payload for EHP v1 forward steps.

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
class EHCStateV1(DetachMixin):
    """Container for the full recurrent state across all EHP region modules.

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
class EHCModelV1(nn.Module):
    """EHP v1 backbone: TEM circuit with PFC reasoning and STR control.

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
        config: ModelSettingsV1,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        """Construct EHP v1 from resolved model settings."""
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
        _hpc_ws_from = workspace_endpoint(
            hpc_flat, fixed=[SLOT_STATE, SLOT_REPLAY, SLOT_CUE]
        )
        _hpc_ws_to = workspace_endpoint(
            hidden_size, fixed=[SLOT_STATE, SLOT_REPLAY, SLOT_CUE]
        )
        self.projections = ProjectionBundle.from_modules(
            # LEC/MEC expose aligned multiscale codes; PFC exposes a flat public summary;
            lec_to_hpc=(self.lec, self.hpc, config.projections.lec_to_hpc),
            mec_to_hpc=(self.mec, self.hpc, config.projections.mec_to_hpc),
            # PFC->HPC parameter block per fixed role (state, replay, cue).
            pfc_to_hpc=(
                flat_endpoint(hidden_size),
                self.hpc,
                config.projections.pfc_to_hpc,
            ),
            # HPC->PFC reverse edge uses a workspace-aligned edge with one independent
            hpc_to_pfc=(
                _hpc_ws_from,
                _hpc_ws_to,
                config.projections.hpc_to_pfc,
            ),
        )

        self.reset_parameters()

    @property
    def config(self) -> ModelSettingsV1:
        """Return the parsed EHP v1 model settings."""
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
        """Reset all projection parameters owned directly by EHP."""
        self.projections.reset_parameters()

    def init_state(  # -----------------------------------------------------------
        self,
        batch_size: int,
        *,
        memory: Optional[MemoryState] = None,
        device: Optional[Device] = None,
    ) -> EHCStateV1:
        """Create an initial full-batch recurrent EHP v1 state.

        Args:
            batch_size: Number of parallel sequences.
            memory: Optional pre-built HPC memory state.  When ``None`` a fresh
                empty memory is allocated on ``device``.
            device: Target device for all fresh state tensors.

        Returns:
            Freshly initialized :class:`EHCStateV1`.
        """
        if device is None:
            device = next(self.parameters()).device
        memory = (
            memory
            if memory is not None
            else self.hpc.init_memory(batch_size=batch_size, device=device)
        )
        return EHCStateV1(
            # NOTE: PFCModel.init_state does NOT accept a device kwarg.
            pfc=self.pfc.init_state(
                batch_size, body_schema=self.config.body_schema, device=device
            ),
            str=self.str.init_state(batch_size, device=device),
            lec=self.lec.init_state(batch_size, device=device),
            mec=self.mec.init_state(batch_size, device=device),
            hpc=self.hpc.init_state(batch_size, device=device, memory=memory),
        )

    def reset_state(  # ----------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: EHCStateV1,
    ) -> EHCStateV1:
        """Reset flagged batch rows to a fresh episode state.

        Args:
            reset_flag: Boolean tensor of shape ``(B,)``.
            state: Current recurrent state.

        Returns:
            Updated :class:`EHCStateV1` with flagged rows reset to initial values.
        """
        device = state.hpc.cells[0].device
        reset_flag = reset_flag.to(device=device, dtype=torch.bool).view(-1)
        if not torch.any(reset_flag):
            return state

        return EHCStateV1(
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
        inputs: EHCInputV1,
        state: Optional[EHCStateV1] = None,
    ) -> tuple[EHCOutputV1, EHCStateV1]:
        """Run one EHP v1 step and return architecture-native latents.

        Cue-timing (V1 contract):
            ``c_prop`` is derived from *previous*-step ``state.pfc.summary``.
            ``c_mem = None`` (contextual reinstatement from bank c deferred).
            ``c_use = c_prop``.

        All HPC recall calls are placed before the generative/inference updates
        so they read the prior memory state (same ordering as TEM v1).

        Args:
            inputs: Task-agnostic EHP v1 input payload.
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

        # --- V1 cue: previous-step PFC summary (before this step's PFC run) --
        # c_prop: project previous-step PFC summary into the hippocampal multi-frequency cue family.
        # c_mem:  reinstated contextual evidence from bank c — deferred in V1.
        # c_use:  routed cue for replay bias and bank-c writes; equals c_prop in V1.
        #
        # Arena stop-gradient seam: detach the PFC summary so arena loss cannot update PFC
        # or propagate back through this edge. pfc_to_hpc is not bypassed so it remains
        # trainable from arena loss when its optimizer exclusion is lifted in a future phase.
        prev_pfc_summary: Tensor = state.pfc.summary.detach()  # (B, D)
        c_prop: list[Tensor] = cast(
            list[Tensor], self.pfc_to_hpc(prev_pfc_summary)
        )
        c_mem: Optional[list[Tensor]] = (
            None  # deferred: bank-c reinstatement not implemented
        )
        c_use: list[Tensor] = c_prop  # V1: use cortical cue proposal directly

        # 1. Path integration: grid prior -------------------------------------
        g_prior, state.mec = self.mec.generative(
            previous_action, episode_start, landmark_id, state=state.mec
        )
        g_query_prior = self.mec_to_hpc(g_prior)

        # 2. LEC sensory encoding ---------------------------------------------
        x_, state.lec = self.lec.inference(observation_embedding, state.lec)
        x_query = self.lec_to_hpc(x_)

        # 3. Sensory-cued recall (read-only on state.hpc) ---------------------
        p_sensory_recall = self.hpc.recall(
            read_cues=ReadCues(families={"x": x_query}),
            state=state.hpc,
            role="inference",
            read=CueRead(kind="cue", cue="x"),
        )

        # 4. MEC correction: grid posterior from sensory recall ---------------
        g_post, state.mec = self.mec.inference(
            p_sensory_recall, landmark_id, state=state.mec
        )
        g_query_post = self.mec_to_hpc(g_post)

        # 5. Grid-cued ancestral recall (prior, read-only) --------------------
        p_grid_prior_recall = self.hpc.recall(
            read_cues=ReadCues(families={"g": g_query_prior}),
            state=state.hpc,
            role="generative",
            read=CueRead(kind="cue", cue="g"),
        )

        # 6. Grid-cued retrieved recall (posterior, read-only) ----------------
        p_grid_post_recall = self.hpc.recall(
            read_cues=ReadCues(families={"g": g_query_post}),
            state=state.hpc,
            role="generative",
            read=CueRead(kind="cue", cue="g"),
        )

        # 7. Contextual replay retrieval (read-only) --------------------------
        # Use both grid (g) and cortical cue (c_use) as source cues; retrieve
        # from the x-bank (sensory/inference memory target).
        # c_use carries previous-step cortical context (V1 cue-timing contract).
        p_replay_read = self.hpc.recall(
            read_cues=ReadCues(families={"g": g_query_post, "c": c_use}),
            state=state.hpc,
            role="generative",
            read=TargetRead(kind="target", sources=("g", "c"), target="x"),
        )

        # 8. Form place beliefs and update HPC grounded-belief state ----------
        p_path, state.hpc = self.hpc.generative(
            p_grid_prior_recall, state=state.hpc
        )
        p_recall, state.hpc = self.hpc.generative(
            p_grid_post_recall, state=state.hpc
        )
        p_post, state.hpc = self.hpc.inference(
            x_query, g_query_post, state=state.hpc
        )

        # 9. Build PFC body workspace tokens ----------------------------------
        # Stack HPC flat codes into a 3-slot workspace-aligned tensor (state/replay/cue),
        # project once through hpc_to_pfc (independent parameter block per role), then
        # unpack the projected tokens for the body assembly below.
        p_post_flat: Tensor = torch.cat(p_post, dim=-1)  # (B, hpc_flat)
        p_replay_flat: Tensor = torch.cat(
            p_replay_read, dim=-1
        )  # (B, hpc_flat)
        c_use_flat: Tensor = torch.cat(c_use, dim=-1)  # (B, hpc_flat)

        hpc_slots_in: Tensor = torch.stack(
            [p_post_flat, p_replay_flat, c_use_flat], dim=1
        )  # (B, 3, hpc_flat)
        hpc_slots_out: Tensor = self.hpc_to_pfc(
            hpc_slots_in
        )  # (B, 3, hidden_size)

        state_token: Tensor = hpc_slots_out[:, 0:1, :]  # (B, 1, D)
        replay_token: Tensor = hpc_slots_out[:, 1:2, :]  # (B, 1, D)
        cue_token: Tensor = hpc_slots_out[:, 2:3, :]  # (B, 1, D)

        # Content family: previous-step cortical substrate from the public workspace.
        prev_content: Tensor = state.pfc.workspace.family(
            FAMILY_CONTENT
        )  # (B, content_size, D)

        # Assemble body in slot order (state, replay, cue, content).
        # Total tokens = 3 fixed + content_size = pfc.seq_length.
        body_tokens: Tensor = torch.cat(
            [state_token, replay_token, cue_token, prev_content], dim=1
        )

        # 10. PFC reasoning step ----------------------------------------------
        # PFCModel.step validates workspace.layout.size == pfc.seq_length and
        # that the derived full layout matches state.pfc.workspace.layout.
        body_layout = WorkspaceLayout.from_schema(self.config.body_schema)
        body_workspace = body_layout.bind(body_tokens)
        pfc_out, state.pfc = self.pfc.step(body_workspace, state=state.pfc)
        control_logits: Tensor = pfc_out.q_values

        # 11. STR reward prediction -------------------------------------------
        # STR receives the detached PFC summary (theta_summary) and Q-value logits.
        # Detaching prevents value gradients from propagating into PFC reasoning.
        theta_summary: Tensor = state.pfc.summary.detach()  # (B, D)
        state.str, reward_prediction = self.str(
            theta_summary, control_logits, state.str
        )

        # 12. HPC memory write ------------------------------------------------
        # generative = p_recall (corrected-grid replay value).
        # inference  = p_sensory_recall (sensory-cued recall value).
        # named_writes["c"] = c_use (contextual cue written into bank c).
        write_payload = WritePayload(
            generative=p_recall,
            inference=p_sensory_recall,
            named_writes={"c": c_use},
        )
        state.hpc = self.hpc.update(p_post, write_payload, state=state.hpc)

        # 13. Package and return outputs --------------------------------------
        grid_codes = GridCodes(prior=g_prior, post=g_post)
        place_codes = PlaceCodes(
            post=p_post,
            path=p_path,
            recall=p_recall,
            sensory=p_sensory_recall,
        )
        x_post = self.projections.lec_to_hpc.inverse(p_post)
        x_path = self.projections.lec_to_hpc.inverse(p_path)
        x_recall = self.projections.lec_to_hpc.inverse(p_recall)
        pred_codes = PredCodes(post=x_post, path=x_path, recall=x_recall)

        # EHCContentV1 exposes post-reasoning PFC workspace slots.
        content = EHCContentV1(
            state_slot=state.pfc.workspace.slot(SLOT_STATE),  # (B, D)
            replay_slot=state.pfc.workspace.slot(SLOT_REPLAY),  # (B, D)
            cue_slot=state.pfc.workspace.slot(SLOT_CUE),  # (B, D)
            content_slots=state.pfc.workspace.family(
                FAMILY_CONTENT
            ),  # (B, content_size, D)
        )
        control = EHCControlV1(
            theta_summary=state.pfc.summary,  # (B, D) — gradient-attached summary
            control_logits=control_logits,  # (B, n_actions)
            reward_prediction=reward_prediction,  # (B,)
        )
        return (
            EHCOutputV1(
                control=control,
                content=content,
                grid_codes=grid_codes,
                place_codes=place_codes,
                pred_codes=pred_codes,
            ),
            state,
        )


# =============================================================================
__all__ = [
    "EHCControlV1",
    "EHCContentV1",
    "EHCInputV1",
    "EHCModelV1",
    "EHCOutputV1",
    "EHCStateV1",
    "ModelSettingsV1",
]
