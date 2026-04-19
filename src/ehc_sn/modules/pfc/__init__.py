"""PFC module public surface: PFCSettings, PFCState, PFCModel, and workspace types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.modules.pfc import reasoning as r
from ehc_sn.modules.pfc.reasoning import HighLvRModule, LowLvRModule, ReasoningSettings, WorkingMemory
from ehc_sn.modules.pfc.values import QEstimatorSettings, QValueEstimator
from ehc_sn.modules.pfc.workspace import FixedSlot, SlotFamily, Workspace, WorkspaceLayout, WorkspaceSchema
from ehc_sn.modules.transformer import TransformerBlockConfig
from ehc_sn.utils import trunc_normal_init_

# PFC-internal controller slot name.  Position 0 of z_H is the CLS token; it is
# exposed in the public workspace under this name.
_CONTROLLER = "controller"


# =============================================================================
class PFCSettings(BaseModel, extra="forbid"):
    """Configuration for :class:`PFCModel`.

    The PFC module is a two-timescale recurrent reasoning system (high/low level)
    with an auxiliary value head (vmPFC analogue).

    Attributes:
        seq_length: Number of body (schema) tokens per step (controller not counted).
        value_head: Settings for the auxiliary Q/value estimator.
        cortex: Base transformer block configuration reused across reasoning modules.
        layers_h/cycles_h: Depth and recurrence cycles for the high-level module.
        layers_l/cycles_l: Depth and recurrence cycles for the low-level module.
    """

    seq_length: int = Field(
        ...,
        ge=1,
        description="Number of body (schema) tokens per step (controller not counted).",
    )
    value_head: QEstimatorSettings = Field(
        ...,
        description="Configuration for the Q-value estimator (vmPFC analogue).",
    )
    cortex: TransformerBlockConfig = Field(
        ...,
        description="Base transformer block config for reasoning modules.",
    )

    @property
    def hidden_size(self) -> int:
        """Return the hidden size from the cortex config."""
        return self.cortex.embedding_dim

    layers_h: int = Field(
        default=4,
        ge=1,
        description="Layers in the high-level reasoning module.",
    )
    cycles_h: int = Field(
        default=2,
        ge=1,
        description="Cycles in the high-level reasoning module.",
    )

    @property
    def reasoning_h(self) -> ReasoningSettings:
        """High-level reasoning config."""
        return ReasoningSettings(
            cortex=self.cortex,
            n_layers=self.layers_h,
            n_cycles=self.cycles_h,
        )

    layers_l: int = Field(
        default=4,
        ge=1,
        description="Layers in the low-level reasoning module.",
    )
    cycles_l: int = Field(
        default=2,
        ge=1,
        description="Cycles in the low-level reasoning module.",
    )

    @property
    def reasoning_l(self) -> ReasoningSettings:
        """Low-level reasoning config."""
        return ReasoningSettings(
            cortex=self.cortex,
            n_layers=self.layers_l,
            n_cycles=self.cycles_l,
        )


# =============================================================================
@dataclass(frozen=True)
class PFCScratchState:
    """Internal tensor-first carry used by the recurrent PFC updater."""

    memory: WorkingMemory

    def detach(self) -> "PFCScratchState":
        """Return a detached copy."""
        return PFCScratchState(memory=self.memory.detach())


# =============================================================================
@dataclass(frozen=True)
class PFCState:
    """Canonical public PFC state.

    The full z_H tensor is exposed as a named :class:`Workspace` that includes
    the controller slot at position 0 and all caller-declared body slots after it.
    Internal recurrent carry (z_H and z_L) remains accessible only through
    ``scratch`` for continuation.

    Attributes:
        workspace: Full public workspace view over z_H (controller + body).
        scratch: Internal recurrent carry; detach before storing as episode state.
    """

    workspace: Workspace
    scratch: PFCScratchState

    def __post_init__(self) -> None:
        B, S, D = self.workspace.tokens.shape
        z_H, z_L = self.scratch.memory.z_H, self.scratch.memory.z_L
        if tuple(z_H.shape) != (B, S, D):
            raise ValueError(f"PFC scratch z_H must have shape {(B, S, D)}, got {tuple(z_H.shape)}.")
        if tuple(z_L.shape) != (B, S, D):
            raise ValueError(f"PFC scratch z_L must have shape {(B, S, D)}, got {tuple(z_L.shape)}.")

    @property
    def summary(self) -> Tensor:
        """Controller summary token (compatibility alias for ``workspace.slot('controller')``)."""
        return self.workspace.slot(_CONTROLLER)

    @property
    def tokens(self) -> Tensor:
        """Full token view of the current state (shape ``(B, S+1, D)``)."""
        return self.workspace.tokens

    def detach(self) -> "PFCState":
        """Return a detached copy of the public state."""
        return PFCState(
            workspace=self.workspace.layout.bind(self.workspace.tokens.detach()),
            scratch=self.scratch.detach(),
        )


# =============================================================================
@dataclass(frozen=True)
class PFCOutput:
    """Output of one :class:`PFCModel` step.

    Attributes:
        workspace: Full public workspace view over z_H (controller + body).
        q_values: Auxiliary Q-value estimates from the value head. Shape ``(B, n_actions)``.
    """

    workspace: Workspace
    q_values: Tensor

    @property
    def summary(self) -> Tensor:
        """Controller summary token (alias for ``workspace.slot('controller')``)."""
        return self.workspace.slot(_CONTROLLER)

    @property
    def tokens(self) -> Tensor:
        """Full token tensor from the workspace. Shape ``(B, S+1, D)``."""
        return self.workspace.tokens


# =============================================================================
class PFCModel(nn.Module):
    """Prefrontal Cortex (PFC) reasoning module.

    Implements a two-level recurrent reasoning process (high-level / low-level)
    with an auxiliary value estimator (vmPFC analogue).

    The PFC-internal CLS token is always mapped to the ``controller`` fixed slot
    at position 0 of the output workspace.  All other slots are caller-declared
    body slots.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: PFCSettings,
        device: Optional[Device] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self._config = config

        self.high_level = HighLvRModule(config.reasoning_h, device=device, dtype=dtype)
        self.low_level = LowLvRModule(config.reasoning_l, device=device, dtype=dtype)
        self.estimator = QValueEstimator(config.value_head, device=device, dtype=dtype)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size, device=device, dtype=dtype))

        # Default schema layout (body-only) used by step_tokens() and forward().
        self._default_schema_layout = WorkspaceLayout.from_schema(WorkspaceSchema(fixed=(), families=(SlotFamily("body", config.seq_length),)))  # fmt: skip
        # Default full workspace layout used when callers omit workspace_layout in init_state().
        self._default_workspace_layout = _body_to_full_layout(self._default_schema_layout)

        self.optimizer = None  # Placeholder for future dACC reward-based updates
        self.reset_parameters()

    @property
    def config(self) -> PFCSettings:
        """PFC module settings."""
        return self._config

    def reset_parameters(  # --------------------------------------------------
        self,
    ) -> None:
        """Initialize parameters and buffers."""
        trunc_normal_init_(self.high_level.reset_vector, std=1)
        trunc_normal_init_(self.low_level.reset_vector, std=1)
        self.cls_token.data.zero_()

    def init_state(  # --------------------------------------------------------
        self,
        batch_size: int,
        workspace_layout: Optional[WorkspaceLayout] = None,
    ) -> PFCState:
        """Create a fresh PFC recurrent state.

        Args:
            batch_size: Number of parallel sequences.
            workspace_layout: Optional *full* workspace layout that must include the
                ``controller`` fixed slot at position 0.  When omitted, the default
                layout with a ``body`` family is used.

        Returns:
            Initialized :class:`PFCState`.
        """
        layout = workspace_layout if workspace_layout is not None else self._default_workspace_layout
        self._validate_workspace_layout(layout)
        memory = r.init_memory(batch_size, layout.size, self.high_level, self.low_level)
        return _build_state_from_memory(memory, layout, detach=False)

    def _validate_workspace_layout(  # ----------------------------------------
        self,
        layout: WorkspaceLayout,
    ) -> None:
        """Validate that *layout* is compatible with this PFC configuration."""
        expected_size = self.config.seq_length + 1
        if layout.size != expected_size:
            raise ValueError(f"workspace_layout size must be seq_length + 1 = {expected_size}, got {layout.size}.")
        try:
            ctrl_pos = layout.slot(_CONTROLLER)
        except KeyError:
            raise ValueError(f"workspace_layout must declare a '{_CONTROLLER}' fixed slot at position 0.") from None
        if ctrl_pos != 0:
            raise ValueError(f"workspace_layout '{_CONTROLLER}' fixed slot must be at position 0, got {ctrl_pos}.")

    def reset_state(  # -------------------------------------------------------
        self,
        state: PFCState,
        reset_flag: Tensor,
    ) -> PFCState:
        """Selectively reset rows of the PFC state."""
        memory = r.reset_memory(state.scratch.memory, reset_flag, self.high_level, self.low_level)
        return _build_state_from_memory(memory, state.workspace.layout, detach=False)

    def step(  # --------------------------------------------------------------
        self,
        workspace: Workspace,
        state: Optional[PFCState] = None,
        prefix_bias: Optional[Tensor] = None,
    ) -> tuple[PFCOutput, PFCState]:
        """Run one PFC step over a body workspace.

        The ``workspace`` carries body/schema tokens only — the ``controller`` slot is
        PFC-internal and is prepended during reasoning.  The returned :class:`PFCOutput`
        exposes the full public workspace (controller at position 0 plus all
        caller-declared body slots).

        Build the input workspace with ``schema_layout.bind(tokens)``.

        Args:
            workspace: Body workspace of shape ``(B, seq_length, D)``.
            state: Optional prior recurrent state.  Its full workspace layout must be
                consistent with the layout derived from ``workspace``; a mismatch
                raises :class:`ValueError`.
            prefix_bias: Optional additive bias applied to the internal CLS token.

        Returns:
            ``(output, next_state)`` — output first, state second (canonical backbone
            seam ordering).  ``output.workspace`` and ``next_state.workspace`` expose
            the same public z_H surface; ``output.q_values`` carries auxiliary Q logits.
        """
        if workspace.layout.size != self.config.seq_length:
            raise ValueError(f"workspace size must equal seq_length {self.config.seq_length}, " f"got {workspace.layout.size}.")
        full_layout = _body_to_full_layout(workspace.layout)
        if state is not None and state.workspace.layout != full_layout:
            raise ValueError(
                "workspace layout does not match the layout of the provided prior state. "
                "Ensure init_state and step are called with consistent layouts."
            )

        if state is None:
            state = self.init_state(int(workspace.tokens.shape[0]), workspace_layout=full_layout)

        # Prepend CLS (controller) to body tokens: (B, seq_length, D) → (B, seq_length+1, D).
        batch_size = int(workspace.tokens.shape[0])
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        if prefix_bias is not None:
            cls_token = cls_token + prefix_bias.unsqueeze(1).to(dtype=cls_token.dtype)
        x = torch.cat([cls_token, workspace.tokens], dim=1)

        # Run the internal tensor-first reasoning core.  The last two steps run
        # with gradient tracking for value estimation; all prior steps are detached.
        total_steps = self.config.reasoning_h.n_cycles * (self.config.reasoning_l.n_cycles + 1)
        memory_gen = r.reasoning_gen(x, state.scratch.memory, self.high_level, self.low_level)
        with torch.no_grad():
            for _ in range(total_steps - 2):
                memory = next(memory_gen)
        memory = next(memory_gen)  # step N-2: with gradients
        memory = next(memory_gen)  # step N-1: with gradients
        q_values = self.estimator(memory.z_H, memory.z_L)

        new_state = _build_state_from_memory(memory, full_layout, detach=False)
        output = PFCOutput(workspace=new_state.workspace, q_values=q_values)
        return output, new_state

    def forward(  # -----------------------------------------------------------
        self,
        x: Tensor,
        state: Optional[PFCState] = None,
        schema_layout: Optional[WorkspaceLayout] = None,
        prefix_bias: Optional[Tensor] = None,
    ) -> tuple[PFCOutput, PFCState]:
        """Tensor path — prefer :meth:`step` with ``layout.bind(tokens)``.

        Args:
            x: Body token inputs of shape ``(B, seq_length, D)``.
            state: Optional prior recurrent state.
            schema_layout: Body-only layout.  When ``None`` the default ``body`` family
                layout is used.
            prefix_bias: Optional additive bias for the internal CLS token.

        Returns:
            ``(output, next_state)`` — output first, state second.
        """
        if x.ndim != 3:
            raise ValueError(f"PFC token inputs must have shape (B, S, D), got {tuple(x.shape)}.")
        layout = schema_layout if schema_layout is not None else self._default_schema_layout
        return self.step(layout.bind(x), state=state, prefix_bias=prefix_bias)


# =============================================================================
def _build_state_from_memory(  # ----------------------------------------------
    memory: WorkingMemory,
    full_layout: WorkspaceLayout,
    *,
    detach: bool,
) -> PFCState:
    """Bind a WorkingMemory to a full workspace layout via :meth:`WorkspaceLayout.bind`."""
    tokens = memory.z_H.detach() if detach else memory.z_H
    carry = memory.detach() if detach else memory
    return PFCState(
        workspace=full_layout.bind(tokens),
        scratch=PFCScratchState(memory=carry),
    )


# =============================================================================
def _body_to_full_layout(  # --------------------------------------------------
    body_layout: WorkspaceLayout,
) -> WorkspaceLayout:
    """Prepend the internal controller fixed slot to a caller-owned body layout."""
    full_schema = WorkspaceSchema(
        fixed=(FixedSlot(_CONTROLLER), *body_layout.schema.fixed),
        families=body_layout.schema.families,
    )
    return WorkspaceLayout.from_schema(full_schema)


# =============================================================================
__all__ = [
    "FixedSlot",
    "PFCModel",
    "PFCOutput",
    "PFCScratchState",
    "PFCSettings",
    "PFCState",
    "SlotFamily",
    "Workspace",
    "WorkspaceLayout",
    "WorkspaceSchema",
]
