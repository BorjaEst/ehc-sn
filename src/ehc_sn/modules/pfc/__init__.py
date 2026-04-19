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
from ehc_sn.modules.pfc.workspace import (
    FixedSlot,
    SlotFamily,
    Workspace,
    WorkspaceLayout,
    WorkspaceSchema,
)
from ehc_sn.modules.transformer import TransformerBlockConfig
from ehc_sn.utils import trunc_normal_init_

# PFC-internal controller slot name.  Position 0 of z_H is the CLS token; it is
# exposed in the public workspace under this name.
_CONTROLLER = "controller"


# =============================================================================
# Helpers
# =============================================================================


def _full_layout_from_body(body_layout: WorkspaceLayout) -> WorkspaceLayout:
    """Prepend the internal controller fixed slot to a body workspace layout.

    The CLS token held by :class:`PFCModel` always occupies position 0 of z_H.
    This helper wraps the caller-owned body schema with an explicit
    ``controller`` fixed slot at that position.

    Args:
        body_layout: Caller-owned layout for body slots (no controller).

    Returns:
        Full layout with ``controller`` at offset 0 and all body slots shifted.
    """
    full_schema = WorkspaceSchema(
        fixed=(FixedSlot(_CONTROLLER), *body_layout.schema.fixed),
        families=body_layout.schema.families,
    )
    return WorkspaceLayout.from_schema(full_schema)


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

    layers_h: int = Field(default=4, ge=1, description="Layers in the high-level reasoning module.")
    cycles_h: int = Field(default=2, ge=1, description="Cycles in the high-level reasoning module.")

    @property
    def reasoning_h(self) -> ReasoningSettings:
        """High-level reasoning config."""
        return ReasoningSettings(cortex=self.cortex, n_layers=self.layers_h, n_cycles=self.cycles_h)

    layers_l: int = Field(default=4, ge=1, description="Layers in the low-level reasoning module.")
    cycles_l: int = Field(default=2, ge=1, description="Cycles in the low-level reasoning module.")

    @property
    def reasoning_l(self) -> ReasoningSettings:
        """Low-level reasoning config."""
        return ReasoningSettings(cortex=self.cortex, n_layers=self.layers_l, n_cycles=self.cycles_l)


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
            workspace=Workspace(layout=self.workspace.layout, tokens=self.workspace.tokens.detach()),
            scratch=self.scratch.detach(),
        )


# =============================================================================
@dataclass(frozen=True)
class PFCOutput:
    """Structured post-reasoning PFC output.

    ``state`` is the post-reasoning state for the current step.
    Callers that want a bounded recurrent carry should explicitly store
    ``state.detach()``.
    """

    state: PFCState
    q_values: Tensor

    @property
    def tokens(self) -> Tensor:
        """Full token view of the post-reasoning state (shape ``(B, S+1, D)``)."""
        return self.state.tokens

    @property
    def workspace(self) -> Workspace:
        """Full post-reasoning workspace (controller at slot 0 + body)."""
        return self.state.workspace

    @property
    def summary(self) -> Tensor:
        """Controller summary token (compatibility alias for ``workspace.slot('controller')``)."""
        return self.state.summary


# =============================================================================
class PFCModel(nn.Module):
    """Prefrontal Cortex (PFC) reasoning module.

    Implements a two-level recurrent reasoning process (high-level / low-level)
    with an auxiliary value estimator (vmPFC analogue).

    The PFC-internal CLS token is always mapped to the ``controller`` fixed slot
    at position 0 of the output workspace.  All other slots are caller-declared
    body slots.
    """

    def __init__(  # --------------------------------------------------------------------
        self, config: PFCSettings, device: Optional[Device] = None, dtype: Optional[Dtype] = None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config

        self.high_level = HighLvRModule(config.reasoning_h, device=device, dtype=dtype)
        self.low_level = LowLvRModule(config.reasoning_l, device=device, dtype=dtype)
        self.estimator = QValueEstimator(config.value_head, device=device, dtype=dtype)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size, device=device, dtype=dtype))
        # Default body layout for callers that pass raw tensors via step_tokens().
        self._default_body_layout = WorkspaceLayout.from_schema(
            WorkspaceSchema(fixed=(), families=(SlotFamily("body", config.seq_length),))
        )
        self.optimizer = None  # Placeholder for future dACC reward-based updates
        self.reset_parameters()

    @property
    def config(self) -> PFCSettings:
        """PFC module settings."""
        return self._config

    def reset_parameters(self) -> None:
        """Initialize parameters and buffers."""
        trunc_normal_init_(self.high_level.reset_vector, std=1)
        trunc_normal_init_(self.low_level.reset_vector, std=1)
        self.cls_token.data.zero_()

    def _build_state_from_memory(
        self, memory: WorkingMemory, full_layout: WorkspaceLayout, *, detach: bool
    ) -> PFCState:
        """Bind a WorkingMemory directly to a full workspace layout."""
        tokens = memory.z_H.detach() if detach else memory.z_H
        carry = memory.detach() if detach else memory
        return PFCState(
            workspace=Workspace(layout=full_layout, tokens=tokens),
            scratch=PFCScratchState(memory=carry),
        )

    def init_state(  # --------------------------------------------------------------------------
        self, batch_size: int, body_layout: Optional[WorkspaceLayout] = None,
    ) -> PFCState:  # fmt: skip
        """Create a fresh PFC recurrent state.

        Args:
            batch_size: Number of parallel sequences.
            body_layout: Optional body workspace layout.  When omitted, the default
                ``body`` family layout is used.

        Returns:
            Initialized :class:`PFCState`.
        """
        layout = body_layout if body_layout is not None else self._default_body_layout
        if layout.size != self.config.seq_length:
            raise ValueError(
                f"body_layout size must match seq_length {self.config.seq_length}, got {layout.size}."
            )
        full_layout = _full_layout_from_body(layout)
        memory = r.init_memory(batch_size, full_layout.size, self.high_level, self.low_level)
        return self._build_state_from_memory(memory, full_layout, detach=False)

    def reset_state(  # -------------------------------------------------------------------------
        self, state: PFCState, reset_flag: Tensor,
    ) -> PFCState:  # fmt: skip
        """Selectively reset rows of the PFC state."""
        memory = r.reset_memory(state.scratch.memory, reset_flag, self.high_level, self.low_level)
        return self._build_state_from_memory(memory, state.workspace.layout, detach=False)

    def _run_reasoning(  # ----------------------------------------------------------------------
        self, x: Tensor, state: PFCState, prefix_bias: Optional[Tensor] = None,
    ) -> tuple[WorkingMemory, Tensor]:  # fmt: skip
        """Run the internal tensor-first reasoning core."""
        total_steps = self.config.reasoning_h.n_cycles * (self.config.reasoning_l.n_cycles + 1)

        # Prepend CLS (controller) to body tokens: (B, S, D) → (B, S+1, D).
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if prefix_bias is not None:
            cls_token = cls_token + prefix_bias.unsqueeze(1).to(dtype=cls_token.dtype)
        x = torch.cat([cls_token, x], dim=1)

        memory_gen = r.reasoning_gen(x, state.scratch.memory, self.high_level, self.low_level)
        with torch.no_grad():
            for _ in range(total_steps - 2):
                memory = next(memory_gen)

        memory = next(memory_gen)  # N-2 step: low-level with gradients
        memory = next(memory_gen)  # N-1 step: high-level with gradients
        q_estimation = self.estimator(memory.z_H, memory.z_L)
        return memory, q_estimation

    def step(  # --------------------------------------------------------------------------------
        self, body: Workspace, state: Optional[PFCState] = None, prefix_bias: Optional[Tensor] = None,
    ) -> PFCOutput:  # fmt: skip
        """Run one PFC step over a body workspace.

        The body workspace contains caller-declared schema slots only (no controller).
        The returned :class:`PFCState` exposes a full workspace that includes the
        ``controller`` fixed slot at position 0.

        Args:
            body: Schema-body workspace of shape ``(B, seq_length, D)``.
            state: Optional prior state.  If ``None``, a fresh state is initialised.
            prefix_bias: Optional additive bias applied to the internal CLS token.

        Returns:
            :class:`PFCOutput` with the updated full workspace and Q-value estimates.
        """
        if body.layout.size != self.config.seq_length:
            raise ValueError(
                f"body workspace size must match seq_length {self.config.seq_length}, "
                f"got {body.layout.size}."
            )
        full_layout = _full_layout_from_body(body.layout)
        state = state or self.init_state(int(body.tokens.shape[0]), body_layout=body.layout)
        memory, q_values = self._run_reasoning(body.tokens, state, prefix_bias=prefix_bias)
        return PFCOutput(state=self._build_state_from_memory(memory, full_layout, detach=False), q_values=q_values)

    def step_tokens(  # -------------------------------------------------------------------------
        self,
        x: Tensor,
        state: Optional[PFCState] = None,
        body_layout: Optional[WorkspaceLayout] = None,
        prefix_bias: Optional[Tensor] = None,
    ) -> PFCOutput:  # fmt: skip
        """Compatibility surface for tensor callers that pass raw body slot tensors."""
        if x.ndim != 3:
            raise ValueError(f"PFC token inputs must have shape (B, S, D), got {tuple(x.shape)}.")
        layout = body_layout if body_layout is not None else self._default_body_layout
        return self.step(Workspace(layout=layout, tokens=x), state=state, prefix_bias=prefix_bias)

    def forward(  # -----------------------------------------------------------------------------
        self, x: Tensor, state: Optional[PFCState] = None, prefix_bias: Optional[Tensor] = None,
    ) -> tuple[PFCState, Tensor, Tensor]:  # fmt: skip
        """Compatibility wrapper that returns (new_state, z_H, q_estimation).

        Args:
            x: Body token inputs of shape ``(B, S, D)``.
            state: Optional prior state.
            prefix_bias: Optional additive bias for the internal CLS token.

        Returns:
            ``(new_state_detached, z_H_tokens, q_estimation)``.
        """
        output = self.step_tokens(x, state=state, prefix_bias=prefix_bias)
        return output.state.detach(), output.tokens, output.q_values


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
