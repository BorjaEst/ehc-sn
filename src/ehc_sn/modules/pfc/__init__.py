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
    ComposedWorkspaceWriter,
    NamedWorkspace,
    WorkspaceSlotWriter,
    WorkspaceSpec,
    workspace_from_prefixed_tokens,
)
from ehc_sn.modules.transformer import TransformerBlockConfig
from ehc_sn.utils import trunc_normal_init_


def _compose_controlled_workspace(controller_slot_name: str, summary: Tensor, workspace: NamedWorkspace) -> NamedWorkspace:
    """Build a public working-memory view with an explicit controller slot."""
    if not controller_slot_name:
        raise ValueError("controller_slot_name must be non-empty.")
    return NamedWorkspace(
        spec=WorkspaceSpec((controller_slot_name, *workspace.spec.names)),
        tokens=torch.cat([summary.unsqueeze(1), workspace.tokens], dim=1),
    )


# =================================================================================================
class PFCSettings(BaseModel, extra="forbid"):
    """Configuration for :class:`PFCModel`.

    The PFC module is a two-timescale recurrent reasoning system (high/low level)
    with an auxiliary value head (vmPFC analogue).

    Attributes:
        seq_length: Number of tokens per example (without the CLS prefix).
        value_head: Settings for the auxiliary Q/value estimator.
        cortex: Base transformer block configuration reused across reasoning modules.
        layers_h/cycles_h: Depth and recurrence cycles for the high-level module.
        layers_l/cycles_l: Depth and recurrence cycles for the low-level module.
    """

    # Model parameters for features
    seq_length: int = Field(
        ...,
        ge=1,
        description="Sequence length for the model (number of tokens per example).",
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
        """Convenience property to access the hidden size from the cortex config."""
        return self.cortex.embedding_dim

    # High-level reasoning module parameters (anterior dlPFC analogue)
    layers_h: int = Field(
        default=4,
        ge=1,
        description="Number of layers in the high-level reasoning module.",
    )
    cycles_h: int = Field(
        default=2,
        ge=1,
        description="Number of cycles to reason in the high-level reasoning module.",
    )

    @property
    def reasoning_h(self) -> ReasoningSettings:
        """Convenience property to access the high-level reasoning config."""
        return ReasoningSettings(cortex=self.cortex, n_layers=self.layers_h, n_cycles=self.cycles_h)

    # Low-level reasoning module parameters (posterior dlPFC analogue)
    layers_l: int = Field(
        default=4,
        ge=1,
        description="Number of layers in the low-level reasoning module.",
    )
    cycles_l: int = Field(
        default=2,
        ge=1,
        description="Number of cycles to reason in the low-level reasoning module.",
    )

    @property
    def reasoning_l(self) -> ReasoningSettings:
        """Convenience property to access the low-level reasoning config."""
        return ReasoningSettings(cortex=self.cortex, n_layers=self.layers_l, n_cycles=self.cycles_l)


# =================================================================================================
@dataclass(frozen=True)
class PFCScratchState:
    """Internal tensor-first carry used by the recurrent PFC updater."""

    memory: WorkingMemory

    def detach(self) -> "PFCScratchState":
        """Return a detached copy of the internal carry state."""
        return PFCScratchState(memory=self.memory.detach())


@dataclass(frozen=True)
class PFCState:
    """Canonical public PFC state.

    The maintained working memory is exposed as a named slot workspace plus an
    external summary vector. Low-level recurrent tensors remain available only
    through ``scratch`` for internal continuation.
    """

    workspace: NamedWorkspace
    summary: Tensor
    scratch: PFCScratchState

    def __post_init__(self) -> None:
        if self.summary.ndim != 2:
            raise ValueError(f"PFC summary must have shape (B, D), got {tuple(self.summary.shape)}.")
        batch_size, slot_count, hidden_size = self.workspace.tokens.shape
        if tuple(self.summary.shape) != (batch_size, hidden_size):
            raise ValueError(
                f"PFC summary must match workspace batch/hidden dims {(batch_size, hidden_size)}, got {tuple(self.summary.shape)}."
            )

        theta = self.scratch.memory.z_H
        gamma = self.scratch.memory.z_L
        expected_shape = (batch_size, slot_count + 1, hidden_size)
        if tuple(theta.shape) != expected_shape:
            raise ValueError(f"PFC scratch z_H must have shape {expected_shape}, got {tuple(theta.shape)}.")
        if tuple(gamma.shape) != expected_shape:
            raise ValueError(f"PFC scratch z_L must have shape {expected_shape}, got {tuple(gamma.shape)}.")

    @property
    def spec(self) -> WorkspaceSpec:
        """Return the declared working-memory workspace layout."""
        return self.workspace.spec

    @property
    def tokens(self) -> Tensor:
        """Return the prefixed token view of the detached public state."""
        return torch.cat([self.summary.unsqueeze(1), self.workspace.tokens], dim=1)

    @property
    def control_token(self) -> Tensor:
        """Return the controller summary token as a batch-aligned slot tensor."""
        return self.summary

    def working_memory(self, controller_slot_name: str = "controller") -> NamedWorkspace:
        """Return the full public working memory with an explicit controller slot."""
        return _compose_controlled_workspace(controller_slot_name, self.summary, self.workspace)

    @property
    def memory(self) -> WorkingMemory:
        """Transitional alias for ``scratch.memory``.

        .. deprecated::
            Read ``state.pfc.scratch.memory`` directly.  This alias will be
            removed once ``EHCModelV1`` and ``EHCModelV3`` are migrated off
            the old field path.
        """
        return self.scratch.memory

    def detach(self) -> "PFCState":
        """Return a detached copy of the public state."""
        return PFCState(
            workspace=NamedWorkspace(spec=self.workspace.spec, tokens=self.workspace.tokens.detach()),
            summary=self.summary.detach(),
            scratch=self.scratch.detach(),
        )


@dataclass(frozen=True)
class PFCOutput:
    """Structured post-reasoning PFC output.

    ``state`` is the semantic post-reasoning state for the current step.
    Callers that want a bounded recurrent carry should explicitly store
    ``state.detach()``.
    """

    state: PFCState
    q_values: Tensor

    @property
    def tokens(self) -> Tensor:
        """Return the differentiable prefixed token view of the current step."""
        return self.state.tokens

    @property
    def workspace(self) -> NamedWorkspace:
        """Return the post-reasoning named workspace."""
        return self.state.workspace

    @property
    def summary(self) -> Tensor:
        """Return the semantic summary vector."""
        return self.state.summary

    @property
    def theta_summary(self) -> Tensor:
        """Backward-compatible alias for the semantic summary vector."""
        return self.state.summary

    def working_memory(self, controller_slot_name: str = "controller") -> NamedWorkspace:
        """Return the full public working memory with an explicit controller slot."""
        return self.state.working_memory(controller_slot_name)


# =================================================================================================
class PFCModel(nn.Module):
    """Prefrontal Cortex (PFC) reasoning module.

    This module implements a two-level recurrent reasoning process:
        - high level (theta-like) state update
        - low level (gamma-like) state update

    It also includes an auxiliary estimator (vmPFC analogue) that predicts values
    from the current working memory.

    Notes:
        The forward pass intentionally detaches the returned carry state while
        keeping the main outputs differentiable.
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, config: PFCSettings, device: Optional[Device]=None, dtype: Optional[Dtype]=None,
    ) -> None:  # fmt: skip
        super().__init__()
        self._config = config

        self.high_level = HighLvRModule(config.reasoning_h, device=device, dtype=dtype)
        self.low_level = LowLvRModule(config.reasoning_l, device=device, dtype=dtype)
        self.estimator = QValueEstimator(config.value_head, device=device, dtype=dtype)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size, device=device, dtype=dtype))
        self._default_workspace_spec = WorkspaceSpec(tuple(f"slot_{index}" for index in range(config.seq_length)))
        self.optimizer = None  # Placeholder for optimizer future dACC reward-based updates
        self.reset_parameters()

    @property
    def config(self) -> PFCSettings:
        """PFC module settings."""
        return self._config

    @property
    def default_workspace_spec(self) -> WorkspaceSpec:
        """Return the generic fallback workspace layout for tensor-only callers."""
        return self._default_workspace_spec

    def reset_parameters(  # ----------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Initialize parameters and buffers."""
        trunc_normal_init_(self.high_level.reset_vector, std=1)
        trunc_normal_init_(self.low_level.reset_vector, std=1)
        self.cls_token.data.zero_()  # Legacy parity: puzzle prefix initialized to zero

    def _validate_workspace_spec(self, spec: WorkspaceSpec) -> WorkspaceSpec:
        """Validate that a workspace spec matches the configured slot count."""
        if spec.size != self.config.seq_length:
            raise ValueError(f"workspace spec size must match configured seq_length {self.config.seq_length}, got {spec.size}.")
        return spec

    def _resolve_workspace_spec(
        self,
        *,
        state: Optional[PFCState] = None,
        workspace_spec: Optional[WorkspaceSpec] = None,
    ) -> WorkspaceSpec:
        """Resolve the workspace spec for semantic or tensor compatibility calls."""
        spec = workspace_spec or (state.spec if state is not None else self.default_workspace_spec)
        spec = self._validate_workspace_spec(spec)
        if state is not None and state.spec != spec:
            raise ValueError("state workspace spec does not match the requested workspace spec.")
        return spec

    def _build_state_from_memory(self, memory: WorkingMemory, spec: WorkspaceSpec, *, detach: bool) -> PFCState:
        """Project tensor-first memory into the canonical public state surface."""
        tokens = memory.z_H.detach() if detach else memory.z_H
        carry_memory = memory.detach() if detach else memory
        return PFCState(
            workspace=workspace_from_prefixed_tokens(tokens, spec, prefix_tokens=1),
            summary=tokens[:, 0],
            scratch=PFCScratchState(memory=carry_memory),
        )

    def init_state(  # ---------------------------------------------------------------------------
        self, batch_size: int, workspace_spec: Optional[WorkspaceSpec] = None,
    ) -> PFCState:  # fmt: skip
        """Create a fresh PFC recurrent state.

        Args:
            batch_size: Number of parallel sequences.
            workspace_spec: Optional named workspace layout. When omitted, a
                generic ``slot_i`` layout is synthesized for tensor callers.

        Returns:
            Initialized :class:`PFCState`.
        """
        # seq_length + 1: CLS prefix occupies position 0; cell tokens fill positions 1..S.
        spec = self._resolve_workspace_spec(workspace_spec=workspace_spec)
        memory = r.init_memory(batch_size, self.config.seq_length + 1, self.high_level, self.low_level)
        return self._build_state_from_memory(memory, spec, detach=False)

    def reset_state(  # --------------------------------------------------------------------------
        self, state: PFCState, reset_flag: Tensor,
    ) -> PFCState:  # fmt: skip
        """Selectively reset rows of the PFC state.

        Args:
            state: Current state.
            reset_flag: Boolean / 0-1 tensor of shape ``(B,)`` indicating which
                rows should be reset.

        Returns:
            New state with flagged rows reset.
        """
        memory = r.reset_memory(state.scratch.memory, reset_flag, self.high_level, self.low_level)
        return self._build_state_from_memory(memory, state.spec, detach=False)

    def _run_reasoning(  # ------------------------------------------------------------------------
        self, x: Tensor, state: PFCState, prefix_bias: Optional[Tensor] = None,
    ) -> tuple[WorkingMemory, Tensor]:  # fmt: skip
        """Run the internal tensor-first reasoning core and return live tensors."""
        total_steps = self.config.reasoning_h.n_cycles * (self.config.reasoning_l.n_cycles + 1)

        # Prepend CLS to cell embeddings: (B, S, D) → (B, S+1, D).
        # Reasoning modules are CLS-agnostic; they see a uniform sequence.
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if prefix_bias is not None:
            cls_token = cls_token + prefix_bias.unsqueeze(1).to(dtype=cls_token.dtype)
        x = torch.cat([cls_token, x], dim=1)

        # Forward iterations without grad for memory efficiency.
        # The final update at each level is executed with gradients below.
        memory_gen = r.reasoning_gen(x, state.scratch.memory, self.high_level, self.low_level)
        with torch.no_grad():
            for _ in range(total_steps - 2):  # Last 2 steps need gradients
                memory = next(memory_gen)

        # One-step grad: provide a training signal while keeping memory bounded.
        memory = next(memory_gen)  # N-2 step to update low-level state with gradients
        memory = next(memory_gen)  # N-1 step to update high-level state with gradients

        # Estimate Q-values from the updated state.
        q_estimation = self.estimator(memory.z_H, memory.z_L)

        return memory, q_estimation

    def step(  # ----------------------------------------------------------------------------------
        self, workspace: NamedWorkspace, state: Optional[PFCState] = None, prefix_bias: Optional[Tensor] = None,
    ) -> PFCOutput:  # fmt: skip
        """Run one semantic PFC step over a named working-memory workspace."""
        spec = self._resolve_workspace_spec(state=state, workspace_spec=workspace.spec)
        if int(workspace.tokens.shape[1]) != spec.size:
            raise ValueError(f"workspace token count must match spec size {spec.size}, got {tuple(workspace.tokens.shape)}.")

        state = state or self.init_state(batch_size=int(workspace.tokens.shape[0]), workspace_spec=spec)
        memory, q_values = self._run_reasoning(workspace.tokens, state, prefix_bias=prefix_bias)
        return PFCOutput(state=self._build_state_from_memory(memory, spec, detach=False), q_values=q_values)

    def step_tokens(  # ----------------------------------------------------------------------------
        self,
        x: Tensor,
        state: Optional[PFCState] = None,
        workspace_spec: Optional[WorkspaceSpec] = None,
        prefix_bias: Optional[Tensor] = None,
    ) -> PFCOutput:  # fmt: skip
        """Compatibility surface for tensor callers that still operate on slot tensors."""
        if x.ndim != 3:
            raise ValueError(f"PFC token inputs must have shape (B, S, D), got {tuple(x.shape)}.")
        spec = self._resolve_workspace_spec(state=state, workspace_spec=workspace_spec)
        if int(x.shape[1]) != spec.size:
            raise ValueError(f"PFC token input length must match workspace spec size {spec.size}, got {tuple(x.shape)}.")
        return self.step(NamedWorkspace(spec=spec, tokens=x), state=state, prefix_bias=prefix_bias)

    def forward_tokens(  # ------------------------------------------------------------------------
        self,
        x: Tensor,
        state: Optional[PFCState] = None,
        workspace_spec: Optional[WorkspaceSpec] = None,
        prefix_bias: Optional[Tensor] = None,
    ) -> PFCOutput:  # fmt: skip
        """Compatibility wrapper that preserves the old tensor-first entry point."""
        return self.step_tokens(x, state=state, workspace_spec=workspace_spec, prefix_bias=prefix_bias)

    def forward_workspace(  # ---------------------------------------------------------------------
        self, workspace: NamedWorkspace, state: Optional[PFCState] = None, prefix_bias: Optional[Tensor] = None,
    ) -> PFCOutput:  # fmt: skip
        """Compatibility wrapper that preserves the old named-workspace entry point."""
        return self.step(workspace, state=state, prefix_bias=prefix_bias)

    def forward(  # -------------------------------------------------------------------------------
        self, x: Tensor, state: Optional[PFCState] = None, prefix_bias: Optional[Tensor] = None,
    ) -> tuple[PFCState, Tensor, Tensor]:  # fmt: skip
        """Run recurrent reasoning and compute auxiliary value estimates.

        Args:
            x: Embedded inputs of shape ``(B, S, D)`` (cell tokens only; no CLS).
            state: Optional carry state. If ``None``, a fresh state is created.
            prefix_bias: Optional additive bias applied to the internal CLS token.

        Returns:
            ``(new_state, z_H, q_estimation)`` where:
                - ``new_state`` is detached carry state
                - ``z_H`` is the high-level activation tensor (includes CLS)
                - ``q_estimation`` are value/Q predictions from the estimator
        """
        output = self.step_tokens(x, state=state, prefix_bias=prefix_bias)
        return output.state.detach(), output.tokens, output.q_values


__all__ = [
    "ComposedWorkspaceWriter",
    "NamedWorkspace",
    "PFCModel",
    "PFCOutput",
    "PFCScratchState",
    "PFCSettings",
    "PFCState",
    "WorkspaceSlotWriter",
    "WorkspaceSpec",
    "workspace_from_prefixed_tokens",
]
