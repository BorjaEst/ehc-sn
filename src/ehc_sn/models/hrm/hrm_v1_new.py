"""HRM v1 core contracts over a task-agnostic PFC working-memory substrate."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch import nn

from ehc_sn.modules.pfc import PFCModel, PFCSettings, PFCState, WorkspaceSpec
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class ModelSettingsV1(BaseModel, extra="forbid"):
    """HRM v1 core settings.

    No task vocabulary, token embedding, positional encoding, or LM head fields
    belong here. Those remain adapter-owned.
    """

    pfc: PFCSettings = Field(
        ...,
        description="Settings for the recurrent PFC core.",
    )
    schema_slot_prefix: str = Field(
        default="schema",
        min_length=1,
        description="Prefix used for the internal schema-slot ontology.",
    )

    @property
    def num_schema_slots(self) -> int:
        """Return the number of schema slots owned by the HRM core."""
        return self.pfc.seq_length

    @property
    def schema_slot_names(self) -> tuple[str, ...]:
        """Return the canonical schema-slot names for the recurrent substrate."""
        return tuple(f"{self.schema_slot_prefix}_{index}" for index in range(self.num_schema_slots))

    @property
    def workspace_spec(self) -> WorkspaceSpec:
        """Return the schema-only workspace layout consumed by the core."""
        return WorkspaceSpec(names=self.schema_slot_names)


# =============================================================================
@dataclass(frozen=True)
class HRMInputV1:
    """Task-agnostic schema payload consumed by the HRM core.

    Attributes:
        schema_tokens: Schema slot tokens with shape ``(B, N, D)``.
        prefix_bias: Optional controller-prefix bias with shape ``(B, D)``.
    """

    schema_tokens: Tensor
    prefix_bias: Tensor | None = None

    @property
    def batch_size(self) -> int:
        """Return the leading batch size."""
        return int(self.schema_tokens.shape[0])

    @property
    def num_schema_slots(self) -> int:
        """Return the declared number of schema slots."""
        return int(self.schema_tokens.shape[1])

    @property
    def hidden_size(self) -> int:
        """Return the shared schema-token width."""
        return int(self.schema_tokens.shape[2])


# =============================================================================
@dataclass
class HRMStateV1(DetachMixin):
    """Recurrent carry owned by the HRM core."""

    pfc: PFCState


# =============================================================================
@dataclass(frozen=True)
class HRMOutputV1:
    """Architecture-native HRM output.

    Attributes:
        theta_summary: Controller summary vector with shape ``(B, D)``.
        schema_slots: Schema-slot bank with shape ``(B, N, D)``.
        q_logits: Control logits consumed by ACT-style controllers.
    """

    theta_summary: Tensor
    schema_slots: Tensor
    q_logits: Tensor


# =============================================================================
class HRModelV1(nn.Module):
    """Signature-only HRM v1 shell over the canonical recurrent core contract."""

    def __init__(  # ----------------------------------------------------------
        self,
        config: ModelSettingsV1,
        *,
        device: Device | None = None,
        dtype: Dtype | None = None,
    ) -> None:
        """Construct the HRM v1 core around the reusable PFC reasoning module."""
        super().__init__()
        self._config = config
        self.pfc = PFCModel(config.pfc, device=device, dtype=dtype)

    @property
    def config(self) -> ModelSettingsV1:
        """Return the immutable model settings used to parameterize the core."""
        return self._config

    def reset_parameters(  # --------------------------------------------------
        self,
    ) -> None:
        """Reset the reusable PFC core surfaces owned by HRM v1."""
        self.pfc.reset_parameters()

    def init_state(  # --------------------------------------------------------
        self,
        batch_size: int,
    ) -> HRMStateV1:
        """Create a fresh recurrent state for one batch."""
        return HRMStateV1(
            pfc=self.pfc.init_state(batch_size, self.config.workspace_spec),
        )

    def reset_state(  # -------------------------------------------------------
        self,
        reset_flag: Tensor,
        state: HRMStateV1,
    ) -> HRMStateV1:
        """Selectively reset rows of the recurrent state."""
        return HRMStateV1(
            pfc=self.pfc.reset_state(state.pfc, reset_flag),
        )

    def step(  # --------------------------------------------------------------
        self,
        payload: HRMInputV1,
        state: HRMStateV1 | None = None,
    ) -> tuple[HRMStateV1, HRMOutputV1]:
        """Run one HRM core step over schema-slot tokens and return state plus readouts.

        Args:
            payload: Task-agnostic schema-slot payload with shape ``(B, N, D)``.
            state: Optional recurrent carry from the previous step.

        Returns:
            ``(next_state, output)`` where ``next_state`` owns the recurrent slot
            substrate and ``output`` exposes the architecture-native readouts for
            the current step.
        """
        pfc_state = None if state is None else state.pfc
        pfc_output = self.pfc.step_tokens(
            payload.schema_tokens,
            state=pfc_state,
            workspace_spec=self.config.workspace_spec,
            prefix_bias=payload.prefix_bias,
        )
        next_state = HRMStateV1(pfc=pfc_output.state)
        output = HRMOutputV1(
            theta_summary=pfc_output.summary,
            schema_slots=pfc_output.workspace.tokens,
            q_logits=pfc_output.q_values,
        )

        return next_state, output

    def forward(  # -----------------------------------------------------------
        self,
        payload: HRMInputV1,
        state: HRMStateV1 | None = None,
    ) -> tuple[HRMStateV1, HRMOutputV1]:
        """Compatibility wrapper over :meth:`step` for module-call users."""
        return self.step(payload, state=state)


# =============================================================================
__all__ = ["ModelSettingsV1", "HRMInputV1", "HRMOutputV1", "HRMStateV1", "HRModelV1"]
